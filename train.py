import argparse
import itertools
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch.nn as nn
import torch.optim

torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed

import wandb
from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.train import train_rl
from src.models.rl.utils import State
from src.utils.logger_singleton import logger
from src.utils.train_config import *


def load_trainable_parameters(model: nn.Module, filepath: str):
    # Load the saved trainable parameters
    trainable_params = torch.load(filepath)

    # Update the model's state dictionary with the loaded trainable parameters
    model_state_dict = model.state_dict()

    for name, param in trainable_params.items():
        model_state_dict[name].copy_(param)


def save_model(model_name: str,
               model: nn.Module,
               optimizer: torch.optim,
               output_dir: str,
               iteration: int,
               val_loss: float):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Saving {model_name} model checkpoint to {output_dir}")
    model.unfreeze()

    torch.save({
        'iteration': iteration,
        'model_trainable_parameters': {name: param for name, param in model.state_dict().items() if
                                       param.requires_grad},
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss},
        f'{output_dir / model_name}.pth')


def save_checkpoint(iteration: int, val_loss: float):
    checkpoint_folder = f"checkpoint-{iteration}"
    output_dir = os.path.join(run_dir, checkpoint_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_model("ltm", ltm_model, ltm_optimizer, output_dir, iteration, val_loss)
    save_model("memory_model", memory_model, rl_optimizer, output_dir, iteration, val_loss)


def _evaluate(data: dict) -> torch.Tensor:
    batch_size, num_steps, _ = data['input_ids'].size()
    episode_loss = 0.

    memory_module = MemoryModule(agent.model.d_mem,
                                 agent.model.num_vectors,
                                 agent.model.dtype,
                                 agent.model.memory_type)
    memory_module.reset(batch_size)

    for step in tqdm(range(num_steps)):
        input_ids, attention_mask = _crop_batch(data['input_ids'][:, step, :].contiguous(),
                                                data['attention_mask'][:, step, :].contiguous())

        input_ids, attention_mask = input_ids.to(ltm_model.first_device), attention_mask.to(ltm_model.first_device)

        # Get high-level embeddings from LLM
        high_level_embeddings = ltm_model.get_embeddings(input_ids, attention_mask)

        # Prepare action for agent
        state = State(memory=memory_module.memory,
                      attention_mask=attention_mask,
                      embeddings=high_level_embeddings).to(agent.model.device)

        # Get new memory vectors and update memory
        action, _, _ = agent.act(state)
        action = action.to(torch.device('cpu'))

        # Compute loss and update
        loss = ltm_model.get_output(high_level_embeddings,
                                    attention_mask,
                                    memory_module.memory.to(ltm_model.second_device))

        # Update memory
        memory_module.update(action)

        episode_loss += loss.float().item()
        return episode_loss / num_steps


def evaluate(dataset):
    eval_dataloader = EpochDataloader(dataset,
                                      tokenizer,
                                      model_max_length=ltm_model.max_seq_length,
                                      max_sequence_len_in_batch=ltm_model.max_seq_length * 100,
                                      batch_size=args.trainer_args.batch_size, shuffle=True,
                                      cut_by_shortest_article=args.trainer_args.cut_by_shortest_article)
    ltm_model.freeze()
    memory_model.freeze()

    it, total_loss = 0, 0.
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if 0 < args.max_eval_steps <= i:
                break
            loss = _evaluate(batch)
            total_loss += loss
            it += 1

    ltm_model.unfreeze()
    memory_model.unfreeze()

    return total_loss / it


def _crop_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop_by_len = random.randint(2, input_ids.shape[1])
    return input_ids[:, :crop_by_len], attention_mask[:, :crop_by_len]


def train_ltm(ltm_model: LTM_GPT,
              agent: Agent,
              optim: torch.optim.SGD | torch.optim.Adam | torch.optim.AdamW,
              data: dict) -> float:
    ltm_loss = 0.
    batch_size, num_steps, _ = data['input_ids'].size()

    memory_module = MemoryModule(agent.model.d_mem,
                                 agent.model.num_vectors,
                                 agent.model.dtype,
                                 agent.model.memory_type)
    memory_module.reset(batch_size)

    for step in tqdm(range(num_steps)):
        optim.zero_grad()
        input_ids, attention_mask = _crop_batch(data['input_ids'][:, step, :].contiguous(),
                                                data['attention_mask'][:, step, :].contiguous())
        input_ids, attention_mask = input_ids.to(ltm_model.first_device), attention_mask.to(ltm_model.first_device)

        # Get high-level embeddings from LLM
        high_level_embeddings = ltm_model.get_embeddings(input_ids, attention_mask)

        # Prepare action for agent
        state = State(memory=memory_module.memory,
                      attention_mask=attention_mask,
                      embeddings=high_level_embeddings).to(agent.model.device)

        # Get new memory vectors and update memory
        with torch.no_grad():
            action, _, _ = agent.act(state)
            action = action.to(torch.device('cpu'))

        # Compute loss and update
        loss = ltm_model.get_output(high_level_embeddings,
                                    attention_mask,
                                    memory_module.memory.to(ltm_model.second_device))

        optim.step()
        torch.cuda.empty_cache()

        ltm_loss += loss.float().item()

        # Update memory
        memory_module.update(action)

    mean_loss_over_episode = ltm_loss / num_steps
    wandb.log({"LTM train iteration loss": mean_loss_over_episode})
    return mean_loss_over_episode


def iterative_training() -> None:
    """Iteratively train LTM and Memory models."""
    global train_cycle, epoch

    ltm_model_iterations = args.trainer_args.ltm_model_iterations
    memory_model_iterations = args.trainer_args.memory_model_iterations

    # First training iterations on LTM model
    ltm_model.unfreeze()
    agent.model.freeze()

    is_ltm_training = True
    ltm_iteration_count, memory_iteration_count = 0, 0
    ltm_loss, memory_model_loss = 0., 0.
    memory_model_success = 0

    train_dataloader_len = len(train_dataloader)
    batch_buffer, num_transitions_in_buffer = [], 0

    for batch in tqdm(train_dataloader, total=train_dataloader_len):
        if is_ltm_training:
            ltm_episode_loss = train_ltm(ltm_model, agent, ltm_optimizer, batch)
            ltm_loss += ltm_episode_loss
            ltm_iteration_count += 1

            if ltm_iteration_count >= ltm_model_iterations:
                ltm_iteration_count = 0
                is_ltm_training = False

                ltm_model.freeze()
                agent.model.unfreeze()
        else:
            bs, num_steps, _ = batch['input_ids'].shape
            cur_transitions = bs * (num_steps - 1)
            if cur_transitions + num_transitions_in_buffer < args.rl_params.min_transitions_per_update:
                batch_buffer.append(batch)
                num_transitions_in_buffer += cur_transitions
            else:
                batch_buffer.append(batch)
                memory_model_episode_loss = train_rl(batch_buffer, agent, rl_optimizer, ltm_model,
                                                     args)
                if memory_model_episode_loss is not None:
                    memory_model_loss += memory_model_episode_loss
                    memory_model_success += 1

                memory_iteration_count += 1
                num_transitions_in_buffer = 0

                if memory_iteration_count >= memory_model_iterations:
                    memory_iteration_count = 0
                    is_ltm_training = True
                    batch_buffer = []

                    # Logging and validation after cycle
                    train_cycle += 1
                    ltm_loss /= ltm_model_iterations
                    if memory_model_success:
                        memory_model_loss /= memory_model_success

                    val_loss = evaluate(val_dataset)

                    logger.info(f"""Training cycle {train_cycle} done.\nLTM train loss: {ltm_loss} \
                    \nLTM val loss: {val_loss}\nMemory model loss: {memory_model_loss}""")

                    wandb.log({"LTM train cycle loss": ltm_loss})
                    wandb.log({"Memory Model train cycle loss": memory_model_loss})
                    wandb.log({"LTM val loss": val_loss})

                    if not train_cycle % args.checkpoint_interval:
                        save_checkpoint(train_cycle, val_loss=val_loss)

                    ltm_loss, memory_model_loss, memory_model_success = 0., 0., 0

                    ltm_model.unfreeze()
                    agent.model.freeze()

    logger.info(f"Epoch {epoch} done!")


###############################################################################
# Parse arguments
###############################################################################

def init_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


args = init_arguments()
args = load_config(args.config)
set_seed(args.seed)
wandb.init(project="rugpt-memory", name=args.experiment_name, config=asdict(args))

# Create run directory and directory for saving checkpoints
run_name = f"run-{wandb.run.id}"
run_dir = os.path.join(args.checkpoint_dir, run_name)

###############################################################################
# Load data
###############################################################################
dataset_path = Path(args.content_dir) / 'data' / 'dataset'
dataset_path.resolve()
dataset_path = str(dataset_path)
train_dataset = WikiDataset(data_path=dataset_path, split='train')
val_dataset = WikiDataset(data_path=dataset_path, split='val')

###############################################################################
# Build the model
###############################################################################
dtype = torch.float16 if args.trainer_args.fp16 else torch.float32
ltm_model, tokenizer = load_ltm_model(args)

memory_model_device = torch.device(args.memory_model_params.device)
memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=torch.float32).to(memory_model_device)

###############################################################################
# Create optimizers
###############################################################################
ltm_model.unfreeze()
memory_model.unfreeze()

ltm_trainable_parameters = [p for p in ltm_model.parameters() if p.requires_grad]
memory_model_trainable_parameters = [p for p in memory_model.parameters() if p.requires_grad]

if args.trainer_args.optimizer.lower() == 'sgd':
    ltm_optimizer = torch.optim.SGD(ltm_trainable_parameters, lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.SGD(memory_model_trainable_parameters, lr=args.trainer_args.memory_model_learning_rate)
elif args.trainer_args.optimizer.lower() == 'adam':
    ltm_optimizer = torch.optim.Adam(ltm_trainable_parameters, lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.Adam(memory_model_trainable_parameters, lr=args.trainer_args.memory_model_learning_rate)
elif args.trainer_args.optimizer.lower() == 'adamw':
    ltm_optimizer = torch.optim.AdamW(ltm_trainable_parameters, lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.AdamW(memory_model_trainable_parameters, lr=args.trainer_args.memory_model_learning_rate)

logger.info('Starting the training process...')
logger.info(f'Number of trainable parameters (LTM) = {get_model_param_count(ltm_model, trainable_only=True)}')
logger.info(
    f'Number of trainable parameters (MemoryModel) = {get_model_param_count(memory_model, trainable_only=True)}')

###############################################################################
# Training code
###############################################################################

train_dataloader = EpochDataloader(train_dataset,
                                   tokenizer,
                                   model_max_length=ltm_model.max_seq_length,
                                   max_sequence_len_in_batch=ltm_model.max_seq_length * 100,
                                   batch_size=args.trainer_args.batch_size, shuffle=True,
                                   cut_by_shortest_article=args.trainer_args.cut_by_shortest_article)

agent = Agent(memory_model)

###############################################################################
# Train
###############################################################################

train_iteration = 0  # iteration is a number of batch in train dataset
train_cycle = 0  # cycle of training ltm and memory_model

try:
    for epoch in itertools.count(start=1):  # epoch == traverse over train dataset once
        iterative_training()
        if epoch == args.trainer_args.num_train_epochs:
            logger.info('-' * 100)
            logger.info('End of training')
            break
except KeyboardInterrupt:
    logger.info('-' * 100)
    logger.info('Exiting from training early')

###############################################################################
# Test
###############################################################################

# todo
ltm_model_path = ...
memory_model_path = ...

ltm_model, tokenizer = load_ltm_model(args)
ltm_model = load_trainable_parameters(ltm_model, ltm_model_path)
memory_model = MemoryModel(**asdict(args.memory_model_params))
memory_model = load_trainable_parameters(memory_model, memory_model_path)

# Run on test data
test_dataset = WikiDataset(data_path=dataset_path,
                           split='test')
test_loss = evaluate(test_dataset)

logger.info('=' * 100)
logger.info('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 100)
