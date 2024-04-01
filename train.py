import argparse
import random
import numpy as np
import time
import itertools
import math

from src.models.load_ltm_model import load_ltm_model
from src.utils.logger_singleton import logger
from data.wiki_dataset import WikiDataset
from data.wiki_dataloader import EpochDataloader
from transformers.trainer_pt_utils import get_model_param_count

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory_model import MemoryModel
from src.models.memory_model.memory import MemoryModule

from src.utils.train_config import *

from rl.agent import Agent
from rl.utils import State
from rl.train import train_rl


###############################################################################
# Parse arguments
###############################################################################


def init_seed(seed: int):
    """Initialize random seeds for reproducibility."""
    logger.info(f"Using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


args = init_arguments()
init_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
train_dataset = WikiDataset(data_path=args.dataset_path,
                            n_context=args.n_context,
                            split='train')

###############################################################################
# Build the model
###############################################################################
ltm_model, tokenizer = load_ltm_model(args.ltm_model_config)
memory_model = MemoryModel(**args.memory_model_params)

# Accelerate logic

# Optimizer
if args.trainer_params.optim.lower() == 'sgd':
    optimizer = torch.optim.SGD(ltm_model.parameters(), lr=args.lr)
elif args.trainer_params.optim.lower() == 'adam':
    optimizer = torch.optim.Adam(ltm_model.parameters(), lr=args.lr)
elif args.trainer_params.optim.lower() == 'adamw':
    optimizer = torch.optim.AdamW(ltm_model.parameters(), lr=args.lr)

logger.info('Starting the training process 🎉')
logger.info(f'Number of trainable parameters (LTM) = {get_model_param_count(ltm_model, trainable_only=True)}')
logger.info(
    f'Number of trainable parameters (MemoryModel) = {get_model_param_count(memory_model, trainable_only=True)}')


###############################################################################
# Training code
###############################################################################

def _evaluate(data) -> torch.Tensor:
    pass


def evaluate(dataloader: torch.utuls.data.Dataloader):
    # Turn on evaluation mode which disables dropout.
    ltm_model.freeze()
    memory_model.freeze()

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            loss = _evaluate(data)
            total_loss += loss.float().item()

    ltm_model.unfreeze()
    memory_model.unfreeze()

    return total_loss


def train_ltm(ltm_model: LTM_GPT,
              agent: Agent,
              optim: torch.optim.optimizer,
              batch: torch.Tensor) -> None:
    global train_loss, log_start_time

    memory = MemoryModule(agent.model.d_mem,
                          agent.model.num_vectors,
                          agent.model.memory_type)
    memory.reset(batch.shape[0])

    for it, chunk in enumerate(torch.split(batch, split_size_or_sections=ltm_model.max_seq_len, dim=1)):
        optim.zero_grad()
        high_level_embeddings = ltm_model.get_high_level_embeddings(chunk)
        state = State(high_level_embeddings, memory)
        action, _, _ = agent.act(state)
        new_memory = memory.update(action)
        loss = ltm_model.get_output(high_level_embeddings, new_memory)

        loss.backward()
        optim.step()

        train_loss += loss.float().item()

    logger.info('-' * 100)
    cur_loss = train_loss / (it + 1)
    # log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
    #           '| ms/batch {:5.2f} | loss {:5.2f}'.format(
    #     epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
    #                        elapsed * 1000 / args.log_interval, cur_loss)
    # log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
    #
    # logger.info(log_str)
    train_loss = 0
    log_start_time = time.time()


def iterative_training(ltm_model: LTM_GPT,
                       agent: Agent,
                       dataloader,
                       train_config: TrainerParams) -> None:
    """Iteratively train LTM and Memory models."""
    global train_step, ltm_optimizer
    ltm_model_iterations = train_config.ltm_model_iterations
    memory_model_iterations = args.rl_params.iterations * args.rl_params.min_transitions_per_update
    rl_data = []

    # First training iterations on LTM model
    is_ltm_training = True
    ltm_iteration_count = 0
    memory_iteration_count = 0

    for iteration, batch in enumerate(dataloader):
        if is_ltm_training:
            train_ltm(ltm_model, agent, ltm_optimizer, batch)
            ltm_iteration_count += 1
            if ltm_iteration_count >= ltm_model_iterations:
                ltm_iteration_count = 0
                is_ltm_training = False
                ltm_model.freeze()
                agent.memory_model.unfreeze()
        else:
            if memory_iteration_count >= memory_model_iterations:
                train_rl(rl_data, agent, ltm_model, args.rl_params)
                memory_iteration_count = 0
                is_ltm_training = True
                ltm_model.unfreeze()
                agent.model.freeze()
                rl_data = []
            else:
                rl_data.append(batch)
                memory_iteration_count += math.ceil(batch.shape[1] / args.ltm_params.n_ctx) * batch.shape[0]

        train_step += 1


###############################################################################
# Load data and models
###############################################################################


ltm_model, tokenizer = load_ltm_model(args.ltm_model_config)
memory_model = MemoryModel(**args.memory_model_params)

train_dataloader = EpochDataloader(train_dataset, tokenizer, args.tokenizer_params)
agent = Agent(memory_model)

###############################################################################
# Optimizers
###############################################################################
ltm_optimizer = ...
memory_model_optimizer = ...

###############################################################################
# Train
###############################################################################


train_step = 0
train_loss = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        iterative_training(...)
        if train_step == args.trainer_params.num_train_epochs:
            logger.info('-' * 100)
            logger.info('End of training')
            break
except KeyboardInterrupt:
    logger.info('-' * 100)
    logger.info('Exiting from training early')

# Load the best saved model
ltm_model, tokenizer = load_ltm_model(args.ltm_model_config)
memory_model = MemoryModel(**args.memory_model_params)

# Run on test data
test_dataset = WikiDataset(data_path=args.dataset_path,
                           n_context=args.n_context,
                           split='test')

test_dataloader = EpochDataloader(test_dataset, tokenizer, args.tokenizer_params)
test_loss = evaluate(test_dataloader)
logger.info('=' * 100)
logger.info('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 100)
