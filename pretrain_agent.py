import logging
import os
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.envs import PretrainEnv
from src.models.rl.reinforce import REINFORCE
from src.models.rl.train import compute_rewards, sample_episodes
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.pretrain_agent_config import load_config
from src.utils.train_utils import create_dir_with_name, init_arguments


def save_models(output_dir: Path) -> None:
    logger.info(f"Saving models checkpoints to {output_dir}")

    torch.save(
        {"model_parameters": reinforce.agent.model.state_dict()},
        output_dir / "memory_model.pt",
    )


def save_checkpoint(cur_iter):
    global checkpoint_dir
    global saved_checkpoints_queue

    checkpoint_folder = f"checkpoint-{cur_iter}"
    output_dir = checkpoint_dir / checkpoint_folder
    saved_checkpoints_queue.append(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_models(output_dir)

    if len(saved_checkpoints_queue) > args.max_checkpoints:
        oldest_checkpoint = saved_checkpoints_queue.popleft()
        shutil.rmtree(oldest_checkpoint)


def sample_episodes(env: PretrainEnv, reinforce: REINFORCE, data: dict, train_config) -> list[tuple]:
    reinforce.agent.model.eval()
    state = env.reset(data)
    done = False
    trajectories = []
    with torch.no_grad():
        while not done:
            action, log_proba, distr = reinforce.act(state)  # cpu
            logger.info(f"Positions: {action.positions[0]}")
            logger.info(f"Mean: {action.memory_vectors[0, action.positions[0]].mean():.4f}")
            logger.info(f"Std: {action.memory_vectors[0, action.positions[0]].std():.4f}")
            next_state, reward, done = env.step(action)  # cpu
            logger.info(f"Reward: {reward[0]:.4f}")
            trajectories.append([state, action, reward, log_proba, distr])
            state = next_state
            logger.info("-" * 50)

    logger.info(">" * 50)
    reinforce.agent.model.train()
    return compute_rewards(trajectories, train_config.gamma)  # There is no reward for the last step


def pretrain(env, reinforce, args, train_dataloader):
    logger.info("Start Memory Model pretraining...")

    # Train only memory model
    ltm_model.freeze()
    memory_model.unfreeze()

    iterations_num = (
        args.pretrain_params.iterations
        if args.pretrain_params.iterations <= len(train_dataloader)
        else len(train_dataloader)
    )

    batch_buffer, num_transitions_in_buffer = [], 0
    transitions_reward = []

    cur_transitions = args.pretrain_params.episode_max_steps * args.trainer_args.batch_size

    for cur_iter, batch in enumerate(tqdm(train_dataloader, total=iterations_num)):
        if cur_transitions + num_transitions_in_buffer < args.rl_params.min_transitions_per_update:
            batch_buffer.append(batch)
            num_transitions_in_buffer += cur_transitions
        else:
            batch_buffer.append(batch)
            transitions = []
            for batch in batch_buffer:
                new_transitions, mean_reward = sample_episodes(env, reinforce, batch, args.rl_params)
                transitions_reward.append(mean_reward)
                transitions.extend(new_transitions)
            mean_loss = reinforce.update(transitions, tensorboard_writer, cur_iter)
            tensorboard_writer.add_scalar("Loss/memory_model", mean_loss, cur_iter)
            tensorboard_writer.add_scalar("Reward/memory_model", np.mean(transitions_reward), cur_iter)

            batch_buffer, num_transitions_in_buffer = [], 0
            transitions_reward = []

        if not cur_iter % args.checkpoint_interval:
            save_checkpoint(cur_iter)

        if cur_iter == iterations_num:
            break

    logger.info("Memory model pretraining done!")


###############################################################################
# Parse args, create dirs
###############################################################################

args = init_arguments()
args = load_config(args.config)
set_seed(args.seed)

# Checkpoints dir
checkpoint_dir = Path(create_dir_with_name(args.checkpoint_dir, args.experiment_name)) / "runs"
checkpoint_dir.mkdir(exist_ok=True)
saved_checkpoints_queue = deque()

# Logs dir
log_dir = create_dir_with_name(args.log_dir, args.experiment_name)
log_file = log_dir + "/train.log"

# Save logs to file
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(ColourFormatter())
logger.addHandler(file_handler)

# Tensorboard writer
tensorboard_writer = SummaryWriter(log_dir=log_dir)

# Save train config to log_dir
content_dir = Path(args.content_dir).resolve()
shutil.copy(content_dir / "configs" / "pretrain_agent_config.yml", log_dir)

logger.info(f"Pretraining agent. We teach the agent to generate vectors similar to embeddings.")
logger.info(f"Experiment name: {args.experiment_name}")
logger.info(f"Checkpoints dir: {checkpoint_dir}")
logger.info(f"Log dir: {log_dir}")


###############################################################################
# Build the model and set up environment
###############################################################################

ltm_model, tokenizer = load_ltm_model(args)

memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=torch.float32)

# Memory Model optimizer
rl_optimizer = torch.optim.AdamW(
    memory_model.parameters(),
    lr=args.pretrain_params.lr,
)

# Adaptive entropy coef
alpha = torch.nn.Parameter(torch.tensor(args.rl_params.alpha_start), requires_grad=True)
alpha_optimizer = torch.optim.SGD([alpha], lr=args.rl_params.alpha_lr)

agent = Agent(memory_model)
memory_module = MemoryModule(
    agent.model.d_mem,
    agent.model.num_vectors,
    agent.model.dtype,
    agent.model.memory_type,
)
env = PretrainEnv(ltm_model, memory_module, episode_max_steps=args.pretrain_params.episode_max_steps, args=args)
reinforce = REINFORCE(
    agent=agent, optimizer=rl_optimizer, train_config=args.rl_params, alpha=alpha, alpha_optimizer=alpha_optimizer
)

###############################################################################
# Load data
###############################################################################
dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
train_dataset = WikiDataset(data_path=str(dataset_path), split="train")

train_dataloader = EpochDataloader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    step_length=args.ltm_params.step_length,
    batch_size=args.trainer_args.batch_size,
    pin_memory=True,
)

###############################################################################
# Load data
###############################################################################
pretrain(env, reinforce, args, train_dataloader)

try:
    pretrain(env, reinforce, args, train_dataloader)
except (KeyboardInterrupt, Exception) as e:
    logger.info("-" * 100)
    logger.info("Exiting from training early")
    logger.error(e)
