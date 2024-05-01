import argparse
import datetime
import itertools
import logging
import os
import pickle
import random
import shutil
import socket

# torch.autograd.set_detect_anomaly(True)
import subprocess
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed
from transformers.utils import logging as tr_logging
from src.utils.logger_singleton import ColourFormatter

tr_logging.set_verbosity_error()

from torch.utils.tensorboard import SummaryWriter

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

# torch._logging.set_logs(dynamo=logging.DEBUG)
# torch._dynamo.config.verbose = True

import faulthandler

faulthandler.enable()


def create_dir_with_name(base_dir: str, name: str):

    # Construct the directory path
    dir = os.path.join(base_dir, name)

    # Create the directory, including all intermediate directories
    os.makedirs(dir, exist_ok=True)

    return dir


def create_name():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Get the current hostname
    hostname = socket.gethostname()
    return f"{current_time}_{hostname}"


def _crop_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop_by_len = random.randint(2, input_ids.shape[1])
    return input_ids[:, :crop_by_len], attention_mask[:, :crop_by_len]


class Trainer:
    def __init__(
        self,
        ltm_model: LTM_GPT,
        ltm_optimizer,
        memory_model: MemoryModel,
        memory_model_optimizer,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
    ):
        self.args = args
        self.ltm_model, self.ltm_optimizer = ltm_model, ltm_optimizer

        self.memory_model, self.memory_model_optimizer = memory_model, memory_model_optimizer

        self.agent = Agent(self.memory_model)
        self.memory_module = MemoryModule(
            self.agent.model.d_mem,
            self.agent.model.num_vectors,
            self.agent.model.dtype,
            self.agent.model.memory_type,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.eval_dataloader = self.get_eval_dataloader()
        self.train_dataloader = self.get_train_dataloader()
        
        self.ltm_clip_grad_norm = args.trainer_args.ltm_clip_grad_norm

    def load_checkpoint(self, checkpoint_path: str):

        # Load LTM model and variables
        with open(checkpoint_path + "/ltm.pkl", "rb") as f:
            checkpoint = pickle.load(f)
            self.cycle = checkpoint["cycle"]
            self.batch_to_start = checkpoint["batch_step"] + 1
            self.batch_step = self.batch_to_start + 1
            self.ltm_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.ltm_model.load_state_dict(checkpoint["model_parameters"])

        # Load Memory Model
        with open(checkpoint_path + "/memory_model.pkl", "rb") as f:
            checkpoint = pickle.load(f)
            self.memory_model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.memory_model.load_state_dict(checkpoint["model_parameters"])
            self.memory_model.to(self.memory_model.device)

    def get_eval_dataloader(self):
        return EpochDataloader(
            self.eval_dataset,
            self.tokenizer,
            model_max_length=self.args.trainer_args.step_length,
            max_sequence_len_in_batch=self.args.trainer_args.step_length * 100,
            batch_size=self.args.trainer_args.batch_size,
            shuffle=False,
            # num_workers=2,
            pin_memory=True,
        )

    def get_train_dataloader(self):
        return EpochDataloader(
            self.train_dataset,
            self.tokenizer,
            model_max_length=self.args.trainer_args.step_length,
            max_sequence_len_in_batch=self.args.trainer_args.step_length * 100,
            batch_size=self.args.trainer_args.batch_size,
            shuffle=True,
            # num_workers=2,
            pin_memory=True,
        )

    def _evaluate(self, data: dict) -> torch.Tensor:
        batch_size, num_steps, _ = data["input_ids"].size()
        episode_loss = 0.0

        self.memory_module.reset(batch_size)

        for step in range(num_steps):
            input_ids, attention_mask = _crop_batch(
                data["input_ids"][:, step, :].contiguous(),
                data["attention_mask"][:, step, :].contiguous(),
            )

            loss, embeddings = self.ltm_model(input_ids, attention_mask, self.memory_module.memory)

            episode_loss += loss.item()

            # Prepare action for agent
            state = State(
                memory=self.memory_module.memory,
                attention_mask=attention_mask,
                embeddings=embeddings,
            )

            # Get new memory vectors and update memory
            with torch.no_grad():
                action, _, _ = self.agent.act(state)

            # Update memory
            self.memory_module.update(action)

        return episode_loss / num_steps

    def evaluate(self):
        self.ltm_model.freeze()
        self.memory_model.freeze()
        torch.cuda.empty_cache()

        it, total_loss = 0, 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                if 0 < self.args.max_eval_steps <= i:
                    break
                loss = self._evaluate(batch)
                total_loss += loss
                it += 1

        return total_loss / it

    def save_models(self, output_dir: Path) -> None:
        logger.info(f"Saving models checkpoints to {output_dir}")

        ltm_state_dict = {k: v.cpu() for k, v in self.ltm_model.state_dict().items()}
        memory_model_state_dict = {k: v.cpu() for k, v in self.memory_model.state_dict().items()}

        with open(str(output_dir / "ltm.pkl"), "wb") as f:
            pickle.dump(
                {
                    "cycle": self.cycle,
                    "batch_step": self.batch_step,
                    "model_parameters": ltm_state_dict,
                    "optimizer_state_dict": self.ltm_optimizer.state_dict(),
                },
                f,
            )

        with open(str(output_dir / "memory_model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "cycle": self.cycle,
                    "batch_step": self.batch_step,
                    "model_parameters": memory_model_state_dict,
                    "optimizer_state_dict": self.memory_model_optimizer.state_dict(),
                },
                f,
            )

        # torch.save(
        #     {
        #         "cycle": self.cycle,
        #         "batch_step": self.batch_step,
        #         "model_parameters": self.ltm_model.state_dict(),
        #         "optimizer_state_dict": self.ltm_optimizer.state_dict(),
        #     },
        # output_dir / "ltm.pt"
        # )

        # torch.save(
        #     {
        #         "cycle": self.cycle,
        #         "batch_step": self.batch_step,
        #         "model_parameters": self.memory_model.state_dict(),
        #         "optimizer_state_dict": self.memory_model_optimizer.state_dict(),
        #     },
        #     output_dir / "memory_model.pt",
        # )

    def save_checkpoint(self):
        global run_dir
        global saved_checkpoints_queue

        checkpoint_folder = f"checkpoint-{self.cycle}"
        output_dir = run_dir / checkpoint_folder
        saved_checkpoints_queue.append(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.save_models(output_dir)

        if len(saved_checkpoints_queue) > self.args.max_checkpoints:
            oldest_checkpoint = saved_checkpoints_queue.popleft()
            shutil.rmtree(oldest_checkpoint)

    def train_ltm_on_episode(self, data: dict) -> float:

        episode_loss = 0.0
        batch_size, num_steps, _ = data["input_ids"].size()

        self.memory_module.reset(batch_size)

        for step in range(num_steps):
            input_ids, attention_mask = _crop_batch(
                data["input_ids"][:, step, :].contiguous(),
                data["attention_mask"][:, step, :].contiguous(),
            )

            loss, embeddings = self.ltm_model(input_ids, attention_mask, self.memory_module.memory)

            loss.backward()
            
            nn.utils.clip_grad_norm_(self.ltm_model.parameters(), self.ltm_clip_grad_norm)
            
            self.ltm_optimizer.step()
            self.ltm_optimizer.zero_grad()

            episode_loss += loss.item()

            # Prepare action for agent
            state = State(
                memory=self.memory_module.memory,
                attention_mask=attention_mask,
                embeddings=embeddings,
            )

            # Get new memory vectors and update memory
            with torch.no_grad():
                action, _, _ = self.agent.act(state)

            # Update memory
            self.memory_module.update(action)

        return episode_loss / num_steps

    def train(self, train_from_checkpoint: bool = False):
        global epoch

        logger.info("Starting the training process...")
        logger.info(
            f"Number of trainable parameters (LTM) = {get_model_param_count(self.ltm_model, trainable_only=True)}"
        )
        logger.info(
            f"Number of trainable parameters (MemoryModel) = {get_model_param_count(self.memory_model, trainable_only=True)}"
        )
        if not train_from_checkpoint:
            self.cycle, self.batch_step = 0, 0

        ltm_model_iterations = self.args.trainer_args.ltm_model_iterations
        memory_model_iterations = self.args.trainer_args.memory_model_iterations

        # First training iterations on LTM model
        self.ltm_model.unfreeze()
        self.agent.model.freeze()

        is_ltm_training = True
        ltm_iteration_count, memory_iteration_count = 0, 0
        ltm_loss, memory_model_loss = 0.0, 0.0
        batch_buffer, num_transitions_in_buffer = [], 0

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            if train_from_checkpoint and batch < self.batch_to_start:
                continue

            if is_ltm_training:
                ltm_loss += self.train_ltm_on_episode(batch)
                ltm_iteration_count += 1
                if ltm_iteration_count >= ltm_model_iterations:
                    ltm_iteration_count = 0
                    is_ltm_training = False
                    self.ltm_model.freeze()
                    self.agent.model.unfreeze()
            else:
                bs, num_steps, _ = batch["input_ids"].shape
                cur_transitions = bs * (num_steps - 1)
                if cur_transitions + num_transitions_in_buffer < self.args.rl_params.min_transitions_per_update:
                    if cur_transitions:
                        batch_buffer.append(batch)
                        num_transitions_in_buffer += cur_transitions
                else:
                    batch_buffer.append(batch)
                    memory_model_loss += train_rl(
                        batch_buffer,
                        self.agent,
                        self.memory_model_optimizer,
                        self.ltm_model,
                        self.memory_module,
                        self.args,
                        logger,
                    )

                    memory_iteration_count += 1
                    batch_buffer, num_transitions_in_buffer = [], 0

                    if memory_iteration_count >= memory_model_iterations:
                        memory_iteration_count = 0
                        is_ltm_training = True

                        # Logging and validation after cycle
                        self.cycle += 1
                        ltm_loss /= ltm_model_iterations
                        memory_model_loss /= memory_model_iterations

                        val_loss = self.evaluate()

                        logger.info(
                            f"""Training cycle {self.cycle} done.\nLTM train loss: {ltm_loss} \
                        \nLTM val loss: {val_loss}\nMemory model loss: {memory_model_loss}"""
                        )

                        tensorboard_writer.add_scalar("Loss/ltm_val_loss", val_loss, self.cycle)
                        tensorboard_writer.add_scalar("Loss/ltm_train_cycle_loss", ltm_loss, self.cycle)
                        tensorboard_writer.add_scalar(
                            "Loss/memory_model_train_cycle_loss", memory_model_loss, self.cycle
                        )

                        if not self.cycle % self.args.checkpoint_interval:
                            self.save_checkpoint()

                        ltm_loss, memory_model_loss = 0.0, 0.0

                        self.ltm_model.unfreeze()
                        self.agent.model.freeze()

            self.batch_step += 1


def init_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


def setup_logger_process():
    global log_dir
    return subprocess.Popen(["python3", "log_mem_usage.py", log_dir + "/memory.log"])


# global tensorboard_writer, saved_checkpoints_queue, run_dir, logger_process
###############################################################################
# Parse arguments and create directories
###############################################################################
args = init_arguments()
args = load_config(args.config)
set_seed(args.seed)

name = create_name()

run_dir = Path(create_dir_with_name(args.checkpoint_dir, name)) / "runs"
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

log_dir = create_dir_with_name(args.log_dir, name)
log_file = log_dir + "/train.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(ColourFormatter())
logger.addHandler(file_handler)

tensorboard_writer = SummaryWriter(log_dir=log_dir)

# PROCESS WILL CLOSE AFTER MAIN COMPLETION
logger_process = setup_logger_process()

# Save train config to log_dir
content_dir = Path(args.content_dir).resolve()
shutil.copy(content_dir / "configs" / "train_config.yml", log_dir)

logger.info(f"Experiment name: {args.experiment_name}")
logger.info(f"Checkpoints dir: {run_dir}")
logger.info(f"Log dir: {log_dir}")

saved_checkpoints_queue = deque()

# try:
###############################################################################
# Load data
###############################################################################
dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
train_dataset = WikiDataset(data_path=str(dataset_path), split="train")
val_dataset = WikiDataset(data_path=str(dataset_path), split="val")

###############################################################################
# Build the model
###############################################################################

ltm_model, tokenizer = load_ltm_model(args)
memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=ltm_model.dtype)

###############################################################################
# Create optimizers
###############################################################################

if args.trainer_args.optimizer.lower() == "sgd":
    ltm_optimizer = torch.optim.SGD(ltm_model.parameters(), lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.SGD(
        memory_model.parameters(),
        lr=args.trainer_args.memory_model_learning_rate,
    )
elif args.trainer_args.optimizer.lower() == "adam":
    ltm_optimizer = torch.optim.Adam(ltm_model.parameters(), lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.Adam(
        memory_model.parameters(),
        lr=args.trainer_args.memory_model_learning_rate,
    )
elif args.trainer_args.optimizer.lower() == "adamw":
    ltm_optimizer = torch.optim.AdamW(ltm_model.parameters(), lr=args.trainer_args.ltm_learning_rate)
    rl_optimizer = torch.optim.AdamW(
        memory_model.parameters(),
        lr=args.trainer_args.memory_model_learning_rate,
    )

###############################################################################
# Train
###############################################################################

trainer = Trainer(ltm_model, ltm_optimizer, memory_model, rl_optimizer, args, train_dataset, val_dataset, tokenizer)

try:
    for epoch in itertools.count(start=1):  # epoch == traverse over train dataset once
        trainer.train()
        if epoch == args.trainer_args.num_train_epochs:
            logger.info("-" * 100)
            logger.info("End of training.")
            break
except (KeyboardInterrupt, Exception) as e:
    logger.info("-" * 100)
    logger.info("Exiting from training early")
    logger.error(e)
    logger_process.kill()
