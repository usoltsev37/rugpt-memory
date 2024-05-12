import itertools
import logging
import os
import pickle
import shutil
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_base_model import load_base_model
from src.models.load_ltm_model import load_ltm_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.models.rl.train import train_rl
from src.models.rl.utils import State
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_config import *
from src.utils.train_utils import (create_dir_with_name, create_name,
                                   crop_batch, init_arguments)


def _collate_fn(batch: list[str]) -> dict:
    batch = [item['text'] for item in batch]  # adjust according to dataset structure
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
    shortest_article_len = tokenized_batch["attention_mask"].sum(dim=-1).min()

    tokenized_batch["input_ids"] = tokenized_batch["input_ids"][:, :shortest_article_len]
    tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"][:, :shortest_article_len]

    add_tokens_num = (256 - shortest_article_len) % 256

    if add_tokens_num:
        if 256 - add_tokens_num == 1:
            tokenized_batch["input_ids"] = tokenized_batch["input_ids"][:, :-1]
            tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"][:, :-1]
        else:
            tokenized_batch["input_ids"] = F.pad(
                tokenized_batch["input_ids"], (0, add_tokens_num), "constant", tokenizer.pad_token_id
            ).long()
            tokenized_batch["attention_mask"] = F.pad(
                tokenized_batch["attention_mask"], (0, add_tokens_num), "constant", tokenizer.pad_token_id
            ).long()

    # Reshape to [batch_size, episode_len, len_seq_in_episode]
    tokenized_batch["input_ids"] = tokenized_batch["input_ids"].view(len(batch), -1, 256)
    tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].view(len(batch), -1, 256)
    return tokenized_batch

def create_dataloader(split):
    dataset = load_dataset("csebuetnlp/xlsum", "russian")[split]
    length_threshold = 4500  # You can adjust this value based on your needs
    filtered_dataset = dataset.filter(lambda example: len(example['text']) > length_threshold)
    dataloader = DataLoader(filtered_dataset, batch_size=1, collate_fn=_collate_fn)
    return dataloader

import copy


class Trainer:
    def __init__(
        self,
        ltm_model: LTM_GPT,
        ltm_optimizer,
        tokenizer,
        environment,
        reinforce,
        memory_module,
        train_dataset,
        val_dataset,
        args,
    ):
        self.args = args
        self.ltm_model, self.ltm_optimizer = ltm_model, ltm_optimizer
        self.memory_model = reinforce.agent.model
        self.env = environment
        self.reinforce = reinforce
        self.memory_module = memory_module

        self.train_dataset = train_dataset
        self.eval_dataset = val_dataset
        self.tokenizer = tokenizer

        self.eval_dataloader = self.get_eval_dataloader()
        self.train_dataloader = self.get_train_dataloader()
        # self.train_dataloader = create_dataloader("train")
        # self.eval_dataloader = create_dataloader("validation")

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
            step_length=self.args.ltm_params.step_length,
            batch_size=self.args.trainer_args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def get_train_dataloader(self):
        return EpochDataloader(
            self.train_dataset,
            self.tokenizer,
            step_length=self.args.ltm_params.step_length,
            batch_size=self.args.trainer_args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def _evaluate(self, data: dict) -> torch.Tensor:
        batch_size, num_steps, _ = data["input_ids"].size()
        episode_loss = 0.0

        self.memory_module.reset(batch_size)

        for step in range(num_steps):
            input_ids, attention_mask = crop_batch(
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
                action, _, _ = self.reinforce.agent.act(state)

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

        torch.save(
            {
                "cycle": self.cycle,
                "batch_step": self.batch_step,
                "model_parameters": self.ltm_model.state_dict(),
                "optimizer_state_dict": self.ltm_optimizer.state_dict(),
            },
            output_dir / "ltm.pt",
        )

        torch.save(
            {
                "cycle": self.cycle,
                "batch_step": self.batch_step,
                "model_parameters": self.memory_model.state_dict(),
                "optimizer_state_dict": self.reinforce.optim.state_dict(),
            },
            output_dir / "memory_model.pt",
        )

        torch.save(
            {
                "cycle": self.cycle,
                "batch_step": self.batch_step,
                "parameter": self.reinforce.alpha,
            },
            output_dir / "alpha.pt",
        )

    def save_checkpoint(self):
        global checkpoint_dir
        global saved_checkpoints_queue

        checkpoint_folder = f"checkpoint-{self.cycle}"
        output_dir = checkpoint_dir / checkpoint_folder
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
            input_ids, attention_mask = crop_batch(
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
                action, _, _ = self.reinforce.agent.act(state)

            # Update memory
            self.memory_module.update(action)

        return episode_loss / num_steps


    def train(self, train_from_checkpoint: bool = False):
        global epoch

        self.ltm_model.unfreeze()
        self.reinforce.agent.model.unfreeze()
        
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
        self.reinforce.agent.model.freeze()

        is_ltm_training = True
        ltm_iteration_count, memory_iteration_count = 0, 0
        ltm_loss, memory_model_loss, memory_model_reward = 0.0, 0.0, 0.0
        batch_buffer, num_transitions_in_buffer = [], 0
        self.rl_steps = 0

        # ltm_params = copy.deepcopy(self.ltm_model.state_dict())
        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            if train_from_checkpoint and batch < self.batch_to_start:
                continue

            if is_ltm_training:
                ltm_loss += self.train_ltm_on_episode(batch)
                
                # for n, p in self.ltm_model.named_parameters():
                #     old_p = ltm_params[n]
                #     if (old_p.data == p.data).all():
                #         print(f"{n} do not update!")
                ltm_iteration_count += 1
                if ltm_iteration_count >= ltm_model_iterations:
                    ltm_iteration_count = 0
                    is_ltm_training = False
                    self.ltm_model.freeze()
                    self.reinforce.agent.model.unfreeze()
            else:
                bs, num_steps, _ = batch["input_ids"].shape
                cur_transitions = bs * (num_steps - 1)
                if cur_transitions + num_transitions_in_buffer < self.args.rl_params.min_transitions_per_update:
                    if cur_transitions:
                        batch_buffer.append(batch)
                        num_transitions_in_buffer += cur_transitions
                else:
                    batch_buffer.append(batch)
                    mean_rl_loss, mean_rl_reward = train_rl(
                        batch_buffer, self.env, self.reinforce, self.args, tensorboard_writer, self.rl_steps
                    )
                    memory_model_loss += mean_rl_loss
                    memory_model_reward += mean_rl_reward

                    memory_iteration_count += 1
                    batch_buffer, num_transitions_in_buffer = [], 0
                    self.rl_steps += 1

                    if memory_iteration_count >= memory_model_iterations:
                        memory_iteration_count = 0
                        is_ltm_training = True

                        # Logging and validation after cycle
                        self.cycle += 1
                        ltm_loss /= ltm_model_iterations
                        memory_model_loss /= memory_model_iterations
                        memory_model_reward /= memory_model_iterations

                        val_loss = self.evaluate()

                        logger.info(
                            f"""Training cycle {self.cycle} done.\nLTM train loss: {ltm_loss} \
                        \nLTM val loss: {val_loss}\nMemory model loss: {memory_model_loss}\nMemory model reward: {memory_model_reward}"""
                        )

                        tensorboard_writer.add_scalar("Loss/ltm_val_loss", val_loss, self.cycle)
                        tensorboard_writer.add_scalar("Loss/ltm_train_cycle_loss", ltm_loss, self.cycle)
                        tensorboard_writer.add_scalar(
                            "Loss/memory_model_train_cycle_loss", memory_model_loss, self.cycle
                        )
                        tensorboard_writer.add_scalar(
                            "Rewars/memory_model_train_cycle_reward", memory_model_reward, self.cycle
                        )

                        if not self.cycle % self.args.checkpoint_interval:
                            self.save_checkpoint()

                        ltm_loss, memory_model_loss, memory_model_reward = 0.0, 0.0, 0.0

                        self.ltm_model.unfreeze()
                        self.reinforce.agent.model.freeze()

            self.batch_step += 1


if __name__ == "__main__":

    ###############################################################################
    # Parse arguments and create directories
    ###############################################################################
    args = init_arguments()
    args = load_config(args.config)
    set_seed(args.seed)

    name_of_experiment = create_name()

    # Checkpoints dir
    checkpoint_dir = Path(create_dir_with_name(args.checkpoint_dir, name_of_experiment)) / "runs"
    checkpoint_dir.mkdir(exist_ok=True)
    saved_checkpoints_queue = deque()

    # Logs dir
    log_dir = create_dir_with_name(args.log_dir, name_of_experiment)
    log_file = log_dir + "/train.log"

    # Save logs to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ColourFormatter())
    logger.addHandler(file_handler)

    # Tensorboard writer
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    # Save train config to log_dir
    content_dir = Path(args.content_dir).resolve()
    shutil.copy(content_dir / "configs" / "train_config.yml", log_dir)
    logger.info(f"Start training..configs/train_config.yml.")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Checkpoints dir: {checkpoint_dir}")
    logger.info(f"Log dir: {log_dir}")

    ###############################################################################
    # Build the model
    ###############################################################################

    ltm_model, tokenizer = load_ltm_model(args)
    # ltm_model.load_state_dict(state_dict)
    # for p in ltm_model.transform_matrix.parameters():
    #     p.requires_grad = False
    # logger.info("Reloaded weigths for pretrained LTM!")
    memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=ltm_model.dtype)
    checkpoint = '/home/akarpov/jbelova/rugpt-memory/checkpoints/pretrain_agent/memory_model:gen_embs:main_reward:transform_const:like_train_params/runs/checkpoint-340'
    state_dict = torch.load(checkpoint + "/memory_model.pt")["model_parameters"]
    memory_model.load_state_dict(state_dict)
    

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

    # Adaptive entropy coefficient
    alpha = torch.nn.Parameter(torch.tensor(args.rl_params.alpha_start), requires_grad=True)
    alpha_optimizer = torch.optim.SGD([alpha], lr=args.rl_params.alpha_lr)

    agent = Agent(memory_model)
    memory_module = MemoryModule(
        agent.model.d_mem,
        agent.model.num_vectors,
        agent.model.dtype,
        agent.model.memory_type,
    )
    env = LTMEnvironment(
        ltm_model, memory_module, num_prefixes_for_reward_calc=args.rl_params.num_prefixes_for_reward_calc
    )
    reinforce = REINFORCE(
        agent=agent, optimizer=rl_optimizer, train_config=args.rl_params, alpha=alpha, alpha_optimizer=alpha_optimizer
    )

    ###############################################################################
    # Load data
    ###############################################################################
    dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
    train_dataset = WikiDataset(data_path=str(dataset_path), split="train")
    val_dataset = WikiDataset(data_path=str(dataset_path), split="val")

    ###############################################################################
    # Train
    ###############################################################################

    trainer = Trainer(
        ltm_model=ltm_model,
        ltm_optimizer=ltm_optimizer,
        tokenizer=tokenizer,
        environment=env,
        reinforce=reinforce,
        memory_module=memory_module,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
    )

    try:
        for epoch in range(args.trainer_args.num_train_epochs):  # epoch == traverse over train dataset once
            trainer.train()
            logger.info("-" * 100)
        logger.info("End of training.")
    except (KeyboardInterrupt, Exception) as e:
        logger.info("-" * 100)
        logger.info("Exiting from training early")
        logger.error(e)
