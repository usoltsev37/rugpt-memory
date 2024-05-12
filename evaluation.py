import itertools
import logging
import math
import os
import pickle
import shutil
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
from src.models.load_ltm_model import load_ltm_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.models.rl.train import train_rl
from src.models.rl.utils import State
from src.utils.evaluation_config import *
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_utils import (create_dir_with_name, create_name,
                                   crop_batch, init_arguments)


def _evaluate(data: dict) -> torch.Tensor:
    batch_size, num_steps, _ = data["input_ids"].size()
    episode_loss = 0.0

    memory_module.reset(batch_size)

    for step in range(num_steps):
        input_ids, attention_mask = (data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous())

        loss, embeddings = ltm_model(input_ids, attention_mask, memory_module.memory)

        episode_loss += loss.item()

        # Prepare action for agent
        state = State(
            memory=memory_module.memory,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )

        # Get new memory vectors and update memory
        with torch.no_grad():
            action, _, _ = agent.act(state)

        # Update memory
        memory_module.update(action)

    return episode_loss / num_steps
    

def format_log(loss: float, split: str) -> str:
    log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str

def evaluate():
    it, total_loss = 0, 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=5000):
            loss = _evaluate(batch)
            total_loss += loss
            it += 1
            
            # if i == 5000:
            #     break
            
    return total_loss / it
   
   

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

def create_dataloader():
    dataset = load_dataset("csebuetnlp/xlsum", "russian")["test"]
    length_threshold = 5000  # You can adjust this value based on your needs
    filtered_dataset = dataset.filter(lambda example: len(example['text']) > length_threshold)
    dataloader = DataLoader(filtered_dataset, batch_size=1, collate_fn=_collate_fn)
    return dataloader
 
if __name__ == "__main__":
    ###############################################################################
    # Parse arguments and create directories
    ###############################################################################
    args = init_arguments()
    args = load_config(args.config)
    set_seed(args.seed)

    name_of_experiment = create_name()

    # Checkpoint dir
    checkpoint_dir = Path(args.pretrained_model_path).resolve()

    # Logs dir
    log_dir = create_dir_with_name(args.log_dir, name_of_experiment)
    log_file = log_dir + "/eval.log"

    # Save logs to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ColourFormatter())
    logger.addHandler(file_handler)

    # Tensorboard writer
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    # Save train config to log_dir
    content_dir = Path(args.content_dir).resolve()
    shutil.copy(content_dir / "configs" / "eval_config.yml", log_dir)
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Log dir: {log_dir}")

    ###############################################################################
    # Build the model
    ###############################################################################

    ltm_model, tokenizer = load_ltm_model(args)
    ltm_checkpoint = torch.load(checkpoint_dir / "ltm.pt")["model_parameters"]
    ltm_model.load_state_dict(ltm_checkpoint)

    memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=ltm_model.dtype)
    memory_model_checkpoint = torch.load(checkpoint_dir / "memory_model.pt")["model_parameters"]
    memory_model.load_state_dict(memory_model_checkpoint)
    
    ltm_model.freeze()
    memory_model.freeze()

    logger.info("Loaded checkpoints for models!")

    # Set up agent
    agent = Agent(memory_model)
    memory_module = MemoryModule(
        agent.model.d_mem,
        agent.model.num_vectors,
        agent.model.dtype,
        agent.model.memory_type,
    )

    ###############################################################################
    # Load data
    ###############################################################################
    # dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
    # test_dataset = WikiDataset(data_path=str(dataset_path), split="test")
    # dataloader = EpochDataloader(
    #         test_dataset,
    #         tokenizer,
    #         step_length=args.ltm_params.step_length,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         num_workers=2,
    #         pin_memory=True,
    #     )
    
    dataloader = create_dataloader()
    
    logger.info("Start evaluation...")
    loss = evaluate()
    metrics = {"loss": loss, "ppl": math.exp(loss)}
    with open(log_dir + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    logger.info(format_log(loss, "test"))
    logger.info("Evaluation done!")
