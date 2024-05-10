import itertools
import logging
import os
import pickle
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_base_model import load_base_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.models.rl.train import train_rl
from src.models.rl.utils import State
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.evaluation_config import *
from src.utils.train_utils import create_dir_with_name, create_name, crop_batch, init_arguments

def _evaluate(data: dict) -> torch.Tensor:
    batch_size, num_steps, _ = data["input_ids"].size()
    start_pos = math.ceil(num_steps / 2)
    episode_loss = 0.0
    for step in range(start_pos, num_steps):
        input_ids, attention_mask = (data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous())
        
        loss = model(input_ids=input_ids.to("cuda:0"),
                     attention_mask=attention_mask.to("cuda:0"),
                     labels=input_ids.to("cuda:0"), 
                     return_dict=True)["loss"]

        episode_loss += loss.item()
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
            
            if i == 5000:
                break
            
    return total_loss / it
    
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

    model, tokenizer = load_base_model(args)
    model.eval()

    ###############################################################################
    # Load data
    ###############################################################################
    dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
    test_dataset = WikiDataset(data_path=str(dataset_path), split="test")
    dataloader = EpochDataloader(
        test_dataset,
        tokenizer,
        step_length=args.ltm_params.step_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    logger.info("Start evaluation...")
    loss = evaluate()
    metrics = {"loss": loss, "ppl": math.exp(loss)}
    with open(log_dir + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    logger.info(format_log(loss, "test"))
    logger.info("Evaluation done!")

