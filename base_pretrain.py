import logging
import os
import shutil
from collections import deque
from pathlib import Path

import torch
import copy
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_base_model import load_base_model
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_config import *
from src.utils.train_utils import create_dir_with_name, create_name, crop_batch, init_arguments

def save_models(output_dir: Path) -> None:
    logger.info(f"Saving models checkpoints to {output_dir}")

    torch.save(
        {
            "model_parameters": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "base_model.pt",
    )

def save_checkpoint():
    global checkpoint_dir
    global saved_checkpoints_queue

    checkpoint_folder = f"checkpoint-{train_step}"
    output_dir = checkpoint_dir / checkpoint_folder
    saved_checkpoints_queue.append(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_models(output_dir)

    if len(saved_checkpoints_queue) > args.max_checkpoints:
        oldest_checkpoint = saved_checkpoints_queue.popleft()
        shutil.rmtree(oldest_checkpoint)


def _evaluate(data: dict) -> torch.Tensor:
    _, num_steps, _ = data["input_ids"].size()
    episode_loss = 0.0
    episode_token_count = 0

    for step in range(num_steps):
        # FULL SEGMENT!
        input_ids, attention_mask = (
            data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous(),
        )
        labels = copy.deepcopy(input_ids)
        labels[labels == tokenizer.pad_token_id] = -100

        out = model(input_ids=input_ids.to("cuda:0"),
                    attention_mask=attention_mask.to("cuda:0"),
                    labels=labels, 
                    return_dict=True)
        
        num_tokens_in_segment = attention_mask[0].sum(-1)
        episode_token_count += num_tokens_in_segment
        episode_loss += out["loss"].item() * num_tokens_in_segment
        

    return episode_loss / episode_token_count

def evaluate():
    model.eval()
    torch.cuda.empty_cache()
    val_dataloader = create_val_dataloader()
    it, total_loss = 0, 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if 0 < args.max_eval_steps <= i:
                break
            loss = _evaluate(batch)
            total_loss += loss
            it += 1

    return total_loss / it

def train_on_episode(data):

    episode_loss = 0.0
    _, num_steps, _ = data["input_ids"].size()
    episode_token_count = 0

    for step in range(num_steps):
        input_ids, attention_mask = crop_batch(
            data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous(),
        )
        
        labels = copy.deepcopy(input_ids)
        labels[labels == tokenizer.pad_token_id] = -100
        
        out = model(input_ids=input_ids.to("cuda:0"),
                    attention_mask=attention_mask.to("cuda:0"),
                    labels=labels, 
                    return_dict=True)
        
        out["loss"].backward()

        optimizer.step()
        optimizer.zero_grad()
        
        num_tokens_in_segment = attention_mask[0].sum(-1)
        episode_token_count += num_tokens_in_segment
        episode_loss += out["loss"].item() * num_tokens_in_segment

    return episode_loss / episode_token_count

def train():
    global train_step
    for train_step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader)), start=1):
        if train_step < 2848:
            continue
        train_loss = train_on_episode(batch)
        logger.info(f"Train loss on iter {train_step}: {train_loss:.4f}")
        tensorboard_writer.add_scalar("Loss/train", train_loss, train_step)
        
        if not train_step % args.checkpoint_interval:
            val_loss = evaluate()
            model.train()
            logger.info(f"Val loss on iter {train_step}: {val_loss:.4f}")
            tensorboard_writer.add_scalar("Loss/val", val_loss, train_step)
        
        if not train_step % args.checkpoint_interval:
            save_checkpoint()

if __name__ == "__main__":

    ###############################################################################
    # Parse arguments and create directories
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
    shutil.copy(content_dir / "configs" / "train_config.yml", log_dir)
    logger.info(f"Start training...")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Checkpoints dir: {checkpoint_dir}")
    logger.info(f"Log dir: {log_dir}")

    ###############################################################################
    # Build the model
    ###############################################################################

    model, tokenizer = load_base_model(args)
    checkpoint_path = "/home/akarpov/jbelova/rugpt-memory/checkpoints/pretrain_base/pretrain_base:new_dataset/runs/checkpoint-2800/base_model.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_parameters"])
    model.train()

    ###############################################################################
    # Create optimizers
    ###############################################################################

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.trainer_args.lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ###############################################################################
    # Load data
    ###############################################################################
    dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
    train_dataset = WikiDataset(data_path=str(dataset_path), split="train")
    
    def create_val_dataloader():
        val_dataset = WikiDataset(data_path=str(dataset_path), split="val")
        return EpochDataloader(
            val_dataset,
            tokenizer,
            step_length=args.ltm_params.step_length,
            batch_size=args.trainer_args.batch_size,
            shuffle=False,
            # num_workers=2,
            pin_memory=True,
        )
    
    train_dataloader = EpochDataloader(
            train_dataset,
            tokenizer,
            step_length=args.ltm_params.step_length,
            batch_size=args.trainer_args.batch_size,
            shuffle=False,
            # num_workers=2,
            pin_memory=True,
        )
    


    ###############################################################################
    # Train
    ###############################################################################
    for epoch in range(args.trainer_args.num_train_epochs):  # epoch == traverse over train dataset once
            train()
    # try:

    #         logger.info("-" * 100)
    #     logger.info("End of training.")
    # except (KeyboardInterrupt, Exception) as e:
    #     logger.info("-" * 100)
    #     logger.info("Exiting from training early")
    #     logger.error(e)
