import logging
import os
import shutil
from collections import deque
from pathlib import Path

import torch
import torch.optim
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.memory_model.memory import MemoryModule
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_config import load_config
from src.utils.train_utils import create_dir_with_name, init_arguments, crop_batch


def save_models(output_dir: Path) -> None:
    logger.info(f"Saving models checkpoints to {output_dir}")

    torch.save(
        {
            "cur_iter": cur_iter,
            "model_parameters": ltm_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "ltm.pt",
    )


def save_checkpoint(val_loss):
    global checkpoint_dir
    global saved_checkpoints_queue
    global best_val_loss

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_folder = f"best_model"
        output_dir = checkpoint_dir / checkpoint_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_models(output_dir)

    checkpoint_folder = f"checkpoint-{cur_iter}"
    output_dir = checkpoint_dir / checkpoint_folder
    saved_checkpoints_queue.append(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_models(output_dir)

    if len(saved_checkpoints_queue) > args.max_checkpoints:
        oldest_checkpoint = saved_checkpoints_queue.popleft()
        shutil.rmtree(oldest_checkpoint)


def _evaluate(data: dict) -> torch.Tensor:
    batch_size, num_steps, _ = data["input_ids"].size()
    episode_loss = 0.0
    token_count = 0
    
    memory_module.reset(batch_size)

    for step in range(num_steps):
        input_ids, attention_mask = (
            data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous(),
        )

        loss, embeddings = ltm_model(input_ids, attention_mask, memory_module.memory)

        # There are no previous embeddings in the first step
        if step != 0:
            num_tokens_in_segment = attention_mask[0].sum(-1)
            episode_loss += loss.item() * num_tokens_in_segment
            token_count += num_tokens_in_segment

        memory_module.update(embeddings)

    return episode_loss / token_count


def evaluate(val_dataloader, ltm_model):
    ltm_model.freeze()
    torch.cuda.empty_cache()

    it, total_loss = 0, 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if 0 < args.max_eval_steps <= i:
                break
            loss = _evaluate(batch)
            total_loss += loss
            it += 1

    return total_loss / it


def train_ltm_on_episode(ltm_model, ltm_optimizer, memory_module, data: dict, ltm_clip_grad_norm) -> float:
    episode_loss = 0.0
    batch_size, num_steps, _ = data["input_ids"].size()
    token_count = 0

    memory_module.reset(batch_size)
    for step in range(num_steps):
        input_ids, attention_mask = crop_batch(
            data["input_ids"][:, step, :].contiguous(),
            data["attention_mask"][:, step, :].contiguous(),
        )

        loss, embeddings = ltm_model(input_ids, attention_mask, memory_module.memory)

        # There are no previous embeddings on the first step
        if step != 0:
            loss.backward()
            nn.utils.clip_grad_norm_(ltm_model.parameters(), ltm_clip_grad_norm)
            ltm_optimizer.step()
            
            num_tokens_in_segment = attention_mask[0].sum(-1)
            episode_loss += loss.item() * num_tokens_in_segment
            token_count += num_tokens_in_segment

        ltm_optimizer.zero_grad()
        memory_module.update(embeddings)

    return episode_loss / token_count


def pretrain(ltm_model, ltm_optimizer, memory_module, train_dataloader, val_dataloader, args):
    logger.info("Start LTM pretraining...")
    global cur_iter
    global best_val_loss
    best_val_loss = 1e6

    for cur_iter, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader)), start=1):
        if batch["input_ids"].shape[1] < 2:
            continue
        ltm_loss = train_ltm_on_episode(
            ltm_model, ltm_optimizer, memory_module, batch, args.trainer_args.ltm_clip_grad_norm
        )
        logger.info(f"""Training iteration {cur_iter} done.\nLTM train loss: {ltm_loss}""")
        tensorboard_writer.add_scalar("Loss/LTM iteration loss", ltm_loss, cur_iter)

        if not cur_iter % 20:
            ltm_val_loss = evaluate(val_dataloader=val_dataloader, ltm_model=ltm_model)
            ltm_model.unfreeze()
            logger.info(f"""Evaluation on {cur_iter} done.\nLTM val loss: {ltm_loss}""")
            tensorboard_writer.add_scalar("Loss/LTM val loss", ltm_val_loss, cur_iter)
            save_checkpoint(ltm_val_loss)

    logger.info("LTM model pretraining done!")


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

global tensorboard_writer
# Tensorboard writer
tensorboard_writer = SummaryWriter(log_dir=log_dir)

# Save train config to log_dir
content_dir = Path(args.content_dir).resolve()
shutil.copy(content_dir / "configs" / "pretrain_ltm_config.yml", log_dir)

logger.info(f"Pretraining LTM. We teach LTM model to look at the memory.")
logger.info(f"Experiment name: {args.experiment_name}")
logger.info(f"Checkpoints dir: {checkpoint_dir}")
logger.info(f"Log dir: {log_dir}")


###############################################################################
# Build the model and set up environment
###############################################################################

ltm_model, tokenizer = load_ltm_model(args)
ltm_model.unfreeze()
optimizer = AdamW(ltm_model.parameters(), lr=args.trainer_args.ltm_learning_rate)

global memory_module
memory_module = MemoryModule(
    args.memory_model_params.d_mem,
    args.memory_model_params.num_vectors,
    torch.float32,
    args.memory_model_params.memory_type,
)
###############################################################################
# Load data
###############################################################################
dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
train_dataset = WikiDataset(data_path=str(dataset_path), split="train")
val_dataset = WikiDataset(data_path=str(dataset_path), split="val")

train_dataloader = EpochDataloader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    step_length=args.ltm_params.step_length,
    batch_size=args.trainer_args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_dataloader = EpochDataloader(
    dataset=val_dataset,
    tokenizer=tokenizer,
    step_length=args.ltm_params.step_length,
    batch_size=args.trainer_args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

###############################################################################
# Pretrain
###############################################################################
pretrain(
    ltm_model=ltm_model,
    ltm_optimizer=optimizer,
    memory_module=memory_module,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    args=args,
)
