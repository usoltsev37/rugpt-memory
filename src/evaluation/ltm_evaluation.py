import logging
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.memory_model.memory import MemoryModule
from src.utils.eval_utils import format_log
from src.utils.evaluation_config import *
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_utils import create_dir_with_name, init_arguments


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

        num_tokens_in_segment = attention_mask[0].sum(-1) - 1
        episode_loss += loss.item() * num_tokens_in_segment
        token_count += num_tokens_in_segment

        memory_module.update(embeddings)
    
    return episode_loss / token_count


def evaluate():
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            loss = _evaluate(batch)
            losses.append(loss)
    return losses


if __name__ == "__main__":
    ###############################################################################
    # Parse arguments and create directories
    ###############################################################################
    args = init_arguments()
    args = load_config(args.config)
    set_seed(args.seed)

    # Checkpoint dir
    checkpoint_dir = Path(args.pretrained_model_path).resolve()

    # Logs dir
    log_dir = create_dir_with_name(args.log_dir, args.experiment_name)
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
    logger.info(f"Start evaluation...")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Log dir: {log_dir}")

    ###############################################################################
    # Build the model
    ###############################################################################

    ltm_model, tokenizer = load_ltm_model(args)
    ltm_checkpoint = torch.load(checkpoint_dir)["model_parameters"]
    ltm_model.load_state_dict(ltm_checkpoint)
    ltm_model.freeze()

    memory_module = MemoryModule(
        args.memory_model_params.d_mem,
        args.memory_model_params.num_vectors,
        ltm_model.dtype,
        args.memory_model_params.memory_type,
    )

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
        shuffle=False,
        pin_memory=True,
    )

    try:
        losses = evaluate()
        loss = np.mean(losses)
        metrics = {"loss": loss, "ppl": math.exp(loss)}
        with open(log_dir + "/metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)
        logger.info(format_log(loss, "test"))
        logger.info("Evaluation done!")
    except Exception as e:
        logger.error(e)
