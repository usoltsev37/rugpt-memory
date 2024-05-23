
import logging
import math
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
import time

import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.utils import State
from src.utils.evaluation_config import *
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_utils import create_dir_with_name, crop_batch, init_arguments

from src.utils.eval_utils import format_log, calculate_confidence_interval

def _evaluate(data: dict) -> torch.Tensor:
    batch_size, num_steps, _ = data["input_ids"].size()
    episode_loss = 0.0
    episode_token_count = 0

    memory_module.reset(batch_size)

    if not args.last_segments:
        range_ = range(num_steps)
    else:
        range_ = range(num_steps - math.ceil(0.25 * num_steps), num_steps)

    for step in range_:
        if args.full_segment:
            input_ids, attention_mask = (
                data["input_ids"][:, step, :].contiguous(),
                data["attention_mask"][:, step, :].contiguous(),
            )
        else:
            input_ids, attention_mask = crop_batch(
                data["input_ids"][:, step, :].contiguous(),
                data["attention_mask"][:, step, :].contiguous(),
            )

        loss, embeddings = ltm_model(input_ids, attention_mask, memory_module.memory)

        num_tokens_in_segment = attention_mask[0].sum(-1) - 1
        episode_token_count += num_tokens_in_segment
        episode_loss += loss.item() * num_tokens_in_segment

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

    return episode_loss / episode_token_count


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
    shutil.copy(content_dir / "configs" / "ssh-91" / "eval_config.yml", log_dir)
    logger.info(f"Start evaluation...")
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

    time.sleep(5400)
    
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
        ppl = np.exp(losses)
        ci_loss = calculate_confidence_interval(losses)
        ci_ppl = calculate_confidence_interval(ppl)
        metrics = {"losses": losses, "ci_loss": ci_loss, "ci_ppl": ci_ppl}
        with open(log_dir + "/metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)
        logger.info(format_log(ci_loss, ci_ppl, "test"))
        logger.info("Evaluation done!")
    except Exception as e:
        logger.error(e)
