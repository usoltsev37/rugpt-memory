import copy
import logging
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_base_model import load_base_model
from src.utils.eval_utils import format_log, calculate_confidence_interval
from src.utils.evaluation_config import *
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_utils import (create_dir_with_name, create_name,
                                   crop_batch, init_arguments)
    

def _evaluate(data: dict) -> torch.Tensor:
    _, num_steps, _ = data["input_ids"].size()
    
    episode_loss = 0.0
    episode_token_count = 0
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
        
        labels = copy.deepcopy(input_ids)
        labels[labels == tokenizer.pad_token_id] = -100

        out = model(input_ids=input_ids.to("cuda:0"),
                    attention_mask=attention_mask.to("cuda:0"),
                    labels=labels, 
                    return_dict=True)
        
        num_tokens_in_segment = attention_mask[0].sum(-1) - 1
        
        episode_token_count += num_tokens_in_segment
        episode_loss += out["loss"].item() * num_tokens_in_segment
    
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

    model, tokenizer = load_base_model(args)
    checkpoint_path = Path(args.pretrained_model_path) / "base_model.pt"
    checkpoint = torch.load(checkpoint_path)["model_parameters"]
    model.load_state_dict(checkpoint)
    model.eval()

    ##############################################################################
    # Load data
    ##############################################################################
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

