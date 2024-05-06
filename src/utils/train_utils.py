import argparse
import datetime
import os
import random
import socket

import torch


def init_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


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


def crop_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop_by_len = random.randint(2, input_ids.shape[1])
    return input_ids[:, :crop_by_len], attention_mask[:, :crop_by_len]
