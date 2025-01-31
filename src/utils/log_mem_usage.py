import psutil
import time
import logging
import sys
import torch

# from src.utils.logger_singleton import ColourFormatter
import subprocess


class ColourFormatter(logging.Formatter):
    green = "\x1b[32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s: %(levelname)s] %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def log_memory_usage(logger1):
    logger1.info(f"RAM usage: {psutil.virtual_memory().percent}%")
    cuda_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    info = cuda_result.split("\n")

    for idx, gpu_info in enumerate(info):
        if gpu_info:
            total, used, free = gpu_info.split(", ")
            free_percent = (float(free) / float(total)) * 100
            logger1.info(
                f"GPU {idx}: Memory Total: {total} MB, Used: {used} MB, Free: {free} MB ({free_percent:.2f}% free)"
            )


if __name__ == "__main__":
    file = sys.argv[1]

    logger1 = logging.getLogger("train")
    logger1.setLevel(logging.INFO)
    handler1 = logging.FileHandler(file)
    handler1.setFormatter(ColourFormatter())
    logger1.addHandler(handler1)

    while True:
        log_memory_usage(logger1)
        time.sleep(5)
