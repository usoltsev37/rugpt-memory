from pathlib import Path

import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger_singleton import logger

# import bitsandbytes as bnb
# from transformers.utils.quantization_config import BitsAndBytesConfig


def load_base_model(main_config):
    logger.info("Loading base model...")

    dtype = torch.float16 if main_config.trainer_args.torch_dtype == "float16" else torch.float32
    checkpoint_base_cache_dir = Path(main_config.checkpoint_base_cache_dir).resolve()
    # bnb = BitsAndBytesConfig(load_in_4bit=main_config.base_model_params.load_in_4bit, bnb_4bit_compute_dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(
        main_config.pretrained_model_name_or_path,
        # quantization_config=bnb,
        cache_dir=checkpoint_base_cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    model.d_mem = main_config.memory_model_params.d_mem
    model._dtype = dtype
    model.step_length = main_config.ltm_params.step_length
    model.add_lora = main_config.base_model_params.add_lora
    tokenizer = AutoTokenizer.from_pretrained(main_config.pretrained_model_name_or_path)
    tokenizer.padding_side = "right"

    if main_config.base_model_params.add_lora:
        peft_config = peft.LoraConfig(inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )
        model = peft.get_peft_model(model, peft_config).base_model.model

    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer
