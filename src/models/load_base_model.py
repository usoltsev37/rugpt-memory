from pathlib import Path

import bitsandbytes as bnb
import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from src.utils.logger_singleton import logger


def load_base_model(main_config):
    logger.info('Loading base model...')

    dtype = torch.float16 if main_config.trainer_args.fp16 else torch.float32
    checkpoint_base_cache_dir = Path(main_config.checkpoint_base_cache_dir).resolve()
    bnb = BitsAndBytesConfig(load_in_4bit=main_config.base_model_params.load_in_4bit, bnb_4bit_compute_dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(
        main_config.pretrained_model_name_or_path,
        cache_dir=checkpoint_base_cache_dir,
        quantization_config=bnb,
        torch_dtype=dtype,
        device_map='auto',
        low_cpu_mem_usage=True,
        offload_state_dict=True
    )

    tokenizer = AutoTokenizer.from_pretrained(main_config.pretrained_model_name_or_path)

    if main_config.base_model_params.add_lora:
        modules_to_add_lora = ['attn.a_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
        cnt_blocks_with_memory = main_config.ltm_params.cnt_blocks_with_memory
        num_layers = len(model.transformer.h)
        target_modules = [f"transformer.h.{num_layers - i - 1}.{module}" for module in modules_to_add_lora for i in
                          range(cnt_blocks_with_memory)]
        peft_config = peft.LoraConfig(target_modules=target_modules, inference_mode=False, r=8, lora_alpha=16,
                                      lora_dropout=0.1)
        model = peft.get_peft_model(model, peft_config).base_model.model

    for param in model.parameters():
        param.requires_grad = False
        if param.dtype == torch.float32:
            param.data = param.data.to(dtype)
            torch.nn.init.zeros_(param.data)

    # optimization
    model.gradient_checkpointing_enable()  # memory optimization: only store a small subset of activations, re-compute the rest.
    model.enable_input_require_grads()  # override an implementation quirk in gradient checkpoints that disables backprop unless inputs require grad

    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)  # cast model ouputs to unfuct the top-k sampler
    return model, tokenizer
