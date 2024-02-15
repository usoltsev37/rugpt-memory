import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

from src.utils.logger_singleton import logger


def load_base_model(main_config):
    logger.info('Loading base model')
    
    content_dir = Path(main_config.content_dir).resolve()
    checkpoint_base_cache_dir = Path(main_config.checkpoint_base_cache_dir).resolve()
    
    model = AutoModelForCausalLM.from_pretrained(
        main_config.pretrained_model_name_or_path,
        cache_dir=checkpoint_base_cache_dir,
        load_in_4bit=main_config.base_model_params.load_in_4bit,
        load_in_8bit=main_config.base_model_params.load_in_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
        offload_state_dict=True
    )
    tokenizer = AutoTokenizer.from_pretrained(main_config.pretrained_model_name_or_path)

    for param in model.parameters():
        param.requires_grad = False

    # optimization 
    model.gradient_checkpointing_enable() # memory optimization: only store a small subset of activations, re-compute the rest.
    model.enable_input_require_grads() # override an implementation quirk in gradient checkpoints that disables backprop unless inputs require grad
    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head) # cast model ouputs to unfuct the top-k sampler
    
    return model, tokenizer