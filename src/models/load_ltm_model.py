import torch
import transformers

from src.models.load_base_model import load_base_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.utils.logger_singleton import logger


def load_ltm_model(main_config) -> tuple[LTM_GPT, transformers.AutoTokenizer]:
    logger.info('Loading LTM model...')
    main_model, tokenizer = load_base_model(main_config)
    dtype = torch.float16 if main_config.trainer_args.fp16 else torch.float32
    model = LTM_GPT(main_model,
                    cnt_blocks_with_memory=main_config.ltm_params.cnt_blocks_with_memory,
                    d_mem=main_config.memory_model_params.d_mem,
                    device=torch.device(main_config.device),
                    dtype=dtype)
    return model, tokenizer
