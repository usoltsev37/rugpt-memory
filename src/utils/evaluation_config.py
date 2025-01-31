from dataclasses import dataclass, field

import yaml
from marshmallow_dataclass import class_schema


@dataclass
class BaseModelParams:
    add_lora: bool = True
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    
@dataclass
class LTMParams:
    cnt_blocks_with_memory: int
    device: str
    step_length: int

@dataclass
class MemoryModelParams:
    device: str
    d_mem: int
    d_embd: int
    num_vectors: int
    n_dec_block: int
    n_enc_block: int
    memory_type: str = field(default="conservative")
    
@dataclass
class TrainerArgs:
    batch_size: int
    num_train_epochs: int
    ltm_clip_grad_norm: float
    ltm_learning_rate: float
    ltm_model_iterations: int
    memory_model_learning_rate: float
    memory_model_iterations: int
    optimizer: str
    torch_dtype: str
    
@dataclass
class EvaluationArguments:
    base_model_params: BaseModelParams
    batch_size: int
    checkpoint_base_cache_dir: str
    content_dir: str
    experiment_name: str
    description: str
    log_dir: str
    ltm_params: LTMParams
    memory_model_params: MemoryModelParams
    pretrained_model_name_or_path: str
    pretrained_model_path: str
    seed: int
    trainer_args: TrainerArgs

def load_config(path) -> EvaluationArguments:
    TrainConfigSchema = class_schema(EvaluationArguments)
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))