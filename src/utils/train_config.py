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
class RLParams:
    batches_per_update: int
    batch_size: int
    clip: float 
    clip_grad_norm: float
    entropy_coef: float
    gamma: float
    kl_target: float
    min_transitions_per_update: int
    num_prefixes_for_reward_calc: int
    

@dataclass
class TrainerArgs:
    batch_size: int
    step_length: int
    num_train_epochs: int
    ltm_clip_grad_norm: float
    ltm_learning_rate: float
    ltm_model_iterations: int
    memory_model_learning_rate: float
    memory_model_iterations: int
    optimizer: str    
    torch_dtype: str


@dataclass
class TrainingArguments:
    base_model_params: BaseModelParams
    checkpoint_base_cache_dir: str
    checkpoint_dir: str
    checkpoint_interval: int
    content_dir: str
    experiment_name: str
    log_dir: str
    ltm_params: LTMParams
    max_checkpoints: int
    max_eval_steps: int
    memory_model_params: MemoryModelParams
    pretrained_model_name_or_path: str
    rl_params: RLParams
    seed: int    
    trainer_args: TrainerArgs


def load_config(path) -> TrainingArguments:
    TrainConfigSchema = class_schema(TrainingArguments)
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))
