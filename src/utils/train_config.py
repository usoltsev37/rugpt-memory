from dataclasses import dataclass, field

import torch
import yaml
from marshmallow_dataclass import class_schema


@dataclass
class BaseModelParams:
    add_lora: bool = True
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)


@dataclass
class LTMParams:
    device: str
    cnt_blocks_with_memory: int = field(default=2)


@dataclass
class MemoryModelParams:
    device: str
    num_vectors: int
    d_mem: int
    memory_type: str = field(default="conservative")
    n_enc_block: int = field(default=1)
    n_dec_block: int = field(default=1)
    d_embd: int = field(default=768)


@dataclass
class RLParams:
    batches_per_update: int
    batch_size: int
    min_transitions_per_update: int
    gamma: float = field(default=0.99)
    clip: float = field(default=0.2)
    entropy_coef: float = field(default=1e-2)
    clip_grad_norm: float = field(default=1.0)
    kl_target: float = field(default=0.1)


@dataclass
class SyntheticTaskEnvParams:
    d_mem: int
    num_vectors: int
    memory_type: str = field(default="conservative")
    max_steps: int = field(default=10)


@dataclass
class TrainerArgs:
    warmup_steps: int
    gradient_accumulation_steps: int
    fp16: bool
    step_length: int
    num_train_epochs: int
    ltm_model_iterations: int
    memory_model_iterations: int
    batch_size: int
    cut_by_shortest_article: bool
    optimizer: str
    ltm_learning_rate: float
    memory_model_learning_rate: float


@dataclass
class TrainingArguments:
    experiment_name: str
    seed: int
    content_dir: str
    pretrained_model_name_or_path: str
    checkpoint_base_cache_dir: str
    checkpoint_dir: str
    checkpoint_interval: int
    max_eval_steps: int
    max_checkpoints: int
    base_model_params: BaseModelParams
    memory_model_params: MemoryModelParams
    rl_params: RLParams
    ltm_params: LTMParams
    trainer_args: TrainerArgs


def load_config(path) -> TrainingArguments:
    TrainConfigSchema = class_schema(TrainingArguments)
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))
