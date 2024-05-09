from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema
from src.utils.train_config import BaseModelParams, LTMParams, MemoryModelParams, RLParams, TrainerArgs


@dataclass
class PretrainParams:
    episode_max_steps: int
    iterations: int
    lr: float


@dataclass
class TrainingArguments:
    base_model_params: BaseModelParams
    checkpoint_base_cache_dir: str
    checkpoint_dir: str
    checkpoint_interval: int
    content_dir: str
    eval_interval: int
    experiment_name: str
    description: str
    log_dir: str
    ltm_params: LTMParams
    max_checkpoints: int
    max_eval_steps: int
    memory_model_params: MemoryModelParams
    pretrained_model_name_or_path: str
    pretrain_params: PretrainParams
    rl_params: RLParams
    seed: int
    trainer_args: TrainerArgs


def load_config(path) -> TrainingArguments:
    TrainConfigSchema = class_schema(TrainingArguments)
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))
