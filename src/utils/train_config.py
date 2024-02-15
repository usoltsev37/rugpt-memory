from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class BaseModelParams:
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)

@dataclass
class TrainerArgs:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    learning_rate: float
    fp16: bool
    logging_steps: int
    report_to: str = field(default=None)

@dataclass
class TrainParams:
    exp_name: str
    device: str
    content_dir: str
    pretrained_model_name_or_path: str
    checkpoint_base_cache_dir: str
    base_model_params: BaseModelParams
    trainer_args: TrainerArgs


TrainConfigSchema = class_schema(TrainParams)


def load_config(path) -> TrainParams:
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))