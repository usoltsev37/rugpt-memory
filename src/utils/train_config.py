from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml
import torch


@dataclass
class BaseModelParams:
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)

@dataclass
class LTMParams:
    n_ctx: int = field(default=2048)
    d_embd: int = field(default=5120)
    cnt_blocks_with_memory: int = field(default=2)


@dataclass
class MemoryModelParams:
    num_vectors: int
    d_mem: int
    memory_type: int = field(default="conservative")
    n_enc_block: int = field(default=2)
    n_dec_block: int = field(default=3)
    d_embd: int = field(default=5120)


@dataclass
class RLParams:
    iterations: int = field(default=10)
    gamma: float = field(default=0.99)
    min_episodes_per_update: int = field(default=4)
    min_transitions_per_update: int = field(default=256)
    device: str = field(default="cpu")
    agent_lr: float = field(default=1e-4)
    clip: float = field(default=0.2)
    batches_per_update: int = field(default=10)
    batch_size: int = field(default=256)
    entropy_coef: float = field(default=1e-2)
    clip_grad_norm: float = field(default=1.0)
    kl_target: float = field(default=0.03)


@dataclass
class SyntheticTaskEnvParams:
    d_mem: int = field(default=5)
    num_vectors: int = field(default=3)
    memory_type: str = field(default="conservative")
    max_steps: int = field(default=10)


@dataclass
class TrainerParams:
    num_train_epochs: int
    ltm_model_iterations: int
    memory_model_iterations: int
    batch_size: int
    log_interval: int
    adaptive_softmax: bool = field(default=False)
    optim: torch.optim = field(default="adamw")
    lr: float = field(default=1e-4)


@dataclass
class TokenizerParams:
    pass


@dataclass
class TrainParams:
    experiment_name: str
    seed: int
    cuda: bool
    dataset_path: str
    n_context: int
    pretrained_model_name_or_path: str
    checkpoint_base_cache_dir: str
    wandb_logging: bool
    tokenizer_params: TokenizerParams
    base_model_params: BaseModelParams
    memory_model_params: MemoryModelParams
    rl_params: RLParams
    ltm_model_params: LTMParams
    trainer_params: TrainerParams

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
