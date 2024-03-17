from dataclasses import dataclass, field
from typing import Callable
import torch
import torch.nn.functional as F


@dataclass
class DenseNetworkArgs:
    n_hid_layers: int
    input_dim: int
    hidden_dim: int
    out_dim: int
    activation_fn: Callable[[torch.Tensor], torch.Tensor] = field(default=F.relu)
    dropout: float = field(default=0.1)


@dataclass
class DecoderArgs:
    d_hid: int
    n_head: int
    dense_embd_args: DenseNetworkArgs
    dropout: float = field(default=0.1)


@dataclass
class EncoderArgs:
    d_hid: int
    n_head: int
    dropout: float = field(default=0.1)


@dataclass
class ActionSamplerArgs:
    dense_for_input: DenseNetworkArgs
    dense_for_sampling_positions: DenseNetworkArgs
    dense_for_sampling_memory_vectors: DenseNetworkArgs


@dataclass
class MemoryModelArgs:
    d_embd: int
    d_mem: int
    memory_type: int
    n_enc_block: int
    n_dec_block: int
    encoder_args: EncoderArgs
    decoder_args: DecoderArgs
    action_sampler_args: ActionSamplerArgs


@dataclass
class RLArgs:
    iterations: int = field(default=2000)
    gamma: float = field(default=0.99)
    min_episodes_per_update: int = field(default=4)
    min_transitions_per_update: int = field(default=2048)
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
