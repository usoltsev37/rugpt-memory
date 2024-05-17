import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_model.dense import DenseNetwork


class ActionSampler(nn.Module):
    """A module that returns distribution parameters for sampling new memory and distribution parameters
    for sampling memory positions to be replaced."""

    def __init__(self, d_mem: int, dtype: torch.dtype, memory_type: str = "conservative") -> None:

        super().__init__()
        self.d_mem = d_mem
        self.memory_type = memory_type
        self.dtype = dtype

        self.dense_inp = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

        self.dense_pos_distr = DenseNetwork(
            n_hid_layers=1, input_dim=d_mem, hidden_dim=d_mem * 2, out_dim=1, dtype=dtype
        )
        self.dense_norm_mu = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

        self.dense_norm_sigma = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

    def forward(self, memory: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        :param memory: memory of size [batch, mem_size, d_mem]
        :return: a tuple containing, respectively, memory / memories to replace and new memory matrix
        """
        memory = self.dense_inp(memory)

        pos_distr = self.dense_pos_distr(memory)  # [batch, mem_size, 1]
        mu_distr = self.dense_norm_mu(memory)  # [batch, num_vectors, d_mem]
        sigma_distr = 2 * (torch.tanh(self.dense_norm_sigma(memory)) - 1)  # [batch, num_vectors, d_mem]
        # sigma_distr = self.dense_norm_sigma(memory)  # [batch, num_vectors, d_mem]

        positions = None
        if self.memory_type == "conservative":
            positions = F.softmax(pos_distr.squeeze(dim=-1), dim=-1)  # [batch_size, num_vectors]
        elif self.memory_type == "flexible":
            positions = F.sigmoid(pos_distr.squeeze(dim=-1))  # [batch_size, num_vectors]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # mu_distr = torch.ones(memory.shape).to(memory.device)
        # sigma_distr = torch.full(memory.shape, -10).to(memory.device)

        return positions, mu_distr, sigma_distr
