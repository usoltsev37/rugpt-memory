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

        # self.conv1 = nn.Conv1d(d_mem, d_mem * 4, kernel_size=4, padding="same")
        # self.conv2 = nn.Conv1d(d_mem * 4, d_mem, kernel_size=4, padding="same")

        self.dense_inp = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

        self.dense_pos = DenseNetwork(n_hid_layers=0, input_dim=d_mem, hidden_dim=d_mem * 2, out_dim=1, dtype=dtype)
        self.dense_mem_mu = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

        self.dense_mem_sigma = DenseNetwork(
            n_hid_layers=1,
            input_dim=d_mem,
            hidden_dim=d_mem * 2,
            out_dim=d_mem,
            dtype=dtype,
        )

    def forward(self, mem: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        :param mem: memory of size [batch, mem_size, d_mem]
        :return: a tuple containing, respectively, memory / memories to replace and new memory matrix
        """
        mem = self.dense_inp(mem)
        # x = mem.permute(0, 2, 1)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # # x = self.fcc(x)
        # mem = x.permute(0, 2, 1)
        pos_distr = self.dense_pos(mem)  # [batch, mem_size, 1]
        mu_distr = self.dense_mem_mu(mem)  # [batch, num_vectors, d_mem]
        sigma_distr = torch.tanh(self.dense_mem_sigma(mem))  # [batch, num_vectors, d_mem]

        positions = None
        if self.memory_type == "conservative":
            positions = F.softmax(pos_distr.squeeze(dim=-1), dim=-1)  # [batch_size, num_vectors]
        elif self.memory_type == "flexible":
            positions = F.sigmoid(pos_distr.squeeze(dim=-1))  # [batch_size, num_vectors]

        assert positions.shape[0] == pos_distr.shape[0], "we lost the first shape - shape of the batch"
        return positions, mu_distr, sigma_distr
