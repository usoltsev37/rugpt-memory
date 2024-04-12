import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_model.dense import DenseNetwork


class ActionSampler(nn.Module):
    """A module that returns distribution parameters for sampling new memory and distribution parameters
     for sampling memory positions to be replaced."""

    def __init__(self,
                 d_mem: int,
                 dtype: torch.dtype,
                 memory_type: str = "conservative"
                 ) -> None:

        super().__init__()
        self.d_mem = d_mem
        self.memory_type = memory_type
        self.dtype = dtype

        self.dense_inp = DenseNetwork(n_hid_layers=1,
                                      input_dim=d_mem,
                                      hidden_dim=d_mem * 2,
                                      out_dim=d_mem,
                                      dtype=dtype)
        self.dense_pos = DenseNetwork(n_hid_layers=1,
                                      input_dim=d_mem,
                                      hidden_dim=d_mem * 2,
                                      out_dim=1,
                                      dtype=dtype)
        self.dense_mem_vec = DenseNetwork(n_hid_layers=1,
                                          input_dim=d_mem,
                                          hidden_dim=d_mem * 2,
                                          out_dim=d_mem * 2,
                                          dtype=dtype)

    def forward(self, mem: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        :param mem: memory of size [batch, mem_size, d_mem]
        :return: a tuple containing, respectively, memory / memories to replace and new memory matrix
        """
        mem = self.dense_inp(mem)
        out1 = self.dense_pos(mem)  # [batch, mem_size, 1]
        out2 = self.dense_mem_vec(mem)  # [batch, num_vectors, d_mem * 2]

        positions = None
        if self.memory_type == "conservative":
            positions = F.softmax(out1.squeeze(dim=-1), dim=-1)  # [batch_size, num_vectors]
        elif self.memory_type == "flexible":
            positions = F.sigmoid(out1.squeeze(dim=-1))  # [batch_size, num_vectors]

        assert positions.shape[0] == out1.shape[0], "we lost the first shape - shape of the batch"

        return positions, out2
