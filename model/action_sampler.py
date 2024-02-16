import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dense import DenseNetwork


class ActionSampler(nn.Module):
    """A module that returns distribution parameters for sampling new memory and distribution parameters
     for sampling memory positions to be replaced."""

    def __init__(self, n_mem: int,
                 mem_type: str = "conservative",
                 **kwargs):
        super().__init__()
        self.n_mem = n_mem
        self.mem_type = mem_type

        dense_inp_config = kwargs["dense_input"]
        dense_pos_config = kwargs["dense_position"]
        dense_distr_config = kwargs["dense_distribution"]

        self.dense_inp = DenseNetwork(**dense_inp_config)
        self.dense_pos = DenseNetwork(**dense_pos_config)
        self.dense_distr = DenseNetwork(**dense_distr_config)

    def forward(self, mem: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        :param mem: memory of size [batch, mem_size, n_mem]
        :return: a tuple containing, respectively, memory / memories to replace and new memory matrix
        """
        mem = self.dense_inp(mem)
        out1 = self.dense_pos(mem)   # [batch, mem_size, 1]
        out2 = self.dense_distr(mem) # [batch, mem_size, n_mem * 2]

        positions = None
        if self.mem_type == "conservative":
            positions = F.softmax(out1.squeeze(), dim=-1)
        elif self.mem_type == "flexible":
            positions = F.sigmoid(out1.squeeze())

        return positions, out2
