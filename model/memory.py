import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """External memory for the language model."""

    def __init__(self, mem_size: int, n_mem: int, mem_type: str = "conservative"):
        """Initialize the MemoryModule matrix.

        The memory's dimensions are (batch_size x mem_size x n_mem).

        :param n_mem: memory vector size (number of features)
        :param mem_size: number of vectors in the memory
        :param mem_type: memory type, defaults to "conservative"
        """
        super().__init__()
        self.n_mem = n_mem
        self.mem_size = mem_size
        self.mem_type = mem_type
        self.batch_size = None
        self.memory = None

    def reset(self, batch_size: int) -> torch.Tensor:
        """Initialize a new memory matrix"""
        self.batch_size = batch_size
        self.memory = torch.zeros((batch_size, self.mem_size, self.n_mem))
        return self.memory

    @property
    def size(self) -> tuple[int, int]:
        return self.mem_size, self.n_mem

    def update(self, positions: torch.Tensor, new_memory: torch.Tensor) -> torch.Tensor:
        assert self.memory.shape == new_memory.shape
        mask = positions.unsqueeze(-1).expand_as(self.memory)
        self.memory = torch.where(mask == 1, new_memory, self.memory)
        return self.memory
