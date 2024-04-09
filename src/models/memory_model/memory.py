import torch
import torch.nn.functional as F

from src.models.rl.utils import Action


class MemoryModule:
    """External memory for the language model."""

    def __init__(self, d_mem: int, num_vectors: int, memory_type: str = "conservative"):
        """Initialize the MemoryModule.

        The memory's dimensions are [batch_size x num_vectors x d_mem].

        :param d_mem: memory vector size (number of features)
        :param num_vectors: number of vectors in the memory
        :param memory_type: type of the memory, defaults to "conservative"
        """
        self.memory = None
        self.d_mem = d_mem
        self.num_vectors = num_vectors
        self.memory_type = memory_type

    def reset(self, batch_size: int) -> None:
        """Initialize a new memory"""
        self.memory = torch.zeros((batch_size, self.num_vectors, self.d_mem))

    def update(self, action: Action) -> torch.Tensor:
        mask = F.one_hot(action.positions,
                         num_classes=self.num_vectors)
        mask = mask.unsqueeze(-1).expand_as(self.memory)
        self.memory = torch.where(mask == 1, action.memory_vectors, self.memory)
        return self.memory
