import torch
import torch.nn.functional as F
import numpy as np

from src.models.rl.utils import Action


class MemoryModule:
    """External memory for the language model."""

    def __init__(
        self, d_mem: int, num_vectors: int, dtype: torch.dtype, transform_matrix, memory_type: str = "conservative"
    ):
        """Initialize the MemoryModule.

        The memory's dimensions are [batch_size x num_vectors x d_mem].

        :param d_mem: memory vector size (number of features)
        :param num_vectors: number of vectors in the memory
        :param memory_type: type of the memory, defaults to "conservative"
        """
        self.memory = None
        self.batch_size = None
        self.d_mem = d_mem
        self.num_vectors = num_vectors
        self.memory_type = memory_type
        self.dtype = dtype
        self.d_embd = 768
        self.transform_matrix = transform_matrix

    def reset(self, batch_size: int) -> None:
        """Initialize a new memory"""
        self.batch_size = batch_size
        if self.memory is None or self.memory.shape[0] != batch_size:
            self.memory = torch.zeros(batch_size, self.num_vectors, self.d_mem)
        else:
            self.memory.zero_()

    def update(self, action: Action):
        mask = F.one_hot(action.positions, num_classes=self.num_vectors)
        mask = mask.unsqueeze(-1).expand_as(self.memory).to(self.memory.device)
        self.memory = torch.where(mask == 1, action.memory_vectors.to(self.memory.device), self.memory)

    def update_on_pretrain(self, embeddings: torch.Tensor):
        ids = torch.randint(0, embeddings.shape[1], (self.batch_size, self.num_vectors))
        ids = ids.unsqueeze(2).expand(self.batch_size, self.num_vectors, embeddings.shape[-1])
        selected_embeddings = torch.gather(embeddings, 1, ids)
        self.memory = selected_embeddings.detach() @ self.transform_matrix
