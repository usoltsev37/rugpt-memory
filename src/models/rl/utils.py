import torch


class State:
    """Represents the state of the environment."""

    def __init__(self,
                 memory: torch.Tensor,
                 embeddings: torch.Tensor,
                 attention_mask: torch.Tensor):
        self.memory = memory
        self.attention_mask = attention_mask
        self.embeddings = embeddings
        self.batch_size = embeddings.shape[0]

    def __str__(self):
        return f"""Memory
    Shape: {self.memory.shape}
    Tensor: {self.memory}
Embeddings
    Shape: {self.embeddings.shape}
    Tensor: {self.embeddings}
Attention mask
    Shape: {self.attention_mask.shape}
    Tensor: {self.attention_mask}
    """

    def __getitem__(self, item: int) -> 'State':
        return State(self.memory[item], self.embeddings[item])

    def to(self, device: torch.device) -> 'State':
        self.memory = self.memory.to(device)
        self.embeddings = self.embeddings.to(device)
        return self


class Action:
    """Represents an action taken by the agent."""

    def __init__(self, positions: torch.Tensor,
                 memory_vectors: torch.Tensor):
        """
        :param positions: Tensor of shape [batch_size, num_vectors]
        :param memory_vectors: Tensor of shape [batch_size, num_vectors, d_mem]
        """
        self.positions = positions
        self.memory_vectors = memory_vectors
        self.batch_size = positions.shape[0]

    def __str__(self):
        return f"""Positions
    Shape: {self.positions.shape}
    Tensor: {self.positions}
Memory vectors
    Shape: {self.memory_vectors.shape}
    Tensor: {self.memory_vectors}
    """

    def __getitem__(self, item: int) -> 'Action':
        return Action(self.positions[item], self.memory_vectors[item])

    def to(self, device: torch.device) -> 'Action':
        """Move tensors to the specified device."""
        self.positions = self.positions.to(device)
        self.memory_vectors = self.memory_vectors.to(device)
        return self
