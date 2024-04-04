import torch


class State:
    """Represents the state of the environment."""

    def __init__(self, memory: torch.Tensor, embeddings: torch.Tensor):
        """
        :param memory: Tensor of shape [batch_size, num_vectors, d_mem]
        :param embeddings: Tensor of shape [batch_size, seq_len, d_embd]
        """
        self.memory = memory
        self.embeddings = embeddings

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory[item], self.embeddings[item]

    def to(self, device: torch.device) -> 'State':
        """Move tensors to the specified device."""
        return State(self.memory.to(device), self.embeddings.to(device))


class Action:
    """Represents an action taken by the agent."""

    def __init__(self, positions: torch.Tensor, memory_vectors: torch.Tensor):
        """
        :param positions: Tensor of shape [batch_size, num_vectors]
        :param memory_vectors: Tensor of shape [batch_size, num_vectors, d_mem]
        """
        self.positions = positions
        self.memory_vectors = memory_vectors

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[item], self.memory_vectors[item]

    def to(self, device: torch.device) -> 'Action':
        """Move tensors to the specified device."""
        return Action(self.positions.to(device), self.memory_vectors.to(device))
