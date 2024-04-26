import torch


class State:
    """Represents the state of the environment."""

    def __init__(
        self,
        memory: torch.Tensor,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        self.memory = memory
        self.attention_mask = attention_mask
        self.embeddings = embeddings
        self.batch_size = embeddings.shape[0]
        self.to(device=device)

    def __str__(self):
        details = (
            f"{name}\n    Shape: {getattr(self, name).shape}\n    Tensor: {getattr(self, name)}"
            for name in ["memory", "embeddings", "attention_mask"]
        )
        return "\n".join(details)

    def to(self, device: str) -> "State":
        if isinstance(device, str):
            device = torch.device(device)
        for attr in ["memory", "embeddings", "attention_mask"]:
            tensor = getattr(self, attr)
            if tensor.device != device:
                setattr(self, attr, tensor.to(device))


class Action:
    """Represents an action taken by the agent."""

    def __init__(
        self, positions: torch.Tensor, memory_vectors: torch.Tensor, device: torch.device = torch.device("cpu")
    ):
        """
        :param positions: Tensor of shape [batch_size, num_vectors]
        :param memory_vectors: Tensor of shape [batch_size, num_vectors, d_mem]
        """
        self.positions = positions
        self.memory_vectors = memory_vectors
        self.batch_size = positions.shape[0]
        self.to(device=device)

    def __str__(self):
        details = (
            f"{name}\n    Shape: {getattr(self, name).shape}\n    Tensor: {getattr(self, name)}"
            for name in ["positions", "memory_vectors"]
        )
        return "\n".join(details)

    def to(self, device: str) -> "Action":
        if isinstance(device, str):
            device = torch.device(device)
        for attr in ["positions", "memory_vectors"]:
            tensor = getattr(self, attr)
            if tensor.device != device:
                setattr(self, attr, tensor.to(device))
