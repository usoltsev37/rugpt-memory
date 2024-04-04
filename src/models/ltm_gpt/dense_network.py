import torch
from torch import nn


class DenseNetwork(nn.Module):
    """ DenseNetwork layer(FeedForward in original paper) """

    def __init__(
            self,
            embed_dim=5120,
            hidden_size=10240,
            dtype=torch.float16,
            initialize_with_zeros=False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.ln1 = nn.Linear(self.embed_dim, self.hidden_size, dtype=self.dtype)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(self.hidden_size, self.embed_dim, dtype=self.dtype)

        assert self.dtype in [torch.float16, torch.float32]

        if initialize_with_zeros:
            nn.init.zeros_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
            nn.init.zeros_(self.ln2.weight)
            nn.init.zeros_(self.ln2.bias)

    def forward(self, x):  # x: (sentence_length, batch_size, self.embed_dim)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        return x
