import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class DenseNetwork(nn.Module):
    def __init__(self,
                 n_hid_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 dropout: float = 0.1):
        """Dense network with adjustable number of hidden layers.
        :param n_hid_layers: number of hidden layers
        :param input_dim: size of each input sample
        :param hidden_dim: size of each hidden sample
        :param out_dim: size of each output sample
        :param activation: activation function
        :param dropout: dropout value
        """
        super().__init__()
        self.act = activation_fn
        self.layers = nn.ModuleList()

        # First linear layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        if dropout:
            self.layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_hid_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # Last layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                x = self.act(layer(x))
            else:
                # apply dropout
                x = layer(x)
        return self.layers[-1](x)


class MLP(nn.Module):
    """Dense network in encoder and decoder of the memory model."""

    def __init__(self, n_embd: int, n_hid: int, dropout: float = 0.1) -> None:
        """
        :param n_embd: main model embedding size
        :param n_hid: hidden embedding size
        :param dropout: dropout value
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_hid)
        self.c_proj = nn.Linear(n_hid, n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: matrix of embeddings with size [batch, seq_len, n_embd]
        :return: matrix of updated embeddings with size [batch, seq_len, n_embd]
        """
        x = self.dropout(self.act(self.c_fc(x)))
        return self.c_proj(x)
