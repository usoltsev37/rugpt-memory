import copy

import torch
import torch.nn as nn

from src.models.memory_model.dense import DenseNetwork


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Creates n clones of a given PyTorch module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """Encoder for high-level embeddings from the LTM model."""

    def __init__(self, block: nn.Module, n_block: int) -> None:
        """
        :param block: The encoder block
        :param n_block: Number of blocks to be created
        """
        super().__init__()
        self.blocks = get_clones(block, n_block)
        self.ln_out = nn.LayerNorm(block.d_embd)

    def forward(self, hidden_states: torch.tensor, attention_mask: torch.Tensor) -> torch.tensor:
        """
        :param hidden_states: Encoder input, tensor of sentence embeddings [batch, seq_len, d_embd]
        :param attention_mask:
        :return: Tensor of updated embeddings [batch, seq_len, d_embd]
        """
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        return self.ln_out(hidden_states)


class EncoderBlock(nn.Module):
    """
    An encoder block consisting of multi-head attention and an MLP.
    """

    def __init__(
        self,
        d_embd: int = 768,
        n_head: int = 4,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0.0,
    ) -> None:
        """
        :param d_embd: LTM model embedding size
        :param d_hid: Hidden layer embedding size
        :param n_head: Number of heads in the multi-head attention mechanism
        :param dtype: Data type of module parameters
        :param dropout: Dropout value
        """
        super().__init__()
        self.d_embd = d_embd
        self.n_head = n_head
        self.dtype = dtype

        self.ln_1 = nn.LayerNorm(d_embd)
        self.attn = nn.MultiheadAttention(d_embd, n_head, dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_embd)
        self.mlp = DenseNetwork(
            n_hid_layers=1, input_dim=d_embd, hidden_dim=d_embd * 2, out_dim=d_embd, dropout=dropout
        )

    def forward(self, hidden_states: torch.tensor, attention_mask: torch.Tensor) -> torch.tensor:
        attn_mask = torch.triu(torch.full((hidden_states.shape[1], hidden_states.shape[1]), 1.0), diagonal=1).to(
            hidden_states.device, dtype=torch.bool
        )
        key_padding_mask = (1 - attention_mask).to(dtype=torch.bool)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            is_causal=True,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = feed_forward_hidden_states + residual
        return hidden_states
