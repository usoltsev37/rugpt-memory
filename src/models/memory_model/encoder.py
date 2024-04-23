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
        self.ln_out = nn.LayerNorm(block.d_embd, dtype=block.dtype)

    def forward(self, x: torch.tensor, attention_mask: torch.Tensor) -> torch.tensor:
        """
        :param x: Encoder input, tensor of sentence embeddings [batch, seq_len, d_embd]
        :param attention_mask:
        :return: Tensor of updated embeddings [batch, seq_len, d_embd]
        """
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_out(x)


class EncoderBlock(nn.Module):
    """
    An encoder block consisting of multi-head attention and an MLP.
    """

    def __init__(
        self,
        d_embd: int = 5120,
        d_hid: int = 5120 // 8,
        n_head: int = 4,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
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
        self.d_hid = d_hid
        self.n_head = n_head
        self.dtype = dtype

        self.ln_1 = nn.LayerNorm(d_embd, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            d_embd, n_head, dropout, batch_first=True, dtype=dtype
        )
        self.ln_2 = nn.LayerNorm(d_embd, dtype=dtype)
        self.mlp = DenseNetwork(1, d_embd, d_hid, d_embd, dropout=dropout, dtype=dtype)

    def forward(self, x: torch.tensor, attention_mask: torch.Tensor) -> torch.tensor:
        x = self.ln_1(x)
        attn_mask = torch.triu(
            torch.full((x.shape[1], x.shape[1]), 1.0), diagonal=1
        ).to(x.device)
        # zero_rows = (attention_mask == 0).all(dim=1)
        # attention_mask_copy = attention_mask.copy()
        # attention_mask_copy[zero_rows, 0] = 1
        # key_padding_mask = (1 - attention_mask_copy).bool()
        key_padding_mask = (1 - attention_mask).bool()

        x = (
            x
            + self.attn(
                x,
                x,
                x,
                is_causal=True,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )[0]
        )
        return x + self.mlp(self.ln_2(x))
