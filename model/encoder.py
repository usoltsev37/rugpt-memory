import copy
import torch
import torch.nn as nn
from model.dense import MLP


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """Encoder for high-level embeddings from the main model."""

    def __init__(self, block: nn.Module, n_block: int) -> None:
        """
        :param block: encoder block
        :param n_block: number of blocks in encoder
        """
        super().__init__()
        self.blocks = get_clones(block, n_block)
        self.out_norm = nn.LayerNorm(block.d_embd)

    def forward(self, x: torch.tensor, attn_mask: torch.tensor = None) -> torch.tensor:
        """
        :param x: encoder input, tensor of sentence embeddings with size [batch, seq_len, d_embd]
        :param attn_mask: attention mask from tokenizer
        :return: tensor of updated embeddings with size [batch, seq_len, d_embd]
        """
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.out_norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_embd: int, d_hid: int, n_head: int, dropout: float = 0.1) -> None:
        """
        :param d_embd: main model embedding size
        :param n_head: number of heads in the attention mechanism
        :param dropout: dropout value
        :key d_hid: hidden embedding size in MLP
        """
        super().__init__()
        self.d_embd = d_embd
        self.ln_1 = nn.LayerNorm(d_embd)
        self.attn = nn.MultiheadAttention(d_embd, n_head, dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_embd)
        self.mlp = MLP(d_embd, d_hid, dropout)

    def forward(self, x: torch.tensor, pad_mask: torch.tensor = None) -> torch.tensor:
        x = self.ln_1(x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        # x = x + self.attn(x, x, x, key_padding_mask=pad_mask, is_causal=True, attn_mask=attn_mask)[0]
        x = x + self.attn(x, x, x, is_causal=True, attn_mask=attn_mask)[0]
        return x + self.mlp(self.ln_2(x))
