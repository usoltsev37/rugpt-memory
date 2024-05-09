import copy

import torch
import torch.nn as nn

from src.models.memory_model.dense import DenseNetwork


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Creates n clones of a given PyTorch module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Decoder(nn.Module):
    """
    Decoder for the memory.
    """

    def __init__(self, block: nn.Module, n_block: int = 3) -> None:
        """
        :param block: Decoder block
        :param n_block: Number of blocks to be created
        """
        super().__init__()
        self.blocks = get_clones(block, n_block)
        # self.ln_out = nn.LayerNorm(block.d_mem)

    def forward(
        self,
        memory: torch.tensor,
        embeddings: torch.tensor,
        attention_mask: torch.Tensor,
    ) -> torch.tensor:
        """
        :param memory: Tensor of memory embeddings with size [batch, mem_size, n_mem]
        :param embeddings: Tensor of sentence embeddings with size [batch, seq_lem, n_embd]
        :param attention_mask:
        :return: Tensor of updated memory embeddings with size [batch, mem_size, n_mem]
        """
        for block in self.blocks:
            memory = block(memory, embeddings, attention_mask)
        return memory
        # return self.ln_out(memory)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_mem: int,
        d_embd: int = 768,
        n_head: int = 4,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0.0,
    ) -> None:
        """
        :param d_embd: LTM model embedding size
        :param d_mem: Memory vector size
        :param n_head: Number of heads in the attention mechanism
        :param dtype: Data type of module parameters
        :param dropout: Dropout value
        """
        super().__init__()
        self.d_mem = d_mem

        self.dense_network_for_embeddings = DenseNetwork(
            n_hid_layers=0,
            input_dim=d_embd,
            out_dim=d_mem,
            dropout=dropout,
        )

        self.ln_1 = nn.LayerNorm(d_mem)
        self.self_attn = nn.MultiheadAttention(d_mem, n_head, dropout, batch_first=True)

        self.ln_2 = nn.LayerNorm(d_mem)
        self.enc_dec_attn = nn.MultiheadAttention(d_mem, n_head, dropout, batch_first=True)

        self.ln_3 = nn.LayerNorm(d_mem)
        self.mlp = DenseNetwork(n_hid_layers=1, input_dim=d_mem, hidden_dim=d_mem * 2, out_dim=d_mem, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        memory: torch.tensor,
        embeddings: torch.tensor,
        attention_mask: torch.Tensor,
    ) -> torch.tensor:
        embeddings = self.dense_network_for_embeddings(embeddings)

        residual = memory
        memory = self.ln_1(memory)
        memory = self.self_attn(query=memory, key=memory, value=memory)[0] + residual

        residual = memory
        memory = self.ln_2(memory)
        attention_mask = (1 - attention_mask).to(memory.device, dtype=torch.bool)
        attn_output = self.enc_dec_attn(
            query=memory, key=embeddings, value=embeddings, key_padding_mask=attention_mask
        )[0]
        memory = attn_output + residual

        residual = memory
        memory = self.ln_3(memory)
        feed_forward_hidden_states = self.mlp(memory)
        memory = feed_forward_hidden_states + residual

        return memory
