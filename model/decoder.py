import copy
import torch
import torch.nn as nn

from model.dense import MLP, DenseNetwork


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Decoder(nn.Module):
    """
    Decoder for the memory
    """

    def __init__(self, block: nn.Module, n_block: int = 3) -> None:
        """
        :param block: decoder block
        :param n_block: number of blocks in decoder
        """
        super().__init__()
        self.blocks = get_clones(block, n_block)
        self.norm = nn.LayerNorm(block.d_mem)

    def forward(self, memory: torch.tensor, embeddings: torch.tensor) -> torch.tensor:
        """
        :param memory: tensor of memory embeddings with size [batch, mem_size, n_mem]
        :param embeddings: tensor of sentence embeddings with size [batch, seq_lem, n_embd]
        :return: tensor of updated memory embeddings with size [batch, mem_size, n_mem]
        """
        for block in self.blocks:
            memory = block(memory, embeddings)
        return self.norm(memory)


class DecoderBlock(nn.Module):

    def __init__(self, d_embd: int, d_mem: int, d_hid: int, n_head: int, dropout: float,
                 dense_embd_args: dict) -> None:
        """
        :param d_embd: sentences embedding size
        :param d_mem: memory vector size
        :param n_head: number of heads in the attention mechanism
        :param dropout: dropout value
        :key d_hid: hidden memory vector size in MLP
        """
        super().__init__()
        self.d_mem = d_mem
        self.fc_inp = DenseNetwork(**dense_embd_args)
        self.ln_1 = nn.LayerNorm(d_mem)
        self.attn = nn.MultiheadAttention(d_mem, n_head, dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_mem)
        self.mlp = MLP(d_mem, d_hid, dropout)
        self.res_drop = nn.Dropout(dropout)

    def forward(self, mem: torch.tensor, inp: torch.tensor) -> torch.tensor:
        inp = self.fc_inp(inp)
        mem = self.ln_1(mem)
        mem = mem + self.res_drop(self.attn(mem, inp, inp)[0])
        return mem + self.res_drop(self.mlp(self.ln_2(mem)))
