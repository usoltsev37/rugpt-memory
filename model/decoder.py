import copy
import torch
import torch.nn as nn

from model.dense import MLP, DenseNetwork


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Decoder(nn.Module):
    """
    Decoder for memory
    """

    def __init__(self, block: nn.Module, n_block: int = 3) -> None:
        """
        :param block: decoder block
        :param n_block: number of blocks in decoder
        """
        super().__init__()
        self.blocks = get_clones(block, n_block)
        self.norm = nn.LayerNorm(block.n_mem)

    def forward(self, mem: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        :param mem: matrix of memory embeddings with size [batch, mem_size, n_mem]
        :param x: matrix of sentence embeddings with size [batch, seq_lem, n_embd]
        :return: matrix of updated memory embeddings with size [batch, mem_size, n_mem]
        """
        for block in self.blocks:
            mem = block(mem, x)
        return self.norm(mem)


class DecoderBlock(nn.Module):

    def __init__(self, n_embd: int, n_mem: int, n_head: int, dropout: float, **kwargs) -> None:
        """
        :param n_embd: main model embedding size
        :param n_mem: memory embedding size
        :param n_head: number of heads in the attention mechanism
        :param dropout: dropout value
        :key dec_n_hid: hidden memory embedding size in MLP
        :key emb_n_hid: hidden sentences embedding size in MLP
        """
        super().__init__()
        self.n_mem = n_mem
        # TODO: config for dense network
        self.fc_inp = DenseNetwork(...)

        self.ln_1 = nn.LayerNorm(n_mem)
        self.attn = nn.MultiheadAttention(n_mem, n_head, dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_mem)
        self.mlp = MLP(n_mem, kwargs.get("dec_n_hid", 4 * n_mem), dropout)
        self.res_drop = nn.Dropout(dropout)

    def forward(self, mem: torch.tensor, inp: torch.tensor) -> torch.tensor:
        inp = self.fc_inp(inp)
        mem = self.ln_1(mem)
        mem = mem + self.res_drop(self.attn(mem, inp, inp)[0])
        return mem + self.res_drop(self.mlp(self.ln_2(mem)))
