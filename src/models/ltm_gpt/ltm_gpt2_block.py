import torch
from torch import nn

from src.models.ltm_gpt.dense_network import DenseNetwork


class LTMGPT2Block(nn.Module):
    """ Custom LTMGPT2Block layer with memory """

    def __init__(
            self,
            gpt2_block,
            num_heads=4,
            attn_dropout=0.1,
            dense_network_hidden_size=10240,
            dtype=torch.float32
    ):
        super().__init__()
        self.gpt2_block = gpt2_block

        self.embed_dim = self.gpt2_block.ln_1.normalized_shape[0]
        self.dense_network_hidden_size = dense_network_hidden_size

        assert dtype in [torch.float16, torch.float32]

        # self.memory: ( , , ) / (target_sentence_length, batch_size, self.embed_dim) (5120) | torch.FloatTensor / nn.Embedding
        self.memory = None

        # goal: convert memory from ( , , ) to (source_sentence_length, batch_size, self.embed_dim)
        self.dense_network1 = DenseNetwork(
            embed_dim=self.embed_dim,
            hidden_size=self.dense_network_hidden_size,
            dtype=dtype,
            initialize_with_zeros=False
        )

        self.attn = nn.MultiheadAttention(  # TODO masked ????
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=False,
            dtype=dtype
        )

        self.ln1 = nn.LayerNorm(self.embed_dim, dtype=dtype)

        self.dense_network2 = DenseNetwork(
            embed_dim=self.embed_dim,
            hidden_size=self.dense_network_hidden_size,
            dtype=dtype,
            initialize_with_zeros=True
        )

        self.ln2 = nn.LayerNorm(self.embed_dim, dtype=dtype)

    def forward(self, x):  # x: (sentence_length, batch_size, self.embed_dim)
        assert self.memory is not None

        # TransformerBlock
        query = self.gpt2_block(x)  # query: (sentence_length, batch_size, self.embed_dim)
        residual = query

        # DenseNetowork
        memory = self.dense_network1(self.memory)

        # MultiHead Attention
        key, value = memory, memory
        x, _ = self.attn(
            query=query,
            key=key,
            value=value
        )

        # Norm & Concat
        x = x + residual
        if self.dtype == torch.float16:
            x = self.ln1(x.float()).type(torch.float16)
        else:
            x = self.ln1(x)

        # DenseNetowork initialized with zeroes
        x = self.dense_network2(x)

        # Norm & Concat
        x = x + residual
        if self.dtype == torch.float16:
            x = self.ln2(x.float()).type(torch.float16)
        else:
            x = self.ln2(x)

        return x

    def update_memory(self, new_memory):
        self.memory = new_memory
