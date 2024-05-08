import torch
from torch import nn

from src.models.memory_model.dense import DenseNetwork

class LTMGPT2Block(nn.Module):
    """Custom LTMGPT2Block layer with memory"""

    def __init__(
        self,
        gpt2_block: nn.Module,
        d_mem: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.gpt2_block = gpt2_block
        self.d_mem = d_mem
        self.dtype = dtype
        self.embed_dim = self.gpt2_block.ln_1.normalized_shape[0]

        assert dtype in [torch.float16, torch.float32]

        self.dense_network1 = DenseNetwork(
            n_hid_layers=0,
            input_dim=d_mem,
            out_dim=self.embed_dim,
            dropout=dropout,
            initialize_with_zeros=False,
            dtype=self.dtype,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, dtype=self.dtype
        )

        self.ln1 = nn.LayerNorm(self.embed_dim, dtype=self.dtype)

        self.dense_network2 = DenseNetwork(
            n_hid_layers=1,
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim // 8,
            out_dim=self.embed_dim,
            dropout=dropout,
            initialize_with_zeros=True,
            dtype=self.dtype,
        )

        self.ln2 = nn.LayerNorm(self.embed_dim, dtype=self.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask[:, None, None, :].contiguous()
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.to(device=hidden_states.device)
        
        query = self.gpt2_block(hidden_states=hidden_states, attention_mask=attention_mask)[0]
        residual = query

        # DenseNetwork
        memory = self.dense_network1(memory)

        # MultiHead Attention
        key, value = memory, memory
        x, _ = self.attn(query=query, key=key, value=value)

        # Norm & Concat
        x = x + residual
        x = self.ln1(x)

        # DenseNetwork initialized with zeroes
        x = self.dense_network2(x)

        # Norm & Sum
        x = x + residual
        
        x = self.ln2(x)

        return x
