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
            hidden_dim=self.embed_dim // 4,
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
        
        hidden_states = self.gpt2_block(hidden_states=hidden_states, attention_mask=attention_mask)[0]
        
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        memory = self.dense_network1(memory)
        
        attn_output = self.attn(query=hidden_states, key=memory, value=memory)[0]
        hidden_states = attn_output + residual

        hidden_states = self.ln2(hidden_states)
        feed_forward_hidden_states = self.dense_network2(hidden_states)

        hidden_states = feed_forward_hidden_states + residual
    
        return hidden_states
