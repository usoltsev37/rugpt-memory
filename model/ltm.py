import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
from model.dense import DenseNetwork


class LTMBlock(nn.Module):
    def __init__(self, d_embd: int, d_mem: int, transformer_block: nn.Module):
        super().__init__()
        self.d_embd = d_embd
        self.d_mem = d_mem
        self.transformer_block = transformer_block
        self.dense_for_memory = DenseNetwork(n_hid_layers=1, input_dim=d_mem, hidden_dim=d_mem * 2, out_dim=d_embd)
        self.dense_after_concat = DenseNetwork(n_hid_layers=1, input_dim=d_embd * 2, hidden_dim=d_embd * 2,
                                               out_dim=d_embd)
        self.attn = nn.MultiheadAttention(d_embd, 4, 0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(d_embd)
        self.ln2 = nn.LayerNorm(d_embd)

    def forward(self, input: torch.Tensor, memory: torch.Tensor):
        transformer_block_out = self.transformer_block(input)
        memory = self.dense_for_memory(memory)
        attn = self.ln1(self.attn(transformer_block_out[0], memory, memory)[0])
        after_concat = torch.cat((transformer_block_out[0], attn), dim=-1)
        after_dense = self.dense_after_concat(after_concat)
        out = self.ln2(transformer_block_out[0] + after_dense)
        return out


class LTMModel(nn.Module):
    def __init__(self, d_embd: int, d_mem: int, max_seq: int = 2048):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.ltm_layer = LTMBlock(d_embd, d_mem, self.model.transformer.h[-1])

        self.d_embd = d_embd
        self.d_mem = d_mem
        self.max_seq = max_seq

    def get_embeddings(self, x: torch.Tensor):
        """Get high-level embeddings"""
        with torch.no_grad():
            all_hidden_states = self.model.transformer(x, output_hidden_states=True).hidden_states
        return all_hidden_states[-2]

    def get_reward(self, labels: torch.Tensor, embeddings: torch.Tensor, memory: torch.Tensor):
        with torch.no_grad():
            out = self.ltm_layer(embeddings, memory)
            logits = self.model.lm_head(out)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduce=False)

            loss = loss_fct(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1))
            loss = loss.view(labels.shape[0], embeddings.shape[1] - 1).mean(dim=1)

            return loss
