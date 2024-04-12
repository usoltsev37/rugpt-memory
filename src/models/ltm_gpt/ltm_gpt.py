import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from src.models.ltm_gpt.ltm_gpt2_block import LTMGPT2Block


class LTM_GPT(nn.Module):
    """ Custom LTM GPT2 layer with memory """

    def __init__(self,
                 model_: GPT2LMHeadModel,
                 d_mem: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 cnt_blocks_with_memory: int = 2,
                 ignore_index: int = 0) -> None:
        super().__init__()

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 2

        self.labels = None
        self.max_seq_length = model_.config.n_ctx
        self.dtype = dtype
        self.ignore_index = ignore_index

        self.model_ = model_
        self.transformer = self.model_.transformer
        self.transformer.h = self.model_.transformer.h[:-cnt_blocks_with_memory]

        self.transformer_ltm_blocks = nn.ModuleList([
            LTMGPT2Block(self.model_.transformer.h[-cnt_blocks_with_memory + i], d_mem, dtype=dtype) for i in
            range(cnt_blocks_with_memory)
        ]).to(device)

        self.transformer.h = self.model_.transformer.h[:-cnt_blocks_with_memory]
        self.lm_head = self.model_.lm_head

        self.first_device = next(self.transformer.parameters()).device
        self.second_device = device

    def get_embeddings(self,
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        self.hidden_states = self.transformer(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              output_hidden_states=True).last_hidden_state
        self.labels = input_ids
        return self.hidden_states

    def get_output(self,
                   hidden_states: torch.tensor,
                   attention_mask: torch.tensor,
                   memory: torch.Tensor,
                   reward_for_agent: bool = False) -> torch.Tensor:

        for block in self.transformer_ltm_blocks:
            hidden_states = block(hidden_states, attention_mask, memory)

        lm_logits = self.lm_head(hidden_states)

        if not reward_for_agent:
            self.labels = self.labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = self.labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        else:
            # todo: add mean cross entropy over text
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = self.labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

            loss = loss_fct(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1))
            loss = loss.view(self.labels.shape[0], self.hidden_states.shape[1] - 1).mean(dim=1)
            return loss

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.train()

        for p in self.lm_head.parameters():
            p.requires_grad = True

        for p in self.transformer.ln_f.parameters():
            p.requires_grad = True

        for name, param in self.transformer_ltm_blocks.named_parameters():
            if 'gpt2_block' in name:
                if 'lora' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = True
