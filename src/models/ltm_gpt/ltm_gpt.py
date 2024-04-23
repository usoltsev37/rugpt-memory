import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from src.models.ltm_gpt.ltm_gpt2_block import LTMGPT2Block


class LTM_GPT(nn.Module):
    """Custom LTM GPT2 layer with memory"""

    def __init__(
        self,
        model_: GPT2LMHeadModel,
        d_mem: int,
        device: torch.device,
        dtype: torch.dtype,
        cnt_blocks_with_memory: int = 2,
        ignore_index: int = 0,
    ) -> None:
        super().__init__()

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 2

        self.labels = None
        self.max_seq_length = model_.config.n_ctx
        self.dtype = dtype
        self.ignore_index = ignore_index

        self.transformer = model_.transformer

        self.transformer_ltm_blocks = nn.ModuleList(
            [
                LTMGPT2Block(copy.deepcopy(self.transformer.h[-cnt_blocks_with_memory + i]), d_mem, dtype=dtype)
                for i in range(cnt_blocks_with_memory)
            ]
        ).to(device)

        self.transformer.h = self.transformer.h[:-cnt_blocks_with_memory]

        self.lm_head = model_.lm_head

        self.first_device = next(self.transformer.parameters()).device
        self.second_device = device

        self.trainable_parameters = self._get_trainable_parameters()

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).last_hidden_state

    def get_output(
        self,
        hidden_states: torch.tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.tensor,
        memory: torch.Tensor,
        reward_for_agent: bool = False,
    ) -> torch.Tensor:

        for block in self.transformer_ltm_blocks:
            hidden_states = block(hidden_states, attention_mask, memory)

        lm_logits = self.lm_head(hidden_states)

        if not reward_for_agent:

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        else:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_ids.shape[0], hidden_states.shape[1] - 1)

            mask = loss != 0
            not_ignore_index_count = torch.sum(mask, dim=1)
            loss = loss.sum(dim=-1) / not_ignore_index_count
            return -loss

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.train()

        for p in self.transformer.ln_f.parameters():
            p.requires_grad = True

        for p in self.lm_head.parameters():
            p.requires_grad = True

        for p in self.transformer_ltm_blocks.parameters():
            p.requires_grad = True

    def _get_trainable_parameters(self) -> list[str]:
        self.unfreeze()
        return [n for n, p in self.named_parameters() if p.requires_grad]
