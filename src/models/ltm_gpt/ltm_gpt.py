import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from src.models.ltm_gpt.ltm_gpt2_block import LTMGPT2Block


class LTM_GPT(GPT2LMHeadModel):
    """ Custom LTM GPT2 layer with memory """

    def __init__(self,
                 model_: GPT2LMHeadModel,
                 d_mem: int,
                 cnt_blocks_with_memory=2):
        super().__init__(model_.config)

        self.labels = None
        self.max_seq_length = model_.config.n_ctx
        self.transformer_ltm_blocks = nn.ModuleList([
            LTMGPT2Block(model_.transformer.h[-cnt_blocks_with_memory + i], d_mem) for i in
            range(cnt_blocks_with_memory)
        ])
        self.transformer = model_.transformer
        self.transformer.h = model_.transformer.h[:-cnt_blocks_with_memory]

        self.lm_head = model_.lm_head

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
                   reward_for_agent: bool = False):

        for block in self.transformer_ltm_blocks:
            hidden_states = block(hidden_states, attention_mask, memory)

        if not reward_for_agent:
            lm_logits = self.lm_head(hidden_states)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = self.labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=0)

            loss = loss_fct(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1))
            loss = loss.view(self.labels.shape[0], self.hidden_states.shape[1] - 1).mean()
            return loss
        else:
            # todo: add mean cross entropy over text
            lm_logits = self.lm_head(hidden_states)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = self.labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=0)

            loss = loss_fct(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1))
            loss = loss.view(self.labels.shape[0], self.hidden_states.shape[1] - 1).mean(dim=1)
            return loss

    def freeze(self) -> None:
        self.eval()
        for p in self.transformer_ltm_blocks.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.train()
        for p in self.transformer_ltm_blocks.parameters():
            p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True
