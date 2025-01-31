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
        device: torch.device,
        cnt_blocks_with_memory: int = 2,
        ignore_index: int = 0,
    ) -> None:
        super().__init__()

        self.labels = None
        self.dtype = model_._dtype
        self.d_embd = model_.config.n_embd
        self.step_length = model_.step_length
        self.add_lora = model_.add_lora
        self.ignore_index = ignore_index

        self.transformer = model_.transformer
        self.first_device = next(self.transformer.parameters()).device
        self.second_device = device

        self.transformer_ltm_blocks = nn.ModuleList(
            [
                LTMGPT2Block(
                    copy.deepcopy(self.transformer.h[-cnt_blocks_with_memory + i]), model_.d_mem, dtype=self.dtype
                )
                for i in range(cnt_blocks_with_memory)
            ]
        ).to(device)

        self.transformer.h = self.transformer.h[:-cnt_blocks_with_memory]
        self.lm_head = model_.lm_head
        self.ln_f = copy.deepcopy(model_.transformer.ln_f).to(device)
        self.transform_matrix = nn.Linear(self.d_embd, model_.d_mem).to(self.second_device)

    def convert_tensor_to_first_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.first_device:
            tensor = tensor.to(self.first_device)
        return tensor

    def convert_tensor_to_second_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.second_device:
            tensor = tensor.to(self.second_device)
        return tensor

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = self.convert_tensor_to_first_device(input_ids)
        attention_mask = self.convert_tensor_to_first_device(attention_mask)
        return self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).last_hidden_state

    def get_output(
        self,
        embeddings: torch.tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.tensor,
        memory: torch.Tensor,
        reward_for_agent: bool = False,
        only_logits: bool = False
    ) -> torch.Tensor:

        input_ids = self.convert_tensor_to_second_device(input_ids)
        attention_mask = self.convert_tensor_to_second_device(attention_mask)
        embeddings = self.convert_tensor_to_second_device(embeddings)
        memory = self.convert_tensor_to_second_device(memory)
        # memory = torch.tanh(self.transform_matrix(memory))

        for block in self.transformer_ltm_blocks:
            embeddings = block(embeddings, attention_mask, memory)
        embeddings = self.ln_f(embeddings)
        lm_logits = self.lm_head(embeddings)
        lm_logits = lm_logits.to(self.second_device)
        
        if only_logits:
            return lm_logits.cpu()

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none" if reward_for_agent else "mean")

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if reward_for_agent:
            loss = -loss.view(input_ids.shape[0], embeddings.shape[1] - 1).sum(dim=-1) / (
                shift_labels != self.ignore_index
            ).sum(dim=-1)

        return loss

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, memory, reward_for_agent=False, only_logits=True):
        embeddings = self.get_embeddings(input_ids, attention_mask)
        return self.get_output(embeddings, input_ids, attention_mask, memory, reward_for_agent, only_logits), embeddings

    def freeze(self) -> None:
        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.train()

        for n, p in self.transformer_ltm_blocks.named_parameters():
            if "gpt2_block" not in n:
                p.requires_grad = True

        # for p in self.transform_matrix.parameters():
        #     p.requires_grad = True