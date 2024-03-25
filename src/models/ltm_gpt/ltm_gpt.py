import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
# from transformers.utils import (
#     ModelOutput,
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     logging,
#     replace_return_docstrings,
# )
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union

from src.models.ltm_gpt.ltm_gpt2_block import LTMGPT2Block


class LTM_GPT(GPT2LMHeadModel):
    """ Custom LTM GPT2 layer with memory """

    def __init__(self, model_: GPT2LMHeadModel, cnt_blocks_with_memory=2):
        super().__init__(model_.config)
        self.model_ = model_
        self.transformer = self.model_.transformer
        self.transformer.h = self.model_.transformer.h[:-cnt_blocks_with_memory]

        # class CastOutputToFloat(torch.nn.Sequential):
        #     def forward(self, x): return super().forward(x).to(torch.float32)
        #
        # self.transformer.h[-cnt_blocks_with_memory - 1] = CastOutputToFloat(self.transformer.h[-cnt_blocks_with_memory - 1])  # cast model ouputs to unfuct the top-k sampler

        self.transformer_ltm_blocks = nn.ModuleList([
            LTMGPT2Block(self.model_.transformer.h[-cnt_blocks_with_memory + i]) for i in range(cnt_blocks_with_memory)
        ]).cuda(device=self.model_.device)

        self.lm_head = self.model_.lm_head

        # Model parallel
        # self.model_parallel = False
        # self.device_map = None

        # Initialize weights and apply final processing
        # self.post_init()

    # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=CausalLMOutputWithCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.model_.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Init memory as hidden_states from 37 layers
        # BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=presents,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )

        print(len(transformer_outputs))
        # print(len(transformer_outputs))
        memory = transformer_outputs[2][-1]

        for block in self.transformer_ltm_blocks:
            block.update_memory(memory)
            hidden_states = block(hidden_states)

        # Set device for model parallelism
        if self.model_.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
