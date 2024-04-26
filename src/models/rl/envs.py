import random

import torch
import torch.nn.functional as F

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.rl.utils import Action, State


def _crop_batch(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_len is not None:
        crop_by_len = random.randint(2, max_len)
    else:
        crop_by_len = random.randint(2, input_ids.shape[1])
    return input_ids[:, :crop_by_len], attention_mask[:, :crop_by_len]


class LTMEnvironment:
    """Memory model training environment. The reward and state come from the LTM model."""

    def __init__(
        self,
        ltm_model: LTM_GPT,
        memory_module: MemoryModule,
        num_prefixes_for_reward_calc: int,
    ) -> None:
        self.attention_mask = None
        self.data = None
        self.embeddings = None
        self.n_steps = None
        self.cur_step = None
        self.iterator = None
        self.cur_state = None

        self.ltm_model = ltm_model
        self.step_length = ltm_model.step_length
        self.memory_module = memory_module
        self.num_prefixes_for_reward_calc = num_prefixes_for_reward_calc

    def reset(self, data: dict) -> State:
        """Returns a new state in the form of empty memory and high-level embeddings of text (max_seq_len)
        received from LTM model.
        :param data: tokenized article with contexts with size [1, num_tokens]
        :return: first observation - tuple of the memory and high-level embeddings
        """
        # Return new state
        bs, n_steps, _ = data["input_ids"].shape
        self.data = data
        self.cur_step = 0
        self.n_steps = n_steps

        input_ids, attention_mask = (
            data["input_ids"][:, self.cur_step, :].contiguous(),
            data["attention_mask"][:, self.cur_step, :].contiguous(),
        )

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        # Get embeddings for the first state
        self.embeddings = self.ltm_model.get_embeddings(input_ids, attention_mask)

        self.memory_module.reset(bs)

        return State(
            self.memory_module.memory,
            self.embeddings,
            self.attention_mask,
        )

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        self.cur_step += 1

        reward = self.ltm_model.get_output(
            self.embeddings,
            self.input_ids,
            self.attention_mask,
            self.memory_module.memory,
            reward_for_agent=True,
        )

        # Try other prefixes
        for _ in range(self.num_prefixes_for_reward_calc - 1):
            input_ids, attention_mask = _crop_batch(self.input_ids, self.attention_mask, self.attention_mask[0].sum(-1))
            prefix_reward, _ = self.ltm_model(
                input_ids, attention_mask, self.memory_module.memory, reward_for_agent=True
            )

            reward += prefix_reward

        reward /= self.num_prefixes_for_reward_calc

        self.memory_module.update(action)

        if self.cur_step >= self.n_steps:
            done = True
            self.embeddings = torch.zeros_like(self.embeddings)
            self.attention_mask = torch.zeros_like(self.attention_mask)
        else:
            done = False
            input_ids, attention_mask = (
                self.data["input_ids"][:, self.cur_step, :].contiguous(),
                self.data["attention_mask"][:, self.cur_step, :].contiguous(),
            )

            embeddings = self.ltm_model.get_embeddings(
                input_ids,
                attention_mask,
            )

            self.input_ids = input_ids
            self.embeddings = embeddings
            self.attention_mask = attention_mask

        return (
            State(self.memory_module.memory, self.embeddings, self.attention_mask),
            reward.cpu(),
            done,
        )
