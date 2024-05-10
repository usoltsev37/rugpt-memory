import random

import torch
import torch.nn.functional as F

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.rl.utils import Action, State

from torch.distributions import Normal


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

    def __init__(self, ltm_model: LTM_GPT, memory_module: MemoryModule, num_prefixes_for_reward_calc: int) -> None:
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


class PretrainEnv:

    def __init__(self, ltm_model: LTM_GPT, memory_module: MemoryModule, episode_max_steps: int, args) -> None:
        self.attention_mask = None
        self.data = None
        self.embeddings = None
        self.n_steps = None
        self.cur_step = None
        self.iterator = None
        self.cur_state = None
        self.prev_dist = None

        self.ltm_model = ltm_model
        self.memory_module = memory_module
        self.episode_max_steps = episode_max_steps
        self.aggregate_fn = "min"
        self.step_length = args.ltm_params.step_length
        self.global_step = 0
        # self.transform_matrix = torch.eye((self.memory_module.d_mem))
        self.transform_matrix = torch.randn((self.ltm_model.d_embd, self.memory_module.d_mem))  # [d_mem, d_embd]
        # self.transform_matrix = torch.ones((self.memory_module.d_mem, self.ltm_model.d_embd))

    def compute_dist(self, aggregate_fn: str = "min"):
        with torch.no_grad():
            # transformed_memory = self.memory_module.memory @ self.transform_matrix  # [num_vectors, d_embd]
            transformed_memory = self.memory_module.memory  # [num_vectors, d_embd]
            transformed_memory = transformed_memory.unsqueeze(2)
            dists = torch.linalg.norm(transformed_memory - (self.embeddings @ self.transform_matrix.to("cuda:0")).unsqueeze(1).cpu(), dim=-1)
        # dists = 1.0 - F.cosine_similarity(transformed_memory, self.embeddings.unsqueeze(1).cpu(), dim=-1)
        if aggregate_fn == "min":
            return torch.min(dists, -1).values
        elif aggregate_fn == "max":
            return torch.max(dists, -1).values
        elif aggregate_fn == "mean":
            return torch.mean(dists, -1)

    def reset(self, data: dict) -> State:
        self.global_step += 1

        # Return embeddings
        bs, n_steps, _ = data["input_ids"].shape
        self.cur_step = 0
        self.pos = [set() for _ in range(bs)]

        selected_step = random.randint(0, n_steps - 1)
        input_ids, attention_mask = (
            data["input_ids"][:, selected_step, :].contiguous(),
            data["attention_mask"][:, selected_step, :].contiguous(),
        )

        self.input_ids = input_ids
        # self.attention_mask = attention_mask

        # Get embeddings for the first state
        with torch.no_grad():
            self.embeddings = self.ltm_model.get_embeddings(input_ids, attention_mask).to("cuda:0")
        # d = Normal(loc=1.0, scale=0.1)
        # self.embeddings = d.sample((bs, self.step_length, self.memory_module.d_mem))
        # self.embeddings = torch.ones_like(self.embeddings)
        self.attention_mask = torch.ones_like(attention_mask)

        self.memory_module.reset(bs)
        self.prev_dist = self.compute_dist(self.aggregate_fn).sum(-1)

        # Dummy reward
        self.pos = [set() for _ in range(bs)]
        
        return State(
            self.memory_module.memory,
            self.embeddings,
            self.attention_mask,
        )

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        self.cur_step += 1
        self.memory_module.update(action=action)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
        # reward = torch.zeros_like(action.positions).to(torch.float32)
        # for i, pos in enumerate(action.positions):
        #     if pos.item() in self.pos[i]:
        #         reward[i] = -5
        #     self.pos[i].add(pos.item())
        
        # bs = action.positions.shape[0]
        # chosen_vectors = action.memory_vectors[torch.arange(bs), action.positions]
        # reward -= torch.sum((chosen_vectors - 1.).pow(2.), dim=-1)
        # # reward -= torch.var(chosen_vectors, dim=-1) * 10.
        
        cur_dist = self.compute_dist(self.aggregate_fn).sum(-1)
        reward =  (self.prev_dist - cur_dist)
        self.prev_dist = cur_dist

        done = True if self.cur_step == self.episode_max_steps else False
        return (State(self.memory_module.memory, self.embeddings, self.attention_mask), reward, done)
