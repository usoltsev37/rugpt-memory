import random

import torch
import torch.nn.functional as F

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.rl.utils import Action, State

from torch.distributions import Normal


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
        self.transformation_layer = None
        self.transform_matrix = torch.randn((self.memory_module.d_mem, self.ltm_model.d_embd))  # [d_mem, d_embd]

    def compute_dist(self, aggregate_fn: str = "min"):
        with torch.no_grad():
            transformed_embeddings = self.embeddings  # [num_vectors, d_embd]
            transformed_memory = torch.tanh((self.memory_module.memory @ self.transform_matrix)).unsqueeze(2)
            dists = torch.linalg.norm(transformed_memory - transformed_embeddings.unsqueeze(1).cpu(), dim=-1)
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

        selected_step = random.randint(0, n_steps - 1)
        input_ids, attention_mask = (
            data["input_ids"][:, selected_step, :].contiguous(),
            data["attention_mask"][:, selected_step, :].contiguous(),
        )

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.attention_mask = torch.ones_like(self.attention_mask)

        # Get embeddings for the first state
        with torch.no_grad():
            self.embeddings = self.ltm_model.get_embeddings(input_ids, attention_mask).to("cuda:0")

        self.memory_module.reset(bs)
        self.prev_dist = self.compute_dist(self.aggregate_fn).sum(-1)

        return State(
            self.memory_module.memory,
            self.embeddings,
            self.attention_mask,
        )

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        self.cur_step += 1
        self.memory_module.update(action=action)

        cur_dist = self.compute_dist(self.aggregate_fn).sum(-1)
        reward = self.prev_dist - cur_dist
        self.prev_dist = cur_dist

        done = True if self.cur_step == self.episode_max_steps else False
        return (State(self.memory_module.memory, self.embeddings, self.attention_mask), reward, done)
