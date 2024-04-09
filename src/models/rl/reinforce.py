import pickle

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from src.models.rl.agent import Agent
from src.models.rl.utils import Action, State
from src.utils.train_config import RLParams


class REINFORCE:
    def __init__(self, agent: Agent,
                 optimizer: torch.optim,
                 train_config: RLParams):
        self.device = torch.device(train_config.device)
        self.agent = agent.to(self.device)
        self.optim = optimizer

        self.clip = train_config.clip
        self.batches_per_update = train_config.batches_per_update
        self.batch_size = train_config.batch_size

        self.entropy_coef = train_config.entropy_coef
        self.clip_grad_norm = train_config.clip_grad_norm
        self.kl_target = train_config.kl_target

    def act(self, state: State) -> (Action, torch.Tensor, dict):
        """
        Takes a state as input and returns the action, its probability, and the distributions used for sampling the
        action.
        :param state: the current state from which the agent decides its action.
        :return: the action taken, the probability of the action, and the distributions used for sampling the action.
        """
        with torch.no_grad():
            action, proba, distr = self.agent.act(state.to(self.device))
        return action.to(torch.device("cpu")), proba.cpu(), distr

    def _get_distr_from_list(self, distr_batch: list[dict], ids: [int]) -> dict:
        selected_distributions = [distr_batch[i] for (i, _) in ids]
        probs = torch.stack([d["pos_distr"].probs[j] for d, (_, j) in zip(selected_distributions, ids)])
        loc = torch.stack([d["normal_distr"].loc[j] for d, (_, j) in zip(selected_distributions, ids)])
        scale = torch.stack([d["normal_distr"].scale[j] for d, (_, j) in zip(selected_distributions, ids)])
        pos_distr_cls = Categorical if self.agent.memory_type == "conservative" else Bernoulli
        pos_distr = pos_distr_cls(probs.detach())
        normal_distr = Normal(loc.detach(), scale.detach())

        return {"pos_distr": pos_distr, "normal_distr": normal_distr}

    @staticmethod
    def select_state_batch(states: [State], ids: [(int, int)]) -> State:
        selected_states = [states[i] for (i, _) in ids]
        embeddings = torch.stack([s.embeddings[j] for s, (_, j) in zip(selected_states, ids)])
        attention_mask = torch.stack([s.attention_mask[j] for s, (_, j) in zip(selected_states, ids)])
        memory = torch.stack([s.memory[j] for s, (_, j) in zip(selected_states, ids)])
        return State(memory.detach(), embeddings.detach(), attention_mask.detach())

    @staticmethod
    def select_action_batch(actions: [State], ids: [(int, int)]) -> Action:
        selected_actions = [actions[i] for (i, _) in ids]
        positions = torch.hstack([a.positions[j] for a, (_, j) in zip(selected_actions, ids)])
        memory_vectors = torch.stack([a.memory_vectors[j] for a, (_, j) in zip(selected_actions, ids)])
        return Action(positions.detach(), memory_vectors.detach())

    def update(self, transitions: [list]) -> float | None:
        """
        Updates the agent's policy based on collected transitions.

        :param transitions: a list of transitions.
        :return: the average loss of the update.
        """
        state, action, reward, old_proba, old_distr = zip(*transitions)
        num_transitions = len(state)
        steps = np.cumsum([s.batch_size for s in state])
        losses = []
        for _ in range(self.batches_per_update):
            first_idx = np.random.choice(range(num_transitions), self.batch_size, replace=True)
            sampled_batches = [state[i].batch_size for i in first_idx]
            second_idx = [np.random.randint(0, bs) for bs in sampled_batches]
            ids = list(zip(first_idx, second_idx))
            state_batch = self.select_state_batch(state, ids).to(self.device)
            action_batch = self.select_action_batch(action, ids).to(self.device)
            reward_batch = torch.stack([reward[i][j] for (i, j) in ids]).detach().to(self.device)
            old_proba_batch = torch.stack([old_proba[i][j] for (i, j) in ids]).detach().to(self.device)
            old_distr_batch = self._get_distr_from_list(old_distr, ids)

            cur_proba, entropy, distr = self.agent.compute_logproba_and_entropy(state_batch, action_batch)

            with torch.no_grad():
                kld = self.agent.compute_kld(old_distr_batch, distr)

            if self.kl_target is not None and torch.mean(kld) > self.kl_target:
                print(f"Early stopping! KLD is {torch.mean(kld)} on iteration {_ + 1}")
                return np.mean(losses) if losses else None

            ratio = torch.exp(cur_proba - old_proba_batch)

            loss = -torch.min(ratio * reward_batch, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * reward_batch)
            loss -= self.entropy_coef * entropy
            loss = loss.mean()
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
            self.optim.step()

            losses.append(loss.item())

        return np.mean(losses)

    def save(self):
        with open("agent.pkl", "wb") as f:
            pickle.dump(self.agent, f)
