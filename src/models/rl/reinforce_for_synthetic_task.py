import pickle

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.optim import Adam
from train_config import RLArgs

from src.models.memory_model.memory_model import SyntheticTaskModel
from src.models.rl.agent import Agent
from src.models.rl.utils import Action, State


class REINFORCE:
    def __init__(self, model: SyntheticTaskModel,
                 train_config: RLArgs):
        self.device = torch.device(train_config.device)
        self.agent = Agent(model).to(self.device)
        self.optim = Adam(self.agent.parameters(), train_config.agent_lr)

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

    def _get_distr_from_list(self, distr_batch: list[dict]) -> dict:
        probs = torch.stack([d["pos_distr"].probs for d in distr_batch]).squeeze(1)
        loc = torch.stack([d["normal_distr"].loc for d in distr_batch]).squeeze(1)
        scale = torch.stack([d["normal_distr"].scale for d in distr_batch]).squeeze(1)

        pos_distr_cls = Categorical if self.agent.memory_type == "conservative" else Bernoulli
        pos_distr = pos_distr_cls(probs)
        normal_distr = Normal(loc, scale)

        return {"pos_distr": pos_distr, "normal_distr": normal_distr}

    def update(self, transitions: [tuple]) -> float | None:
        """
        Updates the agent's policy based on collected transitions.

        :param transitions: a list of transitions.
        :return: the average loss of the update.
        """
        transitions = [t for traj in transitions for t in traj]
        states, actions, rewards, old_proba, old_distr = zip(*transitions)

        old_proba = torch.stack(old_proba).squeeze(-1)
        rewards = torch.stack(rewards).squeeze(-1)

        losses = []
        for _ in range(self.batches_per_update):
            idx = np.random.choice(range(len(transitions)), self.batch_size, replace=False)

            memory = torch.stack([states[i].memory for i in idx]).squeeze(1)
            states_batch = State(memory, torch.empty(0))

            positions = torch.stack([actions[i].positions for i in idx]).squeeze(-1)
            vectors = torch.stack([actions[i].memory_vectors.double() for i in idx]).squeeze(1)
            actions_batch = Action(positions, vectors)

            old_proba_batch = old_proba[idx].clone().to(self.device).double()
            rewards_batch = rewards[idx].clone().to(self.device)
            old_distr_batch = self._get_distr_from_list([old_distr[i] for i in idx])

            cur_proba, entropy, distr = self.agent.compute_logproba_and_entropy(states_batch, actions_batch)

            with torch.no_grad():
                kld = self.agent.compute_kld(old_distr_batch, distr)

            if self.kl_target is not None and torch.mean(kld) > self.kl_target:
                print(f"Early stopping! KLD is {torch.mean(kld)} on iteration {_ + 1}")
                return np.mean(losses) if losses else None

            ratio = torch.exp(cur_proba - old_proba_batch)

            loss = -torch.min(ratio * rewards_batch, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * rewards_batch)
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
