import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from src.models.rl.agent import Agent
from src.models.rl.utils import Action, State
from src.utils.logger_singleton import logger
from src.utils.train_config import RLParams

torch.autograd.set_detect_anomaly(True)

# todo(jbelova): make class from distr


def distr_to_device(distr: dict, device: torch.device, memory_type: str):
    probs = distr["pos_distr"].probs.to(device)
    loc = distr["normal_distr"].loc.to(device)
    scale = distr["normal_distr"].scale.to(device)
    pos_distr_cls = Categorical if memory_type == "conservative" else Bernoulli
    pos_distr = pos_distr_cls(probs)
    normal_distr = Normal(loc, scale)
    return {"pos_distr": pos_distr, "normal_distr": normal_distr}


class REINFORCE:
    def __init__(self, agent: Agent, optimizer: torch.optim, train_config: RLParams):

        self.agent = agent
        self.optim = optimizer
        self.device = self.agent.device

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
            state.to(self.device)
            action, proba, distr = self.agent.act(state=state)
            state.to("cpu")

        # todo: distr to cpu
        action.to("cpu")
        return action, proba.cpu(), distr_to_device(distr, torch.device("cpu"), memory_type="conservative")

    def _get_distr_from_list(self, distr_batch: list[dict], ids: [int]) -> dict:
        selected_distributions = [distr_batch[i] for (i, _) in ids]
        probs = torch.stack([d["pos_distr"].probs[j] for d, (_, j) in zip(selected_distributions, ids)]).to(self.device)
        loc = torch.stack([d["normal_distr"].loc[j] for d, (_, j) in zip(selected_distributions, ids)]).to(self.device)
        scale = torch.stack([d["normal_distr"].scale[j] for d, (_, j) in zip(selected_distributions, ids)]).to(
            self.device
        )
        pos_distr_cls = Categorical if self.agent.memory_type == "conservative" else Bernoulli
        pos_distr = pos_distr_cls(probs.detach())
        normal_distr = Normal(loc.detach(), scale.detach())
        return {"pos_distr": pos_distr, "normal_distr": normal_distr}

    @staticmethod
    def select_state_batch(states: [State], ids: [(int, int)]) -> State:
        selected_states = [states[i] for (i, _) in ids]
        embeddings = torch.stack(
            [s.embeddings[j] for s, (_, j) in zip(selected_states, ids)],
        )
        attention_mask = torch.stack([s.attention_mask[j] for s, (_, j) in zip(selected_states, ids)])
        memory = torch.stack([s.memory[j] for s, (_, j) in zip(selected_states, ids)])
        return State(memory.detach(), embeddings.detach(), attention_mask.detach())

    @staticmethod
    def select_action_batch(actions: [State], ids: [(int, int)]) -> Action:
        selected_actions = [actions[i] for (i, _) in ids]
        positions = torch.hstack([a.positions[j] for a, (_, j) in zip(selected_actions, ids)])
        memory_vectors = torch.stack([a.memory_vectors[j] for a, (_, j) in zip(selected_actions, ids)])
        return Action(positions.detach(), memory_vectors.detach())

    def update(self, transitions: list[list]) -> float | None:
        """
        Updates the agent's policy based on collected transitions.

        :param transitions: a list of transitions.
        :return: the average loss of the update.
        """
        state, action, reward, old_proba, old_distr = zip(*transitions)
        bs, num_transitions = state[0].memory.shape[0], len(state)
        losses = []

        for step in range(self.batches_per_update):
            ids = np.random.choice(range(num_transitions * bs), self.batch_size, replace=False)
            ids = [(idx // bs, idx % bs) for idx in ids]

            state_batch = self.select_state_batch(state, ids)
            action_batch = self.select_action_batch(action, ids)
            reward_batch = torch.stack([reward[i][j] for (i, j) in ids]).detach().to(self.device)
            old_proba_batch = torch.stack([old_proba[i][j] for (i, j) in ids]).detach().to(self.device)
            old_distr_batch = self._get_distr_from_list(old_distr, ids)

            cur_proba, entropy, distr = self.agent.compute_logproba_and_entropy(state_batch, action_batch)

            with torch.no_grad():
                kld = self.agent.compute_kld(old_distr_batch, distr)

            if step > 0 and self.kl_target is not None and torch.mean(kld) > self.kl_target:
                logger.warning(f"Early stopping! KLD is {torch.mean(kld)} on iteration {step + 1}")
                return np.mean(losses) if losses else None

            ratio = torch.exp(cur_proba - old_proba_batch)

            loss = -torch.min(
                ratio * reward_batch,
                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * reward_batch,
            )

            loss -= self.entropy_coef * entropy
            loss = loss.mean()
            loss.backward()

            nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
            self.optim.step()
            self.optim.zero_grad()

            losses.append(loss.item())

        return np.mean(losses)
