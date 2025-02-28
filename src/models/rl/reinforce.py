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


def distr_to_device(distr: dict, device: torch.device, memory_type: str = "conservative"):
    probs = distr["pos_distr"].probs.to(device)
    loc = distr["normal_distr"].loc.to(device)
    scale = distr["normal_distr"].scale.to(device)
    pos_distr_cls = Categorical if memory_type == "conservative" else Bernoulli
    pos_distr = pos_distr_cls(probs)
    normal_distr = Normal(loc, scale)
    return {"pos_distr": pos_distr, "normal_distr": normal_distr}


class REINFORCE:
    def __init__(
        self,
        agent: Agent,
        optimizer: torch.optim,
        train_config: RLParams,
        alpha: torch.nn.parameter.Parameter,
        alpha_optimizer: torch.optim,
    ):

        self.agent = agent
        self.optim = optimizer

        self.alpha = alpha
        self.alpha_optimizer = alpha_optimizer
        self.device = self.agent.device

        self.clip = train_config.clip
        self.batches_per_update = train_config.batches_per_update
        self.batch_size = train_config.batch_size

        self.clip_grad_norm = train_config.clip_grad_norm
        self.kl_target = train_config.kl_target
        self.target_entropy = train_config.target_entropy

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
        action.to("cpu")
        # todo: distr to cpu
        distr = distr_to_device(distr, torch.device("cpu"))

        return (action, proba.cpu(), distr)

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

    def update(self, transitions: list[list], tensorboard_writer, iter) -> float | None:
        """
        Updates the agent's policy based on collected transitions.

        :param transitions: a list of transitions.
        :return: the average loss of the update.
        """
        state, action, reward, old_proba, old_distr = zip(*transitions)
        bs, num_transitions = state[0].memory.shape[0], len(state)
        losses = []
        entropies = []
        rewards_ = []
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
                return np.mean(losses)

            ratio = torch.exp(cur_proba - old_proba_batch)

            loss = -torch.min(
                ratio * reward_batch,
                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * reward_batch,
            )

            loss -= torch.exp(self.alpha).item() * entropy

            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
            self.optim.step()
            self.optim.zero_grad()

            alpha_loss = self.alpha * (entropy.detach() - self.target_entropy).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha_optimizer.zero_grad()

            losses.append(loss.item())
            entropies.append(entropy.mean().item())
            # rewards_.append(reward_batch.mean().item())

        tensorboard_writer.add_scalar("Iteration Alpha", torch.exp(self.alpha).item(), iter)
        tensorboard_writer.add_scalar("Iteration Mean Entropy", np.mean(entropies), iter)
        # tensorboard_writer.add_scalar("Iteration Mean Reward", np.mean(rewards_), iter)

        return np.mean(losses)
