import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from src.models.memory_model.memory_model import MemoryModel, SyntheticTaskModel
from src.models.rl.envs import Action, State


class Agent(nn.Module):
    """
    An agent that learns to process embeddings from a language model and effectively remember information by selecting
    the necessary changes in memory.
    """

    def __init__(self, memory_model: MemoryModel | SyntheticTaskModel) -> None:
        super().__init__()
        self.model = memory_model
        self.d_mem = memory_model.d_mem
        self.num_vectors = memory_model.num_vectors
        self.memory_type = memory_model.memory_type
        self.device = next(self.model.parameters()).device

    def _compute_log_probability(
        self, pos_distr: Categorical | Bernoulli, log_probability_normal_distr: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the logarithm of the probability of selecting specific memory positions and associated memory
        vector updates.

        :param pos_distr: the distribution of memory positions.
        :param log_probability_normal_distr: log probability of the normal distribution for memory vectors.
        :param positions: indices or binary selection of memory positions.
        :return: probability of action.
        """
        p_i, log_p_i = pos_distr.probs, pos_distr.logits
        a_i = positions
        log_q_i = log_probability_normal_distr

        if self.memory_type == "conservative":
            a_i = F.one_hot(a_i, self.num_vectors)
            log_probs = log_p_i + log_q_i
            masked_log_probs = log_probs * a_i
            return torch.sum(masked_log_probs, dim=-1)
        elif self.memory_type == "flexible":
            masked_log_probs = a_i * (log_p_i + log_q_i) + (1 - a_i) * torch.log(1 - p_i)
            return torch.sum(masked_log_probs, dim=-1)
        else:
            raise ValueError(f"Unsupported memory_type: {self.memory_type}")

    def _compute_entropy(self, pos_distr: Categorical | Bernoulli, normal_distr: Normal) -> torch.Tensor:
        """
        Calculates the entropy of the action distribution, which includes both the position and memory vector updates.

        :param pos_distr: the distribution for selecting memory positions.
        :param normal_distr: the normal distribution for memory vectors updates.
        :return: the total entropy of the action distribution.
        """

        normal_entropy = normal_distr.entropy().sum(-1)
        if self.memory_type == "conservative":
            cat_entropy = pos_distr.entropy()
            return cat_entropy + (pos_distr.probs * normal_entropy).sum(-1)
        elif self.memory_type == "flexible":
            bernoulli_entropy = pos_distr.entropy()
            return bernoulli_entropy.sum(-1) + (pos_distr.probs * normal_entropy).sum(-1)
        else:
            raise ValueError(f"Unsupported memory_type: {self.memory_type}")

    def compute_kld_between_normal_distributions(
        self, old_policy_distr: Normal, cur_policy_distr: Normal
    ) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence between two normal distributions.

        :param old_policy_distr: the normal distribution of the old policy.
        :param cur_policy_distr: the normal distribution of the current policy.
        :return: the Kullback-Leibler divergence between the two distributions.
        """
        sigma_current = cur_policy_distr.scale.pow(2)
        sigma_old = old_policy_distr.scale.pow(2)
        diff = old_policy_distr.loc - cur_policy_distr.loc
        inverse_sigma_current = 1 / sigma_current

        term1 = torch.sum(torch.log(sigma_current), dim=-1) - torch.sum(torch.log(sigma_old), dim=-1)
        term2 = torch.full_like(term1, self.d_mem)
        term3 = torch.sum(diff.pow(2) * inverse_sigma_current, dim=-1)
        term4 = torch.sum(inverse_sigma_current * sigma_old, dim=-1)

        kld = 0.5 * (term1 - term2 + term3 + term4)
        return kld

    def compute_kld(self, old_policy: dict, cur_policy: dict) -> torch.Tensor:
        """
        Calculate the Kullback-Leibler (KL) divergence between the old and current policy distributions.

        The KL divergence is computed as D_KL(P||Q) = H(P, Q) - H(P), where:
        - P is the distribution of actions for the old policy,
        - Q is the distribution of actions for the new (current) policy,
        - H(P, Q) is the cross-entropy between the distributions,
        - H(P) is the entropy of the old policy distribution.

        :param old_policy: a dictionary representing the old policy, with keys 'pos_distr' and
        'normal_distr' for distributions.
        :param cur_policy: a dictionary representing the current policy, similar structure to `old_policy`.
        """
        old_policy_entropy = self._compute_entropy(old_policy["pos_distr"], old_policy["normal_distr"])

        if self.memory_type == "conservative":
            kld_between_normal_distr = self.compute_kld_between_normal_distributions(
                old_policy["normal_distr"], cur_policy["normal_distr"]
            )

            cross_entropy_component_1 = -torch.sum(
                old_policy["pos_distr"].probs * cur_policy["pos_distr"].logits, dim=-1
            )
            cross_entropy_component_2 = torch.sum(
                old_policy["pos_distr"].probs
                * (kld_between_normal_distr + old_policy["normal_distr"].entropy().sum(-1)),
                dim=-1,
            )

            cross_entropy = cross_entropy_component_1 + cross_entropy_component_2
            kld = -old_policy_entropy + cross_entropy

            return kld
        elif self.memory_type == "flexible":
            # todo
            pass
        else:
            raise ValueError(f"Unsupported memory_type: {self.memory_type}")

    def act(self, state: State) -> tuple[Action, torch.tensor, dict]:
        """
        Determines the agent's action based on the current state, samples distributions for positions and memory
        vectors, and calculates the log probability of the selected action.

        :param state: the current state of the environment from which the agent decides its action.
        :return: a tuple containing the selected action, the log probability of the action, and the distributions used
        for sampling the action.
        """
        # Transfer State to self.device
        state.to(self.device)

        # Sample distribution parameters for positions and memory vectors based on the current state
        positions_param, mu, sigma = self.model(state)

        # Sample positions based on the memory type
        pos_distr_cls = Categorical if self.memory_type == "conservative" else Bernoulli
        pos_distr = pos_distr_cls(positions_param)
        positions = pos_distr.sample()

        # Sample memory vectors from a normal distribution
        sigma = torch.exp(sigma)
        normal_distr = Normal(mu, sigma)
        memory_vectors = normal_distr.sample()

        # Construct the action and calculate log probabilities
        action = Action(positions, memory_vectors, device=self.device)
        log_proba_normal_distr = normal_distr.log_prob(action.memory_vectors).sum(-1)
        distributions = {"pos_distr": pos_distr, "normal_distr": normal_distr}
        return action, self._compute_log_probability(pos_distr, log_proba_normal_distr, positions), distributions

    def compute_logproba_and_entropy(self, state: State, action: Action) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Computes the log probability of an action according to the current policy and the entropy of the action's
        distribution.

        :param state: the current state based on which the action was decided.
        :param action: the action taken by the agent.
        :return: a tuple containing the log probability of the action, the entropy of the action's distribution,
        and the distributions used for calculating both metrics.
        """
        state.to(self.device)
        action.to(self.device)

        positions_param, mu, sigma = self.model(state)

        pos_distr_cls = Categorical if self.memory_type == "conservative" else Bernoulli
        pos_distr = pos_distr_cls(positions_param)

        sigma = torch.exp(sigma)
        normal_distr = Normal(mu, sigma)
        log_proba_normal_distr = normal_distr.log_prob(action.memory_vectors).sum(-1)

        distributions = {"pos_distr": pos_distr, "normal_distr": normal_distr}

        log_proba = self._compute_log_probability(pos_distr, log_proba_normal_distr, action.positions)
        entropy = self._compute_entropy(pos_distr, normal_distr)
        return log_proba, entropy, distributions
