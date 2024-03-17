import torch
import torch.nn.functional as F

from model.ltm import LTMModel
from model.memory import MemoryModule

from config import SyntheticTaskEnvParams

from rl.utils import Action, State


class LTMEnvironment:
    """Memory model training environment. The reward and state come from the LTM model."""

    def __init__(self,
                 ltm_model: LTMModel,
                 memory: MemoryModule) -> None:

        self.ltm_model = ltm_model
        self.memory = memory
        self.max_seq = ltm_model.max_seq
        self.iterator = None
        self.cur_state = None

    def reset(self, data: torch.Tensor) -> State:
        """Returns a new state in the form of empty memory and high-level embeddings of new texts (max_seq_len)
         received from LTM model.
        :return: first observation - tuple of the memory and high-level embeddings
        """
        batch_size = data.shape[0]
        self.iterator = iter(torch.split(data, self.max_seq, dim=-1))
        self.cur_data = next(self.iterator)
        embeddings = self.ltm_model.get_embeddings(self.cur_data)
        memory = self.memory.reset(batch_size)
        self.cur_state = State(memory, embeddings)
        return self.cur_state

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        new_memory = self.memory.update(action)
        reward = self.ltm_model.get_reward(self.cur_data, self.cur_state.embeddings, new_memory)  # [batch_size]
        done = False
        try:
            self.cur_data = next(self.iterator)
            embeddings = self.ltm_model.get_embeddings(self.cur_data)

            if embeddings.shape[1] != self.cur_state.embeddings.shape[1]:
                done = True
        except StopIteration:
            embeddings = torch.zeros_like(self.cur_state.embeddings)
            done = True

        self.cur_state = State(new_memory, embeddings)

        return self.cur_state, reward, done


class SyntheticTaskEnv:
    """Environment for testing the REINFORСE algorithm"""

    def __init__(self, env_params: SyntheticTaskEnvParams) -> None:
        self.cur_step = 0
        self.state = None
        self.pos = set()

        self.num_vectors = env_params.num_vectors
        self.d_mem = env_params.d_mem
        self.memory_type = env_params.memory_type
        self.max_steps = env_params.max_steps

    def reset(self) -> State:
        """
        Resets the environment to an initial state.
        :return: the initial state with zeroed memory.
        """
        self.state = State(torch.zeros((1, self.num_vectors, self.d_mem)),
                           torch.empty(0))  # [batch_size, num_vectors, d_mem]
        self.cur_step = 0
        self.pos.clear()
        return self.state

    def _get_reward(self, action: Action) -> torch.Tensor:
        """
        Computes the reward based on the action taken.
        :param action: the action taken by the agent.
        :return: the computed reward.
        """
        reward = torch.zeros(1)

        if self.memory_type == "conservative":
            chosen_position = action.positions[0].item()
            reward -= 5 if chosen_position in self.pos else 0
            self.pos.add(chosen_position)

            chosen_vector = action.memory_vectors[:, chosen_position].clone()
            reward -= torch.sum((chosen_vector - 1).pow(2)) * 10
            reward -= torch.var(chosen_vector) * 10

        return reward

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        """
        Performs a step in the environment based on the given action.
        :param action: the action taken by the agent.
        :return: the new state, the reward for the action, and a boolean indicating if the episode is finished.
        """
        mask = F.one_hot(action.positions,
                         num_classes=self.num_vectors) if self.memory_type == "conservative" else action.positions

        mask = mask.unsqueeze(-1).expand_as(self.state.memory)

        next_state = torch.where(mask == 1, action.memory_vectors.clone(), self.state.memory.clone())
        reward = self._get_reward(action)
        done = self.cur_step == self.max_steps or len(self.pos) == self.num_vectors

        self.cur_step += 1
        self.state = State(next_state, torch.zeros(1))

        return self.state, reward, done
