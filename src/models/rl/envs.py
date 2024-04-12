import torch
import torch.nn.functional as F

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.rl.utils import Action, State
from src.utils.train_config import SyntheticTaskEnvParams


class LTMEnvironment:
    """Memory model training environment. The reward and state come from the LTM model."""

    def __init__(self,
                 ltm_model: LTM_GPT,
                 memory_num_vectors: int,
                 d_mem: int,
                 dtype: torch.dtype = torch.float32) -> None:
        self.attention_mask = None
        self.data = None
        self.embeddings = None
        self.n_steps = None
        self.cur_step = None
        self.iterator = None
        self.cur_state = None

        self.ltm_model = ltm_model
        self.max_seq_length = ltm_model.max_seq_length
        self.memory_module = MemoryModule(num_vectors=memory_num_vectors, d_mem=d_mem, dtype=dtype)

    def reset(self, data: dict) -> State:
        """Returns a new state in the form of empty memory and high-level embeddings of text (max_seq_len)
        received from LTM model.
        :param data: tokenized article with contexts with size [1, num_tokens]
        :return: first observation - tuple of the memory and high-level embeddings
        """
        # Return new state
        bs, n_steps, _ = data['input_ids'].shape
        self.data = data
        self.cur_step = 0
        self.n_steps = n_steps

        input_ids, attention_mask = (data['input_ids'][:, self.cur_step, :].contiguous(),
                                     data['attention_mask'][:, self.cur_step, :].contiguous())

        # Get embeddings for the first state
        input_ids, attention_mask = input_ids.to(self.ltm_model.first_device), attention_mask.to(
            self.ltm_model.first_device)
        embeddings = self.ltm_model.get_embeddings(input_ids, attention_mask)

        self.embeddings = embeddings.cpu()
        self.attention_mask = attention_mask
        self.memory_module.reset(bs)

        return State(self.memory_module.memory, self.embeddings, attention_mask)

    def step(self, action: Action) -> tuple[State, torch.Tensor, bool]:
        self.cur_step += 1

        reward = self.ltm_model.get_output(self.embeddings.to(self.ltm_model.second_device),
                                           self.attention_mask.to(self.ltm_model.second_device),
                                           self.memory_module.memory.to(self.ltm_model.second_device),
                                           reward_for_agent=True)

        new_memory = self.memory_module.update(action)

        if self.cur_step >= self.n_steps:
            done = True
            self.embeddings = torch.zeros_like(self.embeddings)
            self.attention_mask = torch.zeros_like(self.attention_mask)
        else:
            done = False
            input_ids, attention_mask = (self.data['input_ids'][:, self.cur_step, :].contiguous(),
                                         self.data['attention_mask'][:, self.cur_step, :].contiguous())

            embeddings = self.ltm_model.get_embeddings(input_ids.to(self.ltm_model.first_device),
                                                       attention_mask.to(self.ltm_model.first_device))

            self.embeddings = embeddings.cpu()
            self.attention_mask = attention_mask

        return State(new_memory, self.embeddings, self.attention_mask), reward, done


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
