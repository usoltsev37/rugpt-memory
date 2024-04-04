import random

import numpy as np
from config import *
from model.memory_model import SyntheticTaskModel
from tqdm import tqdm

from src.models.rl.envs import SyntheticTaskEnv
from src.models.rl.reinforce_for_synthetic_task import REINFORCE


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_rewards(trajectory: [tuple],
                    gamma: float):
    rewards = []
    last_r = 0.
    for _, _, r, _, _ in reversed(trajectory):
        ret = r + gamma * last_r
        last_r = ret
        rewards.append(last_r)

    return [(state, action, reward, proba, distr) for (state, action, _, proba, distr), reward in
            zip(trajectory, reversed(rewards))]


def sample_episode(env: SyntheticTaskEnv,
                   agent: REINFORCE,
                   train_config: RLArgs) -> [tuple]:
    state = env.reset()
    done = False
    trajectory = []
    while not done:
        action, proba, distr = agent.act(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward, proba, distr))
        state = next_state

    return compute_rewards(trajectory, train_config.gamma)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        # print(state.memory)
        total_reward = 0.
        while not done:
            state, reward, done = env.step(agent.act(state)[0])
            # print(state.memory)
            total_reward += reward
        # print("--------------------------------------------------------------------------------------------------------")
        returns.append(total_reward)

    return returns


def train(train_config: RLArgs, env_params: SyntheticTaskEnvParams):
    set_seed(42)
    env = SyntheticTaskEnv(env_params)
    model = SyntheticTaskModel(env_params.num_vectors, env_params.d_mem, env_params.memory_type)
    reinforce = REINFORCE(model, train_config)

    for i in tqdm(range(train_config.iterations)):
        trajectories = []
        steps_cnt = 0

        while steps_cnt < train_config.min_transitions_per_update:
            traj = sample_episode(env, reinforce, train_config)
            steps_cnt += len(traj)
            trajectories.append(traj)

        mean_loss = reinforce.update(trajectories)

        if (i + 1) % (train_config.iterations // 100) == 0:
            rewards = evaluate_policy(env, reinforce, 5)
            print(
                f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)},  Loss: {mean_loss}")
            reinforce.save()


train(RLArgs(), SyntheticTaskEnvParams())
