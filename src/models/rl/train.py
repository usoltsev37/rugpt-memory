import torch

import wandb
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.rl.agent import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.utils.train_config import RLParams


def compute_rewards(trajectory: [list], gamma: float):
    """
    Computes the discounted reward for each step in a given trajectory.

    :param trajectory: A sequence of lists where each list represents a step in the trajectory and is structured as
    (state, action, reward, log_proba, distr).
    :param gamma: The discount factor used to value future rewards. A value of 0 discounts future rewards completely,
    while a value close to 1 gives them nearly equal weight as immediate rewards.

    :return: The modified trajectory where each list is of the form (state, action, updated_reward, log_proba, distr),
    with updated_reward being the discounted reward calculated for each step.
    """
    rewards = []
    last_r = 0.
    for _, _, r, _, _ in reversed(trajectory):
        ret = r + gamma * last_r
        last_r = ret
        rewards.append(last_r)

    return [(state, action, reward, log_proba, distr) for (state, action, _, log_proba, distr), reward in
            zip(trajectory, reversed(rewards))]


def sample_episodes(env: LTMEnvironment,
                    agent: REINFORCE,
                    data: dict,
                    train_config: RLParams) -> [tuple]:
    """
    Samples an episode of interaction between an agent and an environment,
    then computes and returns the discounted rewards for each step in the episode.
    """
    state = env.reset(data)
    done = False
    trajectories = []
    with torch.no_grad():
        while not done:
            action, log_proba, distr = agent.act(state)
            next_state, reward, done = env.step(action)
            if trajectories:
                trajectories[-1][2] = reward  # Reward from the step (i+1) is a true reward for step (i)
            trajectories.append([state, action, reward, log_proba, distr])
            state = next_state

    return compute_rewards(trajectories[:-1], train_config.gamma)  # There is no reward for the last step


def train_rl(data: [dict],
             agent: Agent,
             optimizer: torch.optim,
             ltm_model: LTM_GPT,
             train_config: RLParams):
    """
    Training a memory model using reinforcement learning with a fixed LTM model.
    :param data: Training data from EpochDataloader
    :param agent: Agent with MemoryModel
    :param ltm_model: LTM model with frozen weights
    :param train_config: config with training parameters of the REINFORCE algorithm
    """
    env = LTMEnvironment(ltm_model, agent.num_vectors, agent.d_mem)
    reinforce = REINFORCE(agent, optimizer, train_config)

    transitions = []
    for batch in data:
        batch_traj = sample_episodes(env, reinforce, batch, train_config)
        transitions.extend(batch_traj)

    mean_loss = reinforce.update(transitions)
    wandb.log({"memory_model_loss": mean_loss})

    return mean_loss
