import torch

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.rl.reinforce import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.utils.train_config import RLParams, TrainingArguments


def compute_rewards(trajectory: list[list], gamma: float):
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
    last_r = 0.0
    for _, _, r, _, _ in reversed(trajectory):
        ret = r + gamma * last_r
        last_r = ret
        rewards.append(last_r)

    return [
        (state, action, reward, log_proba, distr)
        for (state, action, _, log_proba, distr), reward in zip(trajectory, reversed(rewards))
    ]


def sample_episodes(env: LTMEnvironment, reinforce: REINFORCE, data: dict, train_config: RLParams) -> [tuple]:
    """
    Samples an episode of interaction between an reinforce and an environment,
    then computes and returns the discounted rewards for each step in the episode.
    """
    state = env.reset(data)
    done = False
    trajectories = []

    with torch.no_grad():
        while not done:
            action, log_proba, distr = reinforce.act(state)  # cpu
            next_state, reward, done = env.step(action)  # cpu
            if trajectories:
                trajectories[-1][2] = reward  # Reward from the step (i+1) is a true reward for step (i)
            trajectories.append([state, action, reward, log_proba, distr])
            state = next_state

    return compute_rewards(trajectories[:-1], train_config.gamma)  # There is no reward for the last step


def train_rl(
    data: list[dict],
    agent: Agent,
    optimizer: torch.optim,
    ltm_model: LTM_GPT,
    memory_module,
    train_config: TrainingArguments,
    logger,
    pretrain_mode: bool = False
):
    """
    Training a memory model using reinforcement learning with a fixed LTM model.
    :param data: Training data from EpochDataloader
    :param reinforce: Agent with MemoryModel
    :param ltm_model: LTM model with frozen weights
    :param train_config: config with training parameters of the REINFORCE algorithm
    """
    if data[0]["input_ids"].shape[0] != data[-1]["input_ids"].shape[0]:
        return None

    agent.model.eval()
    env = LTMEnvironment(
        ltm_model,
        memory_module,
        num_prefixes_for_reward_calc=train_config.rl_params.num_prefixes_for_reward_calc,
        pretrain_mode=pretrain_mode
    )
    reinforce = REINFORCE(agent, optimizer, train_config=train_config.rl_params)

    transitions = []

    for batch in data:
        batch_traj = sample_episodes(env, reinforce, batch, train_config.rl_params)
        transitions.extend(batch_traj)

    agent.model.train()
    mean_loss = reinforce.update(transitions, logger)
    # tensorboard_writer.add_scalar("Loss/memory_model_train_iteration_loss", mean_loss)

    return mean_loss
