import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass

import config

from mbrlv1dot5.agent import MBRLv1dot5Agent
from environment import (
    make_vectorized_env,
    ReplayBuffer,
)
from utils import (
    create_history,
    add_to_history,
    save_checkpoint,
    pad_number_representation, 
)

@dataclass
class MBRLv1do5Config:
    HIDDEN_DIM                    = 256
    DYNAMICS_LR                   = 3e-5
    BATCH_SIZE                    = 128
    N_DYNAMICS_MODEL_UPDATES      = 10
    N_EXPLORATION_EPISODES        = 100
    N_EPISODES                    = 1000
    DYNAMICS_UPDATE_FREQUENCY     = 25
    REPLAY_BUFFER_SIZE            = 1000000
    N_CANDIDATES_ACTIONS          = 400
    MIN_PLANNING_LENGTH           = 1
    MAX_PLANNING_LENGTH           = 10
    EPISODE_START_PLANNING_GROWTH = 100
    EPISODE_END_PLANNING_GROWTH   = 700
    REWARD_LOSS_WEIGHT            = 0.5
    WEIGHT_DECAY                  = 0


# ----------------
# Training Process
# ----------------

def calculate_planning_length(episode: int) -> int:
    planning_length_interval = MBRLv1do5Config.MAX_PLANNING_LENGTH - MBRLv1do5Config.MIN_PLANNING_LENGTH
    episodes_interval = MBRLv1do5Config.EPISODE_END_PLANNING_GROWTH - MBRLv1do5Config.EPISODE_START_PLANNING_GROWTH
    current_episode_offset = episode - MBRLv1do5Config.EPISODE_START_PLANNING_GROWTH
    ratio = current_episode_offset/episodes_interval
    return min(max(int(MBRLv1do5Config.MIN_PLANNING_LENGTH + ratio*planning_length_interval), MBRLv1do5Config.MIN_PLANNING_LENGTH), MBRLv1do5Config.MAX_PLANNING_LENGTH)


def train(agent: MBRLv1dot5Agent, checkpoint_folder: str, history_folder: str):
    if config.VERBOSE:
        print(f'[DEVICE] Training on \'{str(agent.device).upper()}\' device.')

    history = create_history(
        attributes=[
            ('episode_reward_vs_num_episodes', 'Episode Reward', 'Number of Episodes', True),
            ('episode_length_vs_num_episodes', 'Episode Length', 'Number of Episodes', True),
            ('state_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model State Loss', 'Number of Update Steps', False),
            ('reward_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model Reward Loss', 'Number of Update Steps', False),
            ('total_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model Total Loss', 'Number of Update Steps', False),
        ],
    )
    
    envs = make_vectorized_env(
        env_name=config.ENV_NAME, 
        n_envs=config.N_ENVS, 
        healthy_reward=config.HEALTHY_REWARD,
        forward_reward_weight=config.FORWARD_REWARD_WEIGHT, 
        ctrl_cost_weight=config.CTRL_COST_WEIGHT,
    )

    device = agent.device
    replay_buffer = ReplayBuffer(MBRLv1do5Config.REPLAY_BUFFER_SIZE)
    optimizer_dynamics = optim.Adam(
        agent.env_dynamics_model.parameters(), 
        lr=MBRLv1do5Config.DYNAMICS_LR, 
        weight_decay=MBRLv1do5Config.WEIGHT_DECAY,
    )

    pbar = tqdm(total=MBRLv1do5Config.N_EXPLORATION_EPISODES)
    exploration_episodes_completed = 0
    states, _ = envs.reset()
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)

    print('Exploration Episodes...')
    while exploration_episodes_completed < MBRLv1do5Config.N_EXPLORATION_EPISODES:
        actions = envs.action_space.sample()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # add environment transition of not done environments to replay buffer
        replay_buffer.add(
            states=torch.FloatTensor(states[~dones, :]),
            actions=torch.FloatTensor(actions[~dones]),
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
            dones=torch.ByteTensor(dones[~dones]),
        )

        states = next_states
        exploration_episodes_completed += sum(dones)
        pbar.update(sum(dones))
    pbar.close()

    pbar = tqdm(total=MBRLv1do5Config.N_EPISODES)
    dyn_training_step_idx, checkpoint_idx = -1, -1
    total_steps, episodes_completed = 0, 0
    states, _ = envs.reset()
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)
    episodes_acc_reward = np.zeros(shape=config.N_ENVS, dtype=float)
    episodes_length = np.zeros(shape=config.N_ENVS, dtype=float)

    print('\nTraining Episodes...')
    while episodes_completed < MBRLv1do5Config.N_EPISODES:
        episodes_length += 1
        states_pt = torch.FloatTensor(states)

        agent.update_planning_length(calculate_planning_length(episodes_completed))
        actions = agent.choose_action(states_pt.to(device)).cpu()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
        dones = np.logical_or(terminateds, truncateds)

        # add environment transition of not done environments to replay buffer
        replay_buffer.add(
            states=states_pt[~dones, :],
            actions=actions[~dones],
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
            dones=torch.ByteTensor(dones[~dones]),
        )

        states = next_states
        episodes_acc_reward += rewards

        # process finished episodes
        if sum(dones) > 0:
            episodes_completed += sum(dones)
            if config.VERBOSE:
                print(f"[EPISODE {pad_number_representation(episodes_completed, MBRLv1do5Config.N_EPISODES)}/{MBRLv1do5Config.N_EPISODES}] \t Reward: {episodes_acc_reward[dones].mean():.2f}")
            add_to_history(history, 'episode_length_vs_num_episodes', *episodes_length[dones].tolist())
            add_to_history(history, 'episode_reward_vs_num_episodes', *episodes_acc_reward[dones].tolist())
            episodes_length[dones] = 0
            episodes_acc_reward[dones] = 0

        # train the dynamics model periodically
        if total_steps//MBRLv1do5Config.DYNAMICS_UPDATE_FREQUENCY > dyn_training_step_idx and len(replay_buffer) >= MBRLv1do5Config.BATCH_SIZE:
            dyn_training_step_idx = total_steps//MBRLv1do5Config.DYNAMICS_UPDATE_FREQUENCY
            for _ in range(MBRLv1do5Config.N_DYNAMICS_MODEL_UPDATES):
                sampled_states, sampled_actions, sampled_next_states, sampled_rewards, _ = replay_buffer.sample(MBRLv1do5Config.BATCH_SIZE)
                sampled_states = sampled_states.to(device)
                sampled_actions = sampled_actions.to(device)
                sampled_next_states = sampled_next_states.to(device)
                sampled_rewards = sampled_rewards.to(device)

                pred_next_states, pred_rewards = agent.predict_transition(sampled_states, sampled_actions)
                state_loss = F.mse_loss(pred_next_states, sampled_next_states)
                reward_loss = F.mse_loss(pred_rewards.squeeze(-1), sampled_rewards)
                dyn_loss = state_loss + MBRLv1do5Config.REWARD_LOSS_WEIGHT*reward_loss

                optimizer_dynamics.zero_grad()
                dyn_loss.backward()
                optimizer_dynamics.step()

                add_to_history(history, 'state_dynamics_loss_vs_num_dynupdate_steps', state_loss.item())
                add_to_history(history, 'reward_dynamics_loss_vs_num_dynupdate_steps', reward_loss.item())
                add_to_history(history, 'total_dynamics_loss_vs_num_dynupdate_steps', dyn_loss.item())

        # save agent and history checkpoint
        if episodes_completed//config.CHECKPOINT_FREQUENCY > checkpoint_idx:
            checkpoint_idx = episodes_completed//config.CHECKPOINT_FREQUENCY
            save_checkpoint(agent, history, checkpoint_idx, checkpoint_folder, history_folder)

        pbar.update(sum(dones))

        total_steps += config.N_ENVS
    pbar.close()

    envs.close()

    save_checkpoint(agent, history, checkpoint_idx+1, checkpoint_folder, history_folder)

    return agent, history