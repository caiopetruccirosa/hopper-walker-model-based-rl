import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from tqdm import tqdm

import config

from mbrl1dot5_agent import MBRLv1dot5Agent
from environment import (
    make_vectorized_env,
    play_recording_environment,
    ReplayBuffer,
)
from utils import (
    create_history,
    add_to_history,
    save_checkpoint,
    get_device, 
    pad_number_representation, 
    make_history_plots,
    get_arguments,
)


HIDDEN_DIM                = 128
LR                        = 2.5e-4
BATCH_SIZE                = 128
EPOCH_BATCH_AMOUNT        = 10
N_EXPLORATION_EPISODES    = 500
N_EPISODES                = 2000
TRAIN_FREQUENCY           = 100
REPLAY_BUFFER_SIZE        = 100000
N_CANDIDATES_ACTIONS      = 200
PLANNING_LENGTH           = 5
REWARD_LOSS_WEIGHT        = 0.5


# ----------------
# Training Process
# ----------------

def train(agent: MBRLv1dot5Agent, checkpoint_folder: str, history_folder: str):
    if config.VERBOSE:
        print(f'[DEVICE] Training on \'{str(agent.device).upper()}\' device.')

    history = create_history(
        attributes=[
            ('episode_reward_vs_num_episodes', 'Episode Reward', 'Number of Episodes'),
            ('episode_length_vs_num_episodes', 'Episode Length', 'Number of Episodes'),
            ('state_dynamics_loss_vs_num_update_steps', 'Agent\'s Dynamics Model State Loss', 'Number of Update Steps'),
            ('reward_dynamics_loss_vs_num_update_steps', 'Agent\'s Dynamics Model Reward Loss', 'Number of Update Steps'),
            ('total_dynamics_loss_vs_num_update_steps', 'Agent\'s Dynamics Model Total Loss', 'Number of Update Steps'),
            ('lr_vs_num_episodes', 'Learning Rate', 'Number of Episodes'),
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
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(agent.env_dynamics_model.parameters(), lr=LR)

    print('Exploration Episodes...')
    if not config.VERBOSE:
        pbar = tqdm(total=N_EXPLORATION_EPISODES)
    exploration_episodes_completed = 0
    states, _ = envs.reset()
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)
    while exploration_episodes_completed < N_EXPLORATION_EPISODES:
        actions = envs.action_space.sample()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # add environment transition of not done environments to replay buffer
        buffer.add(
            states=torch.FloatTensor(states[~dones, :]), 
            actions=torch.FloatTensor(actions[~dones]), 
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
        )

        states = next_states
        exploration_episodes_completed += sum(dones)
        if not config.VERBOSE:
            pbar.update(sum(dones))

    print('\nTraining Episodes...')
    if not config.VERBOSE:
        pbar = tqdm(total=N_EPISODES)
    states, _ = envs.reset()
    episodes_reward = []
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)
    total_steps, episodes_completed = 0, 0
    training_step_idx, checkpoint_idx = -1, -1
    while episodes_completed < N_EPISODES:
        actions = agent.choose_action(torch.FloatTensor(states))
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # add environment transition of not done environments to replay buffer
        buffer.add(
            states=torch.FloatTensor(states[~dones, :]), 
            actions=torch.FloatTensor(actions[~dones]), 
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
        )

        states = next_states
        
        # process finished episodes
        if 'episode' in infos.keys():
            current_episodes_completed = len(infos['episode']['r'])
            episodes_completed += current_episodes_completed
            episodes_reward = infos['episode']['r'].tolist()

            # train policy model with REINFORCE
            # TODO

            add_to_history(history, 'episode_reward_vs_num_episodes', *(episodes_reward))
            add_to_history(history, 'episode_length_vs_num_episodes', *(infos['episode']['l'].tolist()))

        # train the dynamics model periodically
        if total_steps//TRAIN_FREQUENCY > training_step_idx and len(buffer) >= BATCH_SIZE:
            training_step_idx = total_steps//TRAIN_FREQUENCY
            for _ in range(EPOCH_BATCH_AMOUNT):
                sampled_states, sampled_actions, sampled_next_states, sampled_rewards = buffer.sample(BATCH_SIZE)
                sampled_states = sampled_states.to(device)
                sampled_actions = sampled_actions.to(device)
                sampled_next_states = sampled_next_states.to(device)
                sampled_rewards = sampled_rewards.to(device)

                pred_next_states, pred_rewards = agent.predict_transition(sampled_states, sampled_actions)
                state_loss = F.mse_loss(pred_next_states, sampled_next_states)
                reward_loss = F.mse_loss(pred_rewards.squeeze(-1), sampled_rewards)
                loss = state_loss + REWARD_LOSS_WEIGHT*reward_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                add_to_history(history, 'state_dynamics_loss_vs_num_update_steps', loss.item())
                add_to_history(history, 'reward_dynamics_loss_vs_num_update_steps', loss.item())
                add_to_history(history, 'total_dynamics_loss_vs_num_update_steps', loss.item())

        total_steps += config.N_ENVS
        
        # save agent and history checkpoint
        if episodes_completed//config.CHECKPOINT_EPISODE_FREQUENCY > checkpoint_idx:
            checkpoint_idx = episodes_completed//config.CHECKPOINT_EPISODE_FREQUENCY
            save_checkpoint(agent, history, checkpoint_idx, checkpoint_folder, history_folder)

        if config.VERBOSE:
            print(f"[EPISODE {pad_number_representation(episodes_completed, N_EPISODES)}/{N_EPISODES}] \
                  {'\t'.join(f'Reward {i}: {r:.2f}' for i, r in enumerate(episodes_reward, 1))}")
        else:
            pbar.update(sum(dones))

    envs.close()

    save_checkpoint(agent, history, checkpoint_idx+1, checkpoint_folder, history_folder)

    return agent, history


# ----
# Main
# ----

def main():
    # set random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_arguments()

    experiment_folder = f'experiments/experiment_{args.experiment_id}'
    os.makedirs(experiment_folder, exist_ok=True)

    device = get_device(args.device)
    agent = MBRLv1dot5Agent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        planning_length=PLANNING_LENGTH,
        n_candidate_actions=N_CANDIDATES_ACTIONS,
        device=device,
        verbose=config.VERBOSE,
    )

    agent, history = train(
        agent=agent, 
        checkpoint_folder=f'{experiment_folder}/{config.CHECKPOINT_FOLDER}',
        history_folder=f'{experiment_folder}/{config.HISTORY_FOLDER}',
    )

    make_history_plots(
        history=history,
        experiment_folder=experiment_folder,
        fig_folder=config.FIG_PATH,
    )

    play_recording_environment(
        env_name=config.ENV_NAME,
        agent=agent,
        video_folder=f'{experiment_folder}/{config.VIDEO_FOLDER}',
        video_name_prefix=config.VIDEO_NAME_PREFIX,
    )

if __name__ == "__main__":
    main()