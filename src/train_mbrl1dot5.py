import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

import config

from mbrl1dot5_agent import MBRLv1dot5Agent
from environment import (
    make_env,
    make_vectorized_env,
    ReplayBuffer,
)
from utils import (
    create_history,
    add_to_history,
    save_checkpoint,
    pad_number_representation, 
    make_history_plots,
    get_arguments,
    get_device,
)


HIDDEN_DIM                = 64
POLICY_LR                 = 3e-4
DYNAMICS_LR               = 1e-4
BATCH_SIZE                = 128
EPOCH_BATCH_AMOUNT        = 10
N_EXPLORATION_EPISODES    = 250
N_EPISODES                = 1000
DYNAMICS_TRAIN_FREQUENCY  = 500
REPLAY_BUFFER_SIZE        = 100000
N_CANDIDATES_ACTIONS      = 200
PLANNING_LENGTH           = 5
REWARD_LOSS_WEIGHT        = 0.1
GAMMA                     = 0.99


# ----------------
# Training Process
# ----------------

def train(agent: MBRLv1dot5Agent, checkpoint_folder: str, history_folder: str):
    if config.VERBOSE:
        print(f'[DEVICE] Training on \'{str(agent.device).upper()}\' device.')

    history = create_history(
        attributes=[
            ('episode_reward_vs_num_episodes', 'Episode Reward', 'Number of Episodes'),
            ('state_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model State Loss', 'Number of Update Steps'),
            ('reward_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model Reward Loss', 'Number of Update Steps'),
            ('total_dynamics_loss_vs_num_dynupdate_steps', 'Agent\'s Dynamics Model Total Loss', 'Number of Update Steps'),
            ('policy_action_distribution_entropy_vs_num_episodes', 'Entropy of Policy Action Distribution', 'Number of Episodes'),
            ('policy_model_loss_vs_num_polupdate_steps', 'Policy Model Loss', 'Number of Update Steps'),
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
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    optimizer_policy = optim.Adam(agent.policy_model.parameters(), lr=POLICY_LR)
    optimizer_dynamics = optim.Adam(agent.env_dynamics_model.parameters(), lr=DYNAMICS_LR)

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
        replay_buffer.add(
            states=torch.FloatTensor(states[~dones, :]),
            actions=torch.FloatTensor(actions[~dones]),
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
        )

        states = next_states
        exploration_episodes_completed += sum(dones)
        if not config.VERBOSE:
            pbar.update(sum(dones))
    pbar.close()

    print('\nTraining Episodes...')
    if not config.VERBOSE:
        pbar = tqdm(total=N_EPISODES)
    states, _ = envs.reset()
    episodes_acc_reward = np.zeros(shape=config.N_ENVS, dtype=float)
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)
    trajectory_buffer = [ [] for _ in range(config.N_ENVS) ]
    total_steps, episodes_completed = 0, 0
    dyn_training_step_idx, checkpoint_idx = -1, -1
    while episodes_completed < N_EPISODES:
        states_pt = torch.FloatTensor(states)

        actions = agent.choose_action_learned_policy(states_pt).cpu()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
        dones = np.logical_or(terminateds, truncateds)

        next_states_pt = torch.FloatTensor(next_states)
        rewards_pt = torch.FloatTensor(rewards)

        # add environment transition of not done environments to replay buffer
        replay_buffer.add(
            states=states_pt[~dones, :],
            actions=actions[~dones],
            next_states=next_states_pt[~dones],
            rewards=rewards_pt[~dones],
        )

        # add entries to trajectories buffer
        for i in range(config.N_ENVS):
            trajectory_buffer[i].append((states_pt[i], actions[i], rewards_pt[i]))

        states = next_states
        episodes_acc_reward += rewards

        # process finished episodes and train policy with REINFORCE
        if sum(dones) > 0:
            add_to_history(history, 'episode_reward_vs_num_episodes', *episodes_acc_reward[dones].tolist())
            episodes_completed += sum(dones)
            episodes_acc_reward[dones] = 0

            policy_loss = 0
            for i in range(config.N_ENVS):
                if dones[i]:
                    t_states, t_actions, t_rewards = zip(*trajectory_buffer[i])
                    t_states = torch.stack(t_states, dim=0).to(device)
                    t_actions = torch.stack(t_actions, dim=0).to(device)
                    t_rewards = torch.stack(t_rewards, dim=0).to(device)

                    log_probs, entropy = agent.policy_model.get_action_logprobs_entropy(t_states, t_actions)

                    R = 0
                    returns = torch.zeros_like(t_rewards).to(device)
                    for i in range(len(returns)-1, -1, -1):
                        R = t_rewards[i] + GAMMA*R
                        returns[i] = R
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normalize returns
                    
                    # calculate policy loss
                    policy_loss += (-log_probs * returns.detach()).mean()

                    trajectory_buffer[i] = []
                
                    # add action distribution entropy to history
                    add_to_history(history, 'policy_action_distribution_entropy_vs_num_episodes', entropy.mean().item())

            # backpropagate once for all trajectories
            if policy_loss != 0:
                optimizer_policy.zero_grad()
                policy_loss.backward() # type: ignore
                optimizer_policy.step()

                # add policy loss to history
                add_to_history(history, 'policy_model_loss_vs_num_polupdate_steps', policy_loss.item())

        # train the dynamics model periodically
        if total_steps//DYNAMICS_TRAIN_FREQUENCY > dyn_training_step_idx and len(replay_buffer) >= BATCH_SIZE:
            dyn_training_step_idx = total_steps//DYNAMICS_TRAIN_FREQUENCY
            for _ in range(EPOCH_BATCH_AMOUNT):
                sampled_states, sampled_actions, sampled_next_states, sampled_rewards = replay_buffer.sample(BATCH_SIZE)
                sampled_states = sampled_states.to(device)
                sampled_actions = sampled_actions.to(device)
                sampled_next_states = sampled_next_states.to(device)
                sampled_rewards = sampled_rewards.to(device)

                pred_next_states, pred_rewards = agent.predict_transition(sampled_states, sampled_actions)
                state_loss = F.mse_loss(pred_next_states, sampled_next_states)
                reward_loss = F.mse_loss(pred_rewards.squeeze(-1), sampled_rewards)
                dyn_loss = state_loss + REWARD_LOSS_WEIGHT*reward_loss

                optimizer_dynamics.zero_grad()
                dyn_loss.backward()
                optimizer_dynamics.step()

                add_to_history(history, 'state_dynamics_loss_vs_num_dynupdate_steps', state_loss.item())
                add_to_history(history, 'reward_dynamics_loss_vs_num_dynupdate_steps', reward_loss.item())
                add_to_history(history, 'total_dynamics_loss_vs_num_dynupdate_steps', dyn_loss.item())

        # save agent and history checkpoint
        if episodes_completed//config.CHECKPOINT_EPISODE_FREQUENCY > checkpoint_idx:
            checkpoint_idx = episodes_completed//config.CHECKPOINT_EPISODE_FREQUENCY
            save_checkpoint(agent, history, checkpoint_idx, checkpoint_folder, history_folder)

        if config.VERBOSE:
            print(f"[EPISODE {pad_number_representation(episodes_completed, N_EPISODES)}/{N_EPISODES}] \
                  {'\t'.join(f'Reward {i}: {r:.2f}' for i, r in enumerate(episodes_acc_reward[dones], 1))}")
        else:
            pbar.update(sum(dones))

        total_steps += config.N_ENVS
    pbar.close()

    envs.close()

    save_checkpoint(agent, history, checkpoint_idx+1, checkpoint_folder, history_folder)

    return agent, history

# ------------------------------------------
# Play Recording Environment For One Episode
# ------------------------------------------
def play_recording_environment(env_name: str, agent: MBRLv1dot5Agent, video_folder: str, video_name_prefix: str):
    os.makedirs(video_folder, exist_ok=True)
    
    env = make_env(env_name, healthy_reward=0, forward_reward_weight=0, ctrl_cost_weight=0, render_mode='rgb_array', is_recording=True)
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name_prefix, episode_trigger=lambda _: True)
    
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action_learned_policy(torch.FloatTensor(state)).cpu()
        next_state, _, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        state = next_state
    
    env.close()

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