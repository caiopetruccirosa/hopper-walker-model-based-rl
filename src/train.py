import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os

import math

from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

import config

from agent import CheetahAgent
from utils import (
    make_env, 
    make_vectorized_env,
    preprocess_env_step,
    preprocess_vectorized_env_step,
    create_history,
    add_to_history,
    save_checkpoint,
    get_device, 
    pad_number_representation, 
    make_plot, 
    to_float_tensor,
)


# -------------------------------
# Record Episode Of Agent Running
# -------------------------------

def record_episode(env_name: str, agent: CheetahAgent, video_folder: str, video_name_prefix: str):
    os.makedirs(video_folder, exist_ok=True)
    
    env = make_env(env_name, render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name_prefix, episode_trigger=lambda _: True)
    
    state, _ = env.reset()
    done = False
    while not done:
        action, _, = agent.choose_action(*to_float_tensor(state))
        action = action.detach().cpu().numpy()
        next_state, _, done, _ = preprocess_env_step(*env.step(action))
        state = next_state
    
    env.close()


# ----------------
# Training process
# ----------------

def preprocess_trajectories_data(
    data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    n_steps: int, 
    n_workers: int, 
    state_dim: int,
    action_dim: int,
    check_tensor_shapes: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states, actions, rewards, dones, values, actions_logprobs = zip(*data)
    
    # batch of tensors of states at step t; 
    #   shape (n_policy_rollout_steps, n_envs, state_dim)
    states = torch.stack(states, dim=0)
    
    # batch of tensors of actions taken at step t; 
    #   shape (n_policy_rollout_steps, n_envs, action_dim)
    actions = torch.stack(actions, dim=0)

    # batch of tensors of rewards received step t; 
    #   shape (n_policy_rollout_steps, n_envs)
    rewards = torch.stack(rewards, dim=0)
    
    # batch of tensors of representing whether the episode ended at step t;
    #   shape (n_policy_rollout_steps, n_envs)
    dones = torch.stack(dones, dim=0)

    # batch of tensors of state values predicted by value network for state at step t;
    #   shape (n_policy_rollout_steps, n_envs)
    values = torch.stack(values, dim=0).squeeze(dim=-1)

    # batch of tensors of log probability actions taken at step t, considering all components of an action as independent;
    #   shape (n_policy_rollout_steps, n_envs)
    actions_logprobs = torch.stack(actions_logprobs, dim=0)

    # assert tensor shapes to expected shapes
    if check_tensor_shapes:
        assert states.shape == (n_steps, n_workers, state_dim)
        assert actions.shape == (n_steps, n_workers, action_dim)
        assert rewards.shape == (n_steps, n_workers)
        assert dones.shape == (n_steps, n_workers)
        assert values.shape == (n_steps, n_workers)
        assert actions_logprobs.shape == (n_steps, n_workers)

    return states, actions, rewards, dones, values, actions_logprobs


def calculate_generalized_advantage_estimation(
    rewards: torch.Tensor, 
    dones: torch.Tensor, 
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    n_steps: int, 
    n_workers: int, 
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        rewards = rewards.to(device)
        dones = dones.to(device)
        values = values.to(device)

        # batch of tensors of representing the advantange of the action taken at step t, considering current state;
        #   shape (n_policy_rollout_steps, n_envs)
        advantages = torch.zeros((n_steps, n_workers), dtype=torch.float32, device=device)

        for t in reversed(range(n_steps-1)):
            # mask if episode completed after step t
            values[t+1] = values[t+1] * (1-dones[t])
            advantages[t+1] = advantages[t+1] * (1-dones[t])

            delta = rewards[t] + gamma*values[t+1] - values[t]
            advantages[t] = delta + gamma*gae_lambda *advantages[t+1]

        returns = advantages + values

        return advantages, returns


def train_agent(
    agent: CheetahAgent,
    env_name: str,
    n_envs: int,
    n_training_steps: int,
    n_policy_rollout_steps: int,
    n_minibatches: int,
    policy_update_epochs: int,
    lr: float,
    lr_end: float,
    gamma: float,
    clip_epsilon: float,
    gae_lambda: float,
    max_gradient_norm: float,
    value_loss_coefficient: float,
    n_checkpoints: int,
    checkpoint_folder: str,
    history_attributes: list[tuple[str, str, str]],
    history_folder: str,
    check_tensor_shapes: bool=False,
    verbose: bool=False,
) -> tuple[CheetahAgent, dict[str, list]]:
    if verbose:
        print(f'[DEVICE] Training on \'{str(agent.device).upper()}\' device.')

    batch_size = n_envs*n_policy_rollout_steps
    minibatch_size = math.ceil(batch_size/n_minibatches)
    n_training_update_steps = math.ceil(n_training_steps/batch_size)

    checkpoint_step_frequency = math.ceil(n_training_update_steps/n_checkpoints)

    history = create_history(history_attributes)
    
    envs = make_vectorized_env(env_name, n_envs)

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lr_end/lr, total_iters=n_training_update_steps)

    episodes_completed = 0

    states, _ = envs.reset()
    dones = np.zeros(shape=n_envs, dtype=bool)
    for update_step in tqdm(range(1, n_training_update_steps+1)):
        trajectories_data = []

        # ---
        # policy rollout that populates a list of trajectories batch
        # ---
        for _ in range(n_policy_rollout_steps):
            with torch.no_grad():
                actions, actions_logprobs = agent.choose_action(*to_float_tensor(states))
                values = agent.evaluate_state_value(*to_float_tensor(states))
                actions_dist_entropy = agent.get_action_distribution_entropy(*to_float_tensor(states))

            # dones[i] is True => next_states[i] is the initial of the next episode AND rewards[i] is the reward for the last step of the finished episode
            next_states, rewards, dones, infos = preprocess_vectorized_env_step(*envs.step(actions.detach().cpu().numpy()))

            # ---
            # track training with history dictionary
            # ---
            add_to_history(history, 'state_value_estimation_vs_num_env_steps', *(values.tolist()))
            add_to_history(history, 'policy_action_distribution_entropy_vs_num_env_steps', *(actions_dist_entropy.tolist()))
            if 'episode' in infos.keys():
                episodes_completed += len(infos['episode']['r'])
                add_to_history(history, 'episode_reward_vs_num_episodes', *(infos['episode']['r'].tolist()))
                add_to_history(history, 'episode_length_vs_num_episodes', *(infos['episode']['l'].tolist()))

            trajectories_data.append((*to_float_tensor(states), actions, *to_float_tensor(rewards, dones), values, actions_logprobs))
            states = next_states

        # ---
        # policy optimization
        # ---

        # preprocess trajectory data generated during rollout
        states_pt, actions_pt, rewards_pt, dones_pt, values_pt, actions_logprobs_pt = preprocess_trajectories_data(
            data=trajectories_data, 
            n_steps=n_policy_rollout_steps, 
            n_workers=n_envs, 
            state_dim=agent.state_dim, 
            action_dim=agent.action_dim,
            check_tensor_shapes=check_tensor_shapes,
        )

        # calculate advantages and returns using the Generalized Advantage Estimation method
        advantages_pt, returns_pt = calculate_generalized_advantage_estimation(
            rewards=rewards_pt,
            dones=dones_pt,
            values=values_pt,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_steps=n_policy_rollout_steps,
            n_workers=n_envs,
            device=agent.device,
        )

        # flattens necessary tensors to merge "policy step" and "env id" dimensions
        states_pt = states_pt.view(-1, agent.state_dim)
        actions_pt = actions_pt.view(-1, agent.action_dim)
        actions_logprobs_pt = actions_logprobs_pt.view(-1)
        advantages_pt = advantages_pt.view(-1)
        returns_pt = returns_pt.view(-1)
        values_pt = values_pt.view(-1)

        acc_step_policy_loss, acc_step_value_loss = 0, 0
        
        for _ in range(policy_update_epochs):
            # shuffle indices in random order and then select permutation
            permutation_indices = torch.randperm(batch_size)
            states_pt = states_pt[permutation_indices]
            actions_pt = actions_pt[permutation_indices]
            actions_logprobs_pt = actions_logprobs_pt[permutation_indices]
            advantages_pt = advantages_pt[permutation_indices]
            returns_pt = returns_pt[permutation_indices]
            values_pt = values_pt[permutation_indices]

            acc_epoch_policy_loss, acc_epoch_value_loss = 0, 0

            for i in range(n_minibatches):
                mb_start, mb_end = i*minibatch_size, min((i+1)*minibatch_size, batch_size)

                mb_states_pt = states_pt[mb_start:mb_end]
                mb_actions_pt = actions_pt[mb_start:mb_end]
                mb_old_actions_logprobs_pt = actions_logprobs_pt[mb_start:mb_end]
                mb_advantages_pt = advantages_pt[mb_start:mb_end]
                mb_returns_pt = returns_pt[mb_start:mb_end]

                mb_new_actions_logprobs_pt = agent.get_logprobability_for_chosen_action(mb_states_pt, mb_actions_pt)
                mb_new_values_pt = agent.evaluate_state_value(mb_states_pt)

                # calculate policy network loss
                logratio = mb_new_actions_logprobs_pt - mb_old_actions_logprobs_pt
                ratio = logratio.exp()

                policy_loss_non_clipped = ratio*mb_advantages_pt
                policy_loss_clipped = torch.clip(ratio, min=1-clip_epsilon, max=1+clip_epsilon)*mb_advantages_pt
                
                policy_loss = -1 * torch.minimum(policy_loss_non_clipped, policy_loss_clipped).mean()

                # calculate value network loss
                value_loss = F.mse_loss(mb_new_values_pt, mb_returns_pt)

                # calculate agent loss
                loss = policy_loss + value_loss_coefficient*value_loss
                
                # optimize parameters through backpropagation; the gradients are clipped in order to have a max norm                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_gradient_norm, norm_type=2)
                optimizer.step()

                acc_epoch_policy_loss += policy_loss.item()
                acc_epoch_value_loss += value_loss.item()

            acc_step_policy_loss += acc_epoch_policy_loss
            acc_step_value_loss += acc_epoch_value_loss

            add_to_history(history, 'policy_network_loss_vs_num_update_epochs', (acc_epoch_policy_loss/n_minibatches))
            add_to_history(history, 'value_network_loss_vs_num_update_epochs', (acc_epoch_value_loss/n_minibatches))

        add_to_history(history, 'policy_network_loss_vs_num_update_steps', (acc_epoch_policy_loss/(policy_update_epochs*n_minibatches)))
        add_to_history(history, 'value_network_loss_vs_num_update_steps', (acc_epoch_value_loss/(policy_update_epochs*n_minibatches)))

        scheduler.step()

        # save agent and history checkpoint
        if update_step % checkpoint_step_frequency == 0:
            save_checkpoint(agent, history, update_step//checkpoint_step_frequency, checkpoint_folder, history_folder)

        if verbose and episodes_completed > 0:
            print(f"[TRAINING UPDATE {pad_number_representation(update_step, n_training_update_steps)}/{n_training_update_steps}] \t \
                  Last Episode Reward: \t {history['episode_reward_vs_num_episodes'][episodes_completed-1]:.2f}")

    envs.close()

    save_checkpoint(agent, history, n_checkpoints, checkpoint_folder, history_folder)

    return agent, history


# ----
# Main
# ----

def main():
    # set random seed
    random.seed(42)
    torch.manual_seed(42)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--experiment_id', '-e', help='Experiment ID. Used to store artifacts from experiment in a specific folder.', type=int)
    arg_parser.add_argument('--device', '-d', help='PyTorch device to use for tensor operations during training.', type=str)
    args = arg_parser.parse_args()

    experiment_folder = f'experiments/experiment_{args.experiment_id}'
    os.makedirs(experiment_folder, exist_ok=True)

    device = get_device(args.device)
    agent = CheetahAgent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        device=device,
        verbose=config.VERBOSE,
    )

    history_attributes = [
        ('episode_reward_vs_num_episodes', 'Episode Reward', 'Number of Training Episodes'),
        ('episode_length_vs_num_episodes', 'Episode Length', 'Number of Training Episodes'),
        ('policy_network_loss_vs_num_update_steps', 'Policy Network Loss', 'Number of Update Steps'),
        ('policy_network_loss_vs_num_update_epochs', 'Policy Network Loss', 'Number of Update Epochs'),
        ('value_network_loss_vs_num_update_steps', 'Value Network Loss', 'Number of Update Steps'),
        ('value_network_loss_vs_num_update_epochs', 'Value Network Loss', 'Number of Update Epochs'),
        ('state_value_estimation_vs_num_env_steps', 'State Value Estimation', 'Number of Environment Steps'),
        ('policy_action_distribution_entropy_vs_num_env_steps', 'Entropy of Policy Action Distribution', 'Number of Environment Steps'),
    ]
    
    agent, history = train_agent(
        agent=agent,
        env_name=config.ENV_NAME,
        n_envs=config.N_ENVS,
        n_training_steps=config.N_TRAINING_STEPS,
        n_policy_rollout_steps=config.N_POLICY_ROLLOUT_STEPS,
        n_minibatches=config.N_MINIBATCHES,
        policy_update_epochs=config.POLICY_UPDATE_EPOCHS,
        lr=config.LEARNING_RATE,
        lr_end=config.LEARNING_RATE_END,
        gamma=config.GAMMA,
        clip_epsilon=config.CLIP_EPSILON,
        gae_lambda=config.GAE_LAMBDA,
        max_gradient_norm=config.MAX_GRADIENT_NORM,
        value_loss_coefficient=config.VALUE_LOSS_COEFFICIENT,
        n_checkpoints=config.N_CHECKPOINTS,
        checkpoint_folder=f'{experiment_folder}/{config.CHECKPOINT_FOLDER}',
        history_attributes=history_attributes,
        history_folder=f'{experiment_folder}/{config.HISTORY_FOLDER}',
        check_tensor_shapes=config.CHECK_TENSOR_SHAPES,
        verbose=config.VERBOSE,
    )

    for attr_key, val_name, delta_name in history_attributes:
        make_plot(
            values=history[attr_key],
            title=f'{val_name} vs. {delta_name}',
            xlabel=delta_name,
            ylabel=val_name,
            fig_name=attr_key,
            fig_path=f'{experiment_folder}/{config.FIG_PATH}',
        )

    record_episode(
        env_name=config.ENV_NAME, 
        agent=agent,
        video_folder=f'{experiment_folder}/{config.VIDEO_FOLDER}',
        video_name_prefix=config.VIDEO_NAME_PREFIX,
    )
    

if __name__ == "__main__":
    main()