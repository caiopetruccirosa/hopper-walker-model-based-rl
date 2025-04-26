import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

from tqdm import tqdm
from typing import Any
from dataclasses import dataclass

import config

from mbpo.agent import MBPOAgent
from environment import (
    make_vectorized_env,
    ReplayBuffer,
)
from utils import (
    create_history,
    add_to_history,
    save_checkpoint,
)


@dataclass
class MBPOConfig:
    HIDDEN_DIM                  = 64
    DYNAMICS_LR                 = 3e-5
    POLICY_LR                   = 1e-4
    BATCH_SIZE                  = 128
    REPLAY_BUFFER_SIZE          = 100000
    N_EPOCHS                    = 125
    N_EXPLORATION_STEPS         = 500
    N_TRAINING_STEPS_PER_EPOCH  = 1000
    N_POLICY_UPDATES            = 20
    N_DYNAMICS_MODEL_UPDATES    = 10
    N_DYNAMICS_MODELS           = 5
    REWARD_LOSS_WEIGHT          = 0.25
    DISCOUNT_FACTOR             = 0.99
    CRITIC_SOFT_UPDATE_FACTOR   = 0.995
    PLANNING_ROLLOUT_SIZE       = 100
    MIN_PLANNING_LENGTH         = 1
    MAX_PLANNING_LENGTH         = 15
    EPOCH_START_PLANNING_GROWTH = 20
    EPOCH_END_PLANNING_GROWTH   = 100
    MAX_PLANNING_LENGTH         = 15
    WEIGHT_DECAY                = 1e-4



# ----------------
# Training Process
# ----------------

def calculate_planning_length(epoch: int) -> int:
    planning_length_interval = MBPOConfig.MAX_PLANNING_LENGTH - MBPOConfig.MIN_PLANNING_LENGTH
    epochs_interval = MBPOConfig.EPOCH_END_PLANNING_GROWTH - MBPOConfig.EPOCH_START_PLANNING_GROWTH
    current_epoch_offset = epoch - MBPOConfig.EPOCH_START_PLANNING_GROWTH
    ratio = current_epoch_offset/epochs_interval
    return min(max(int(MBPOConfig.MIN_PLANNING_LENGTH + ratio*planning_length_interval), MBPOConfig.MIN_PLANNING_LENGTH), MBPOConfig.MAX_PLANNING_LENGTH)

def add_planning_rollout_to_buffer(agent: MBPOAgent, env_replay_buffer: ReplayBuffer, planning_replay_buffer: ReplayBuffer, epoch: int):
    states, _, _, _, _ = env_replay_buffer.sample(MBPOConfig.PLANNING_ROLLOUT_SIZE)
    states = states.to(agent.device)
    
    planning_length = calculate_planning_length(epoch)
    for _ in range(planning_length):
        with torch.no_grad():
            actions, _ = agent.choose_action(states)
            next_states, rewards = agent.predict_transition(states, actions)

        planning_replay_buffer.add(
            states=states.cpu(), 
            actions=actions.detach().cpu(), 
            next_states=next_states.detach().cpu(), 
            rewards=rewards.detach().cpu(), 
            dones=torch.zeros(size=rewards.size()),
        )
        states = next_states


def optimize_agent_policy(
    agent: MBPOAgent, 
    env_replay_buffer: ReplayBuffer, 
    planning_replay_buffer: ReplayBuffer, 
    optimizer_actor: optim.Adam,
    optimizer_critic: optim.Adam,
    optimizer_alpha: optim.Adam,
    target_entropy: float,
    history: dict[str, dict[str, Any]],
):
    env_states, env_actions, env_next_states, env_rewards, env_dones = env_replay_buffer.sample(MBPOConfig.BATCH_SIZE//2)
    planning_states, planning_actions, planning_next_states, planning_rewards, planning_dones = planning_replay_buffer.sample(MBPOConfig.BATCH_SIZE//2)
    
    states = torch.cat([env_states, planning_states], dim=0).to(agent.device)
    actions = torch.cat([env_actions, planning_actions], dim=0).to(agent.device)
    rewards = torch.cat([env_rewards, planning_rewards], dim=0).to(agent.device)
    next_states = torch.cat([env_next_states, planning_next_states], dim=0).to(agent.device)
    dones = torch.cat([env_dones, planning_dones], dim=0).to(agent.device)
    
    # calculate and optimize critic loss
    with torch.no_grad():
        next_actions, next_actions_log_probs = agent.choose_action(next_states)
        next_q1, next_q2 = agent.target_critic(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2)
        alpha = agent.log_alpha.exp()
        target_q = rewards + (1-dones) * MBPOConfig.DISCOUNT_FACTOR * (next_q - alpha*next_actions_log_probs)
    
    current_q1, current_q2 = agent.critic(states, actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    # calculate and optimize actor loss
    new_actions, new_actions_log_probs = agent.choose_action(states)
    q1, q2 = agent.critic(states, new_actions)
    q = torch.min(q1, q2)
    actor_loss = (alpha*new_actions_log_probs - q).mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # calculate and optimize alpha loss
    alpha_loss = -(agent.log_alpha * (new_actions_log_probs.detach()+target_entropy)).mean()

    optimizer_alpha.zero_grad()
    alpha_loss.backward()
    optimizer_alpha.step()
    
    # update target critic network
    for param, target_param in zip(agent.critic.parameters(), agent.target_critic.parameters()):
        target_param.data.copy_(MBPOConfig.CRITIC_SOFT_UPDATE_FACTOR*target_param.data + MBPOConfig.CRITIC_SOFT_UPDATE_FACTOR*param.data)

    add_to_history(history, 'policy_critic_loss_vs_num_polupdate_steps', critic_loss.item())
    add_to_history(history, 'policy_actor_loss_vs_num_polupdate_steps', actor_loss.item())
    add_to_history(history, 'policy_alpha_loss_vs_num_polupdate_steps', alpha_loss.item())


def optimize_agent_dynamics_model(agent: MBPOAgent, replay_buffer: ReplayBuffer, optimizer: optim.Adam, history: dict[str, dict[str, Any]]):
    sampled_states, sampled_actions, sampled_next_states, sampled_rewards, _ = replay_buffer.sample(MBPOConfig.BATCH_SIZE)
    sampled_states = sampled_states.to(agent.device)
    sampled_actions = sampled_actions.to(agent.device)
    sampled_next_states = sampled_next_states.to(agent.device)
    sampled_rewards = sampled_rewards.to(agent.device)

    pred_next_states, pred_rewards = agent.predict_transition(sampled_states, sampled_actions)
    state_loss = F.mse_loss(pred_next_states, sampled_next_states)
    reward_loss = F.mse_loss(pred_rewards.squeeze(-1), sampled_rewards)
    dyn_loss = state_loss + MBPOConfig.REWARD_LOSS_WEIGHT*reward_loss

    optimizer.zero_grad()
    dyn_loss.backward()
    optimizer.step()

    add_to_history(history, 'state_dynamics_loss_vs_num_dynupdate_steps', state_loss.item())
    add_to_history(history, 'reward_dynamics_loss_vs_num_dynupdate_steps', reward_loss.item())
    add_to_history(history, 'total_dynamics_loss_vs_num_dynupdate_steps', dyn_loss.item())


def train(agent: MBPOAgent, checkpoint_folder: str, history_folder: str):
    if config.VERBOSE:
        print(f'[DEVICE] Training on \'{str(agent.device).upper()}\' device.')

    history = create_history(
        attributes=[
            ('episode_reward_vs_num_episodes', 'Episode Reward', 'Number of Episodes', True),
            ('policy_critic_loss_vs_num_polupdate_steps', 'Agent\'s Policy Critic Loss', 'Number of Update Steps', False),
            ('policy_actor_loss_vs_num_polupdate_steps', 'Agent\'s Policy Actor Loss', 'Number of Update Steps', False),
            ('policy_alpha_loss_vs_num_polupdate_steps', 'Agent\'s Policy Alpha Loss', 'Number of Update Steps', False),
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

    target_entropy = -math.prod(envs.action_space.shape) # type: ignore

    env_replay_buffer = ReplayBuffer(MBPOConfig.REPLAY_BUFFER_SIZE)
    planning_replay_buffer = ReplayBuffer(MBPOConfig.REPLAY_BUFFER_SIZE)

    optimizer_dynamics = optim.Adam(agent.env_dynamics_model.parameters(), lr=MBPOConfig.DYNAMICS_LR, weight_decay=MBPOConfig.WEIGHT_DECAY)
    optimizer_actor = optim.Adam(agent.actor.parameters(), lr=MBPOConfig.POLICY_LR, weight_decay=MBPOConfig.WEIGHT_DECAY)
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=MBPOConfig.POLICY_LR, weight_decay=MBPOConfig.WEIGHT_DECAY)
    optimizer_alpha = optim.Adam([agent.log_alpha], lr=MBPOConfig.POLICY_LR, weight_decay=MBPOConfig.WEIGHT_DECAY)

    # exploration steps
    states, _ = envs.reset()
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)

    print('Exploration steps...')
    for _ in tqdm(range(0, MBPOConfig.N_EXPLORATION_STEPS, config.N_ENVS)):
        actions = envs.action_space.sample()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # add environment transition of not done environments to replay buffer
        env_replay_buffer.add(
            states=torch.FloatTensor(states[~dones, :]),
            actions=torch.FloatTensor(actions[~dones]),
            next_states=torch.FloatTensor(next_states[~dones, :]),
            rewards=torch.FloatTensor(rewards[~dones]),
            dones=torch.ByteTensor(dones[~dones]),
        )

        states = next_states

    # training agent
    states, _ = envs.reset()
    dones = np.zeros(shape=config.N_ENVS, dtype=bool)
    episodes_acc_reward = np.zeros(shape=config.N_ENVS, dtype=float)

    print('Training...')
    for epoch in tqdm(range(MBPOConfig.N_EPOCHS)):
        if len(env_replay_buffer) >= MBPOConfig.BATCH_SIZE:
            for _ in range(MBPOConfig.N_DYNAMICS_MODEL_UPDATES):
                optimize_agent_dynamics_model(agent, env_replay_buffer, optimizer_dynamics, history)

        training_steps = 0
        while training_steps < MBPOConfig.N_TRAINING_STEPS_PER_EPOCH:
            states_pt = torch.FloatTensor(states)

            with torch.no_grad():
                actions, _ = agent.choose_action(states_pt.to(device))
                actions = actions.detach().cpu()
            
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
            dones = np.logical_or(terminateds, truncateds)

            # add environment transition of not done environments to replay buffer
            env_replay_buffer.add(
                states=states_pt[~dones, :],
                actions=actions[~dones],
                next_states=torch.FloatTensor(next_states[~dones, :]),
                rewards=torch.FloatTensor(rewards[~dones]),
                dones=torch.ByteTensor(dones[~dones]),
            )

            states = next_states
            episodes_acc_reward += rewards

            # generate planning rollout to add to replay experience buffer
            if len(env_replay_buffer) >= MBPOConfig.PLANNING_ROLLOUT_SIZE:
                add_planning_rollout_to_buffer(agent, env_replay_buffer, planning_replay_buffer, epoch)


            # optimize agent's policy
            if len(env_replay_buffer) >= MBPOConfig.BATCH_SIZE//2 and len(planning_replay_buffer) >= MBPOConfig.BATCH_SIZE//2:
                for _ in range(MBPOConfig.N_POLICY_UPDATES):
                    optimize_agent_policy(
                        agent, 
                        env_replay_buffer, 
                        planning_replay_buffer, 
                        optimizer_actor,
                        optimizer_critic,
                        optimizer_alpha,
                        target_entropy,
                        history,
                    )

            # process finished episodes and train policy with REINFORCE
            if sum(dones) > 0:
                if config.VERBOSE:
                    print('\t'.join(f'Reward {i}: {r:.2f}' for i, r in enumerate(episodes_acc_reward[dones], 1)))
                add_to_history(history, 'episode_reward_vs_num_episodes', *episodes_acc_reward[dones].tolist())
                episodes_acc_reward[dones] = 0

            training_steps += config.N_ENVS
                
        # save agent and history checkpoint
        save_checkpoint(agent, history, epoch, checkpoint_folder, history_folder)

    envs.close()

    save_checkpoint(agent, history, MBPOConfig.N_EPOCHS, checkpoint_folder, history_folder)

    return agent, history