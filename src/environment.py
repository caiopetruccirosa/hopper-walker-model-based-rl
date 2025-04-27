import gymnasium as gym
import random
import torch
import os

from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    NormalizeReward,
    RecordVideo,
)
from collections import deque

from mbrlv1dot5.agent import MBRLv1dot5Agent
from mbpo.agent import MBPOAgent


# ----------------------------
# Replay Buffer of transitions
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        self.buffer.extend(zip(states, actions, next_states, rewards, dones))
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, next_states, rewards, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.stack(rewards, dim=0)
        dones = torch.stack(dones, dim=0)
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.buffer)


# ------------------------------------------
# Play Recording Environment For One Episode
# ------------------------------------------
def play_recording_environment(env_name: str, agent: MBRLv1dot5Agent|MBPOAgent, video_folder: str, video_name_prefix: str):
    os.makedirs(video_folder, exist_ok=True)
    
    env = make_env(env_name, healthy_reward=0, forward_reward_weight=0, ctrl_cost_weight=0, render_mode='rgb_array', is_recording=True)
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name_prefix, episode_trigger=lambda _: True)
    
    state, _ = env.reset()
    done = False

    while not done:
        state_pt = torch.FloatTensor(state).to(agent.device)
        if type(agent) is MBRLv1dot5Agent:
            action = agent.choose_action(state_pt).cpu()
        elif type(agent) is MBPOAgent:
            action, _ = agent.choose_action(state_pt)
            action = action.detach().cpu()
        next_state, _, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        state = next_state
    
    env.close()


# --------------------------
# Make Gymnasium Environment
# --------------------------
def make_env(env_name: str, healthy_reward: float, forward_reward_weight: float, ctrl_cost_weight: float, render_mode=None, is_recording: bool=False) -> gym.Env:
    env = gym.make(
        env_name, 
        healthy_reward=healthy_reward,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        terminate_when_unhealthy=(not is_recording),
        render_mode=render_mode,
    )
    env = RecordEpisodeStatistics(env) # adds episode statistics to env.step return values
    env = NormalizeReward(env)
    return env

def make_vectorized_env(env_name: str, n_envs: int, healthy_reward: float, forward_reward_weight: float, ctrl_cost_weight: float) -> gym.vector.SyncVectorEnv:
    make_env_named = lambda : make_env(env_name, healthy_reward, forward_reward_weight, ctrl_cost_weight, render_mode=None, is_recording=False)
    return gym.vector.SyncVectorEnv([make_env_named for _ in range(n_envs)])