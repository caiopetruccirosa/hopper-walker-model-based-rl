import gymnasium as gym
import random
import torch
import os

from gymnasium.wrappers import (
    RecordEpisodeStatistics, 
    # NormalizeObservation, 
    # NormalizeReward, 
    RecordVideo,
)
from collections import deque

from mbrl1dot5_agent import MBRLv1dot5Agent


# ----------------------------
# Replay Buffer of transitions
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor):
        self.buffer.extend(zip(states, actions, next_states, rewards))
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, next_states, rewards = zip(*random.sample(self.buffer, batch_size))
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.stack(rewards, dim=0)
        return states, actions, next_states, rewards
    
    def __len__(self):
        return len(self.buffer)


# --------------------------
# Make Gymnasium Environment
# --------------------------
def make_env(env_name: str, healthy_reward: float, forward_reward_weight: float, ctrl_cost_weight: float, render_mode=None, is_recording: bool=False) -> gym.Env:
    env = gym.make(
        env_name, 
        healthy_reward=healthy_reward,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        terminate_when_unhealthy=False,
        render_mode=render_mode,
    )
    env = RecordEpisodeStatistics(env) # adds episode statistics to env.step return values
    # env = NormalizeObservation(env)    # normalizes state observation (subtracting mean and dividing by variance)
    # env = NormalizeReward(env)         # normalizes rewards by applying discount-based scaling (divides by the standard deviation of a rolling discounted sum of the rewards)
    return env

def make_vectorized_env(env_name: str, n_envs: int, healthy_reward: float, forward_reward_weight: float, ctrl_cost_weight: float) -> gym.vector.SyncVectorEnv:
    make_env_named = lambda : make_env(env_name, healthy_reward, forward_reward_weight, ctrl_cost_weight, render_mode=None, is_recording=False)
    return gym.vector.SyncVectorEnv([make_env_named for _ in range(n_envs)])