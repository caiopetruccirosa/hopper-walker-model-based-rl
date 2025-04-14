import gymnasium as gym
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from gymnasium.wrappers import RecordEpisodeStatistics, NormalizeObservation, NormalizeReward
from typing import Any, SupportsFloat

from agent import CheetahAgent


# ---------------------
# Environment functions
# ---------------------

def make_env(env_name: str, render_mode=None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    env = RecordEpisodeStatistics(env) # adds episode statistics to env.step return values
    env = NormalizeObservation(env) # normalizes state observation (subtracting mean and dividing by variance)
    env = NormalizeReward(env) # normalizes rewards by applying discount-based scaling (divides by the standard deviation of a rolling discounted sum of the rewards)
    return env

def make_vectorized_env(env_name: str, n_envs: int) -> gym.vector.SyncVectorEnv:
    make_env_named = lambda : make_env(env_name)
    return gym.vector.SyncVectorEnv([make_env_named for _ in range(n_envs)])

def preprocess_env_step(
    next_state: np.ndarray, 
    reward: SupportsFloat, 
    terminated: bool, 
    truncated: bool, 
    info: dict[str, Any]
) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    return (next_state, float(reward), (terminated or truncated), info)

def preprocess_vectorized_env_step(
    next_state: np.ndarray, 
    reward: np.ndarray, 
    terminated: np.ndarray, 
    truncated: np.ndarray, 
    info: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return (next_state, reward, np.logical_or(terminated, truncated), info)


# --------------------------
# History tracking functions
# --------------------------

def create_history(attributes: list[tuple[str, str, str]]) -> dict[str, list]:
    return { attr_key: [] for attr_key, _, _ in attributes }

def add_to_history(history: dict[str, list], key: str, *values: float):
    history[key] += values

def save_checkpoint(
    agent: CheetahAgent, 
    history: dict[str, list],  
    checkpoint_idx: int, 
    checkpoint_folder: str, 
    history_folder: str,
):
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(history_folder, exist_ok=True)

    with open(f'{checkpoint_folder}/agent_chkpt_{checkpoint_idx}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    with open(f'{history_folder}/history_{checkpoint_idx}.pkl', 'wb') as f:
        pickle.dump(history, f)


# ----------------------------
# Training arguments functions
# ----------------------------

def get_device(device_argument: str|None) -> torch.device:
    if device_argument is not None:
        return torch.device(device_argument)
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# -----------------------
# Miscellaneous functions
# ----------------

def to_float_tensor(*arrays: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(arr, dtype=torch.float32) for arr in arrays)

def pad_number_representation(n: int, max_n: int) -> str:
    return str(n).zfill(len(str(max_n)))

def make_plot(values: list[float], title: str, xlabel: str, ylabel: str, fig_name: str, fig_path: str):
    os.makedirs(fig_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(values, linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{fig_path}/{fig_name}', dpi=300)
    plt.close()
