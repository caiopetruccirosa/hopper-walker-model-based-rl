import torch
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy

from argparse import Namespace
from typing import Any

from mbrlv1dot5.agent import MBRLv1dot5Agent
from mbpo.agent import MBPOAgent


# --------------------------
# History Tracking Functions
# --------------------------

def create_history(attributes: list[tuple[str, str, str, bool]]) -> dict[str, dict[str, Any]]:
    return { 
        attr_key: { 
            'attr_name': attr_name,
            'delta_name': delta_name,
            'values': [],
            'plot_rolling_avg': plot_rolling_avg
        } for attr_key, attr_name, delta_name, plot_rolling_avg in attributes 
    }

def add_to_history(history: dict[str, dict[str, Any]], key: str, *values: float):
    history[key]['values'] += values

def save_checkpoint(
    agent: MBRLv1dot5Agent|MBPOAgent, 
    history: dict[str, dict[str, Any]],  
    checkpoint_idx: int, 
    checkpoint_folder: str, 
    history_folder: str,
):
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(history_folder, exist_ok=True)

    agent = copy.deepcopy(agent)
    agent = agent.to_device(torch.device('cpu'))

    with open(f'{checkpoint_folder}/agent_chkpt_{checkpoint_idx}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    with open(f'{history_folder}/history_{checkpoint_idx}.pkl', 'wb') as f:
        pickle.dump(history, f)

def make_history_plots(
    history: dict[str, dict[str, Any]], 
    fig_folder: str,
):
    for attr_key, attr_hist in history.items():
        make_plot(
            values=attr_hist['values'],
            plot_rolling_avg=attr_hist['plot_rolling_avg'],
            title=f'{attr_hist['attr_name']} vs. {attr_hist['delta_name']}',
            xlabel=attr_hist['delta_name'],
            ylabel=attr_hist['attr_name'],
            fig_name=attr_key,
            fig_folder=fig_folder,
        )


# ----------------------------
# Training Arguments Functions
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
# Miscellaneous Functions
# -----------------------

def pad_number_representation(n: int, max_n: int) -> str:
    return str(n).zfill(len(str(max_n)))

def make_plot(
    values: list[float], 
    plot_rolling_avg: bool,
    title: str, 
    xlabel: str, 
    ylabel: str, 
    fig_name: str, 
    fig_folder: str,
):
    os.makedirs(fig_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 6))

    if plot_rolling_avg:
        window_size = len(values)//25
        x = np.arange(len(values))
        
        series = pd.Series(values)
        rolling_avg = series.rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
        
        sns.lineplot(x=x, y=values, linewidth=0.5, alpha=0.4, color='tab:blue')
        sns.lineplot(x=x, y=rolling_avg, linewidth=1.5, color='tab:blue')
        plt.fill_between(x, rolling_avg - rolling_std, rolling_avg + rolling_std, alpha=0.15, color='tab:blue')
    else:
        sns.lineplot(values, linewidth=1.2, color='tab:blue')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{fig_folder}/{fig_name}', dpi=300, bbox_inches='tight')
    plt.close()

def get_arguments() -> Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--method', '-m', help='Method/algorithm to use for agent.', type=str, required=True, choices=['mbrlv1dot5', 'mbpo'])
    arg_parser.add_argument('--experiment_id', '-e', help='Experiment ID. Used to store artifacts from experiment in a specific folder.', type=str, required=True)
    arg_parser.add_argument('--device', '-d', help='PyTorch device to use for tensor operations during training.', type=str, required=False)
    return arg_parser.parse_args()