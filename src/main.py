import torch
import numpy as np
import random
import os

import config

from mbrlv1dot5.agent import MBRLv1dot5Agent
from mbrlv1dot5.train import (
    train as train_mbrlv1dot5,
    MBRLv1do5Config,
)

from mbpo.agent import MBPOAgent
from mbpo.train import (
    train as train_mbpo,
    MBPOConfig,
)

from environment import (
    play_recording_environment,
)
from utils import (
    make_history_plots,
    get_arguments,
    get_device,
    save_checkpoint,
)


def main():
    # set random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_arguments()

    experiment_folder = f'experiments/{args.method}/{args.experiment_id}'
    os.makedirs(experiment_folder, exist_ok=True)

    device = get_device(args.device)
    
    if args.method == 'mbrlv1dot5':
        print('Agent based on Model-based RL v1.5 algorithm')

        agent = MBRLv1dot5Agent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dim=MBRLv1do5Config.HIDDEN_DIM,
            planning_length=MBRLv1do5Config.MIN_PLANNING_LENGTH,
            n_candidate_actions=MBRLv1do5Config.N_CANDIDATES_ACTIONS,
            device=device,
            verbose=config.VERBOSE,
        )

        agent, history = train_mbrlv1dot5(
            agent=agent, 
            checkpoint_folder=f'{experiment_folder}/{config.CHECKPOINT_FOLDER}',
            history_folder=f'{experiment_folder}/{config.HISTORY_FOLDER}',
        )
    elif args.method == 'mbpo':
        print('Agent based on Model-Based Policy Optimization algorithm')

        agent = MBPOAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dim=MBPOConfig.HIDDEN_DIM,
            n_dyn_models=MBPOConfig.N_DYNAMICS_MODELS,
            device=device,
            verbose=config.VERBOSE,
        )

        agent, history = train_mbpo(
            agent=agent, 
            checkpoint_folder=f'{experiment_folder}/{config.CHECKPOINT_FOLDER}',
            history_folder=f'{experiment_folder}/{config.HISTORY_FOLDER}',
        )

    # import pickle
    # with open('experiments/mbrlv1dot5/v9/history/history_6.pkl', 'rb') as f:
    #     history = pickle.load(f)

    make_history_plots(
        history=history,
        fig_folder=f'{experiment_folder}/{config.FIG_PATH}',
    )

    play_recording_environment(
        env_name=config.ENV_NAME,
        agent=agent,
        video_folder=f'{experiment_folder}/{config.VIDEO_FOLDER}',
        video_name_prefix=config.VIDEO_NAME_PREFIX,
    )

if __name__ == "__main__":
    main()