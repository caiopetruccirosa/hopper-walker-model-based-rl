import torch
import torch.nn as nn
import numpy as np

from torchinfo import summary


class EnvDynamicsEnsembleModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(EnvDynamicsEnsembleModel, self).__init__()

        self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_state = nn.Linear(hidden_dim, state_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        next_state = self.fc_state(x)
        reward = self.fc_reward(x)
        return next_state, reward
    

class MBPOAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int, 
        # planning_length: int,
        # n_candidate_actions: int,
        device: torch.device, 
        verbose=False,
    ):
        super(MBPOAgent, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.planning_length = planning_length
        self.n_candidate_actions = n_candidate_actions

        self.env_dynamics_model = EnvDynamicsModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )

        if verbose:
            print('================================================================')
            print(f'==                 Environment Dynamics Model                ==')
            print('================================================================')
            summary(
                self.env_dynamics_model, 
                input_size=[(self.state_dim,), (self.action_dim,)],
                dtypes=[torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )

        self.env_dynamics_model = self.env_dynamics_model.to(device)

    def predict_transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.to(self.device)
        action = action.to(self.device)
        return self.env_dynamics_model(state, action)

    @torch.no_grad()
    def choose_action(self, state: torch.Tensor) -> np.ndarray:
        state = state.to(self.device)

        # check if state is batched
        not_batched = state.dim() == 1
        if not_batched:
            state = state.unsqueeze(0)

        batch_size, _ = state.size()
        
        # candidate_action_sequences shape: (batch_size, n_candidate_actions, planning_length, action_dim)
        candidate_action_sequences = np.random.uniform(
            low=-1,
            high=1,
            size=(batch_size, self.n_candidate_actions, self.planning_length, self.action_dim),
        )
        # cumulative_rewards shape: (batch_size, n_candidate_actions)
        cumulative_rewards = np.zeros(shape=(batch_size, self.n_candidate_actions))

        # state shape: (batch_size, state_dim)
        # current_state shape: (batch_size, n_candidate_actions, state_dim)
        current_state = state.unsqueeze(1).expand(-1, self.n_candidate_actions, -1)
        for i in range(self.planning_length):
            # action shape: (batch_size, n_candidate_actions, action_dim)
            action = torch.FloatTensor(candidate_action_sequences[:, :, i, :]).to(self.device)

            # next_state shape: (batch_size, n_candidate_actions, state_dim)
            # reward shape: (batch_size, n_candidate_actions, 1)
            next_state, reward = self.predict_transition(current_state, action)

            cumulative_rewards += reward.detach().cpu().numpy().squeeze(-1)
            current_state = next_state

        # best_idx shape: (batch_size)
        best_idx = np.argmax(cumulative_rewards, axis=1)
        
        # best_action shape: (batch_size, action_dim)
        best_action = candidate_action_sequences[np.arange(batch_size), best_idx, 0, :]

        if not_batched:
            best_action = best_action.squeeze(0)

        return best_action
