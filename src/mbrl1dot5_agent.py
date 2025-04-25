import torch
import torch.nn as nn
import numpy as np

from torch import distributions
from torchinfo import summary


def make_linear_layer(in_dim: int, out_dim: int, std: float = np.sqrt(2), bias_const: float = 0) -> nn.Linear:
    layer = nn.Linear(in_features=in_dim, out_features=out_dim, dtype=torch.float32)
    torch.nn.init.orthogonal_(layer.weight, std) # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const) # type: ignore
    return layer


class EnvDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(EnvDynamicsModel, self).__init__()

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


class PolicyModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(PolicyModel, self).__init__()

        self.base_net = nn.Sequential(
            make_linear_layer(in_dim=state_dim, out_dim=hidden_dim),
            nn.Tanh(),
            make_linear_layer(in_dim=hidden_dim, out_dim=hidden_dim),
            nn.Tanh(),
        )
        self.mean_out = nn.Sequential(
            make_linear_layer(in_dim=hidden_dim, out_dim=action_dim, std=1e-2), 
            nn.Tanh(),
        )
        self.log_std_out = make_linear_layer(in_dim=hidden_dim, out_dim=action_dim, std=1e-2)
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.base_net(state)
        return self.mean_out(x), self.log_std_out(x)
    
    def _get_action_distribution(self, state: torch.Tensor) -> distributions.Normal:
        # state shape: (batch_size, state_dim)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std.expand_as(mean))
        action_dist = distributions.Normal(mean, std)
        return action_dist

    def get_action_logprobs_entropy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_dist = self._get_action_distribution(state)

        # action_logprobs and action_entropy shape: (batch_size)
        action_logprobs = action_dist.log_prob(action).sum(dim=-1)
        action_entropy = action_dist.entropy().sum(dim=-1)
        
        return action_logprobs, action_entropy

    def choose_action(self, state: torch.Tensor) -> torch.Tensor:
        # state shape: (batch_size, state_dim)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std.expand_as(mean))
        action_dist = distributions.Normal(mean, std)
    
        # action shape: (batch_size, action_dim)
        action = action_dist.sample().clamp(min=-1, max=1)

        return action


class MBRLv1dot5Agent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int, 
        planning_length: int,
        n_candidate_actions: int,
        device: torch.device, 
        verbose=False,
    ):
        super(MBRLv1dot5Agent, self).__init__()

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

        self.policy_model = PolicyModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )

        if verbose:
            print('=============================================')
            print(f'==       Environment Dynamics Model       ==')
            print('=============================================')
            summary(
                self.env_dynamics_model, 
                input_size=[(self.state_dim,), (self.action_dim,)],
                dtypes=[torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )
            print('=============================================')
            print(f'==              Policy Model              ==')
            print('=============================================')
            summary(
                self.policy_model, 
                input_size=[(self.state_dim,),],
                dtypes=[torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )

        self.env_dynamics_model = self.env_dynamics_model.to(device)
        self.policy_model = self.policy_model.to(device)

    def predict_transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.to(self.device)
        action = action.to(self.device)
        return self.env_dynamics_model(state, action)

    def choose_action_on_random_policy(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)

        # check if state is batched
        not_batched = state.dim() == 1
        if not_batched:
            state = state.unsqueeze(0)

        batch_size, _ = state.size()
        
        # candidate_action_sequences shape: (batch_size, n_candidate_actions, planning_length, action_dim)
        candidate_action_sequences = torch.tensor(
            np.random.uniform(
                low=-1,
                high=1,
                size=(batch_size, self.n_candidate_actions, self.planning_length, self.action_dim),
            )
        ).to(self.device)

        # cumulative_rewards shape: (batch_size, n_candidate_actions)
        cumulative_rewards = torch.zeros(size=(batch_size, self.n_candidate_actions)).to(self.device)

        # state shape: (batch_size, state_dim)
        # current_state shape: (batch_size, n_candidate_actions, state_dim)
        current_state = state.unsqueeze(1).expand(-1, self.n_candidate_actions, -1)
        for i in range(self.planning_length):
            # action shape: (batch_size, n_candidate_actions, action_dim)
            action = torch.FloatTensor(candidate_action_sequences[:, :, i, :]).to(self.device)

            # next_state shape: (batch_size, n_candidate_actions, state_dim)
            # reward shape: (batch_size, n_candidate_actions, 1)
            with torch.no_grad():
                next_state, reward = self.predict_transition(current_state, action)

            cumulative_rewards += reward.squeeze(-1)
            current_state = next_state

        # best_idx shape: (batch_size)
        best_idx = torch.argmax(cumulative_rewards, dim=1).to(self.device)
        
        # best_action shape: (batch_size, action_dim)
        best_action = candidate_action_sequences[np.arange(batch_size), best_idx, 0]

        if not_batched:
            best_action = best_action.squeeze(0)

        return best_action.detach()
    
    def choose_action_learned_policy(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)

        # check if state is batched
        not_batched = state.dim() == 1
        if not_batched:
            state = state.unsqueeze(0)

        batch_size, _ = state.size()
        
        # cumulative_rewards shape: (batch_size, n_candidate_actions)
        cumulative_rewards = torch.zeros(size=(batch_size, self.n_candidate_actions)).to(self.device)

        # state shape: (batch_size, state_dim)
        # current_state shape: (batch_size, n_candidate_actions, state_dim)
        current_state = state.unsqueeze(1).expand(-1, self.n_candidate_actions, -1).to(self.device)

        # planning_steps length: (planning_length) of tensors of shape (batch_size, n_candidate_actions, action_dim)
        planning_steps = []
        for _ in range(self.planning_length):
            with torch.no_grad():
                # action shape: (batch_size, n_candidate_actions, action_dim)
                action = self.policy_model.choose_action(current_state)

                # next_state shape: (batch_size, n_candidate_actions, state_dim)
                # reward shape: (batch_size, n_candidate_actions, 1)
                next_state, reward = self.predict_transition(current_state, action)

            planning_steps.append(action)
            cumulative_rewards += reward.squeeze(-1)
            current_state = next_state

        # actions_seqs shape: (batch_size, n_candidate_actions, planning_length, action_dim)
        actions_seqs = torch.stack(planning_steps, dim=2) 

        # best_idx shape: (batch_size)
        best_idx = torch.argmax(cumulative_rewards, dim=1)
        
        # best_action shape: (batch_size, action_dim)
        best_action = actions_seqs[torch.arange(batch_size), best_idx, 0]

        if not_batched:
            best_action = best_action.squeeze(0)

        return best_action.detach()