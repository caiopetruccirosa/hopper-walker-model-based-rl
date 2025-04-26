import torch
import torch.nn as nn

from torch.distributions import Normal
from torchinfo import summary


class EnvDynamicsEnsembleModel(nn.Module):
    def __init__(self, n_models: int, state_dim: int, action_dim: int, hidden_dim: int):
        super(EnvDynamicsEnsembleModel, self).__init__()

        self.n_models = n_models
        
        # layers parameters
        self.fc1_w = nn.Parameter(torch.randn(n_models, state_dim+action_dim, hidden_dim))
        self.fc1_b = nn.Parameter(torch.randn(n_models, 1, hidden_dim))

        self.fc2_w = nn.Parameter(torch.randn(n_models, hidden_dim, hidden_dim))
        self.fc2_b = nn.Parameter(torch.randn(n_models, 1, hidden_dim))

        self.fc_state_out_w = nn.Parameter(torch.randn(n_models, hidden_dim, state_dim))
        self.fc_state_out_b = nn.Parameter(torch.randn(n_models, 1, state_dim))

        self.fc_reward_out_w = nn.Parameter(torch.randn(n_models, hidden_dim, 1))
        self.fc_reward_out_b = nn.Parameter(torch.randn(n_models, 1, 1))

        self.relu = nn.ReLU()

        # he initialization
        nn.init.kaiming_normal_(self.fc1_w)
        nn.init.kaiming_normal_(self.fc2_w)
        nn.init.kaiming_normal_(self.fc_state_out_w)
        nn.init.kaiming_normal_(self.fc_reward_out_w)
        nn.init.zeros_(self.fc1_b)
        nn.init.zeros_(self.fc2_b)
        nn.init.kaiming_normal_(self.fc_state_out_b)
        nn.init.kaiming_normal_(self.fc_reward_out_b)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)

        # add model dimension, x shape: (batch_size, input_size) -> (1, batch_size, input_size)
        x = x.unsqueeze(0)
        x = x.expand(self.n_models, -1, -1)
        
        # first layer. x shape (1, batch_size, input_size) -> (n_models, batch_size, hidden_dim)
        x = torch.bmm(x, self.fc1_w) + self.fc1_b
        x = self.relu(x)
        
        # second layer. x shape (n_models, batch_size, hidden_dim) -> (n_models, batch_size, hidden_dim)
        x = torch.bmm(x, self.fc2_w) + self.fc2_b

        # shape (n_models, batch_size, hidden_dim) -> (n_models, batch_size, state_dim) -> (batch_size, state_dim)
        next_state = torch.bmm(x, self.fc_state_out_w) + self.fc_state_out_b
        next_state = next_state.mean(dim=0)

        # shape (n_models, batch_size, hidden_dim) -> (n_models, batch_size, 1) -> (batch_size, 1)
        reward = torch.bmm(x, self.fc_reward_out_w) + self.fc_reward_out_b
        reward = reward.mean(dim=0).squeeze(dim=-1)

        return next_state, reward


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_out = nn.Linear(hidden_dim, action_dim)
        self.log_std_out = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_out(x)
        log_std = self.log_std_out(x)
        return mean, log_std
    
    def get_distribution(self, state: torch.Tensor) -> Normal:
        mean, log_std = self(state)
        std = log_std.clamp(-20, 1).exp()
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Critic, self).__init__()
        self.q1_net = self._make_q_network(state_dim, action_dim, hidden_dim)
        self.q2_net = self._make_q_network(state_dim, action_dim, hidden_dim)

    def _make_q_network(self, state_dim: int, action_dim: int, hidden_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(x).squeeze(dim=-1)
        q2 = self.q2_net(x).squeeze(dim=-1)
        return q1, q2


class MBPOAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int, 
        n_dyn_models: int,
        device: torch.device, 
        verbose=False,
    ):
        super(MBPOAgent, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.env_dynamics_model = EnvDynamicsEnsembleModel(n_dyn_models, state_dim, action_dim, hidden_dim)
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(size=(1,), requires_grad=True, device=device))

        if verbose:
            print(f'======== Environment Dynamics Ensemble Model ========')
            summary(
                self.env_dynamics_model, 
                input_size=[(1, self.state_dim), (1, self.action_dim)],
                dtypes=[torch.float32, torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )
            print(f'=================== Actor Network ===================')
            summary(
                self.actor, 
                input_size=[(1, self.state_dim)],
                dtypes=[torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )
            print(f'================== Critic Networks ==================')
            summary(
                self.critic, 
                input_size=[(1, self.state_dim), (1, self.action_dim)],
                dtypes=[torch.float32, torch.float32],
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
            )

        self.env_dynamics_model = self.env_dynamics_model.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        self.log_alpha = self.log_alpha.to(device)

    def predict_transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.env_dynamics_model(state, action)
    
    def choose_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_dist = self.actor.get_distribution(state)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action).sum(dim=-1)

        return action, log_probs

    def to_device(self, device: torch.device) -> 'MBPOAgent':
        self.device = device
        self.env_dynamics_model = self.env_dynamics_model.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        return self