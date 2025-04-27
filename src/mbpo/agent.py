import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal
from torchinfo import summary


def make_linear_layer(in_dim: int, out_dim: int, std: float = np.sqrt(2), bias_const: float = 0) -> nn.Linear:
    layer = nn.Linear(in_features=in_dim, out_features=out_dim, dtype=torch.float32)
    torch.nn.init.orthogonal_(layer.weight, std) # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const) # type: ignore
    return layer


class EnvDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.base_network = nn.Sequential(
            make_linear_layer(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            make_linear_layer(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.state_mean_out = make_linear_layer(hidden_dim, state_dim)
        self.state_log_std_out = make_linear_layer(hidden_dim, state_dim)
        self.reward_mean_out = make_linear_layer(hidden_dim, 1)
        self.reward_log_std_out = make_linear_layer(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        x = self.base_network(x)

        state_mean = self.state_mean_out(x)
        state_log_std = self.state_log_std_out(x).clamp(-20, 2)

        reward_mean = self.reward_mean_out(x).squeeze(dim=-1)
        reward_log_std = self.reward_log_std_out(x).squeeze(dim=-1).clamp(-20, 2)
        
        return state_mean, state_log_std, reward_mean, reward_log_std


class EnvDynamicsEnsemble(nn.Module):
    def __init__(self, n_models: int, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.models = nn.ModuleList([
            EnvDynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(n_models)
        ])
        self.n_models = n_models

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = []
        for model in self.models:
            preds.append(model(state, action))

        s_means, s_log_stds, r_means, r_log_stds = zip(*preds)
        s_means = torch.stack(s_means, dim=0)
        s_log_stds = torch.stack(s_log_stds, dim=0)
        r_means = torch.stack(r_means, dim=0)
        r_log_stds = torch.stack(r_log_stds, dim=0)

        return s_means, s_log_stds, r_means, r_log_stds

    def sample(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_batched = False
        if state.dim() > 1:
            is_batched = True
            batch_size = state.size(dim=0)
        else:
            batch_size = 1
            state = state.unsqueeze(dim=0)
            action = action.unsqueeze(dim=0)

        s_means, s_log_stds, r_means, r_log_stds = self(state, action)
        model_per_sample = torch.randint(0, self.n_models, size=(batch_size,)).tolist()
        
        # reparameterization trick
        next_state_delta = s_means[model_per_sample, torch.arange(batch_size)] + torch.randn_like(s_log_stds[model_per_sample, torch.arange(batch_size)]) * s_log_stds[model_per_sample, torch.arange(batch_size)].exp()
        reward = r_means[model_per_sample, torch.arange(batch_size)] + torch.randn_like(r_log_stds[model_per_sample, torch.arange(batch_size)]) * r_log_stds[model_per_sample, torch.arange(batch_size)].exp()

        next_state = state + next_state_delta

        if not is_batched:
            next_state = next_state.squeeze(dim=0)
            reward = reward.squeeze(dim=0)
        
        return next_state, reward


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        self.mean_out = nn.Sequential(
            make_linear_layer(in_dim=state_dim, out_dim=hidden_dim),
            nn.Tanh(),
            make_linear_layer(in_dim=hidden_dim, out_dim=hidden_dim),
            nn.Tanh(),
            make_linear_layer(in_dim=hidden_dim, out_dim=action_dim, std=1e-2),
            nn.Tanh(),
        )
        self.log_std_out = nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float32)))
        
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_out(state)
        log_std = self.log_std_out.clamp(-20, 2).expand_as(mean)
        return mean, log_std
    
    def get_distribution(self, state: torch.Tensor) -> Normal:
        mean, log_std = self(state)
        std = log_std.exp()
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

        self.env_dynamics_model = EnvDynamicsEnsemble(n_dyn_models, state_dim, action_dim, hidden_dim)
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

    def predict_transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.env_dynamics_model.sample(state, action)
    
    def choose_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_dist = self.actor.get_distribution(state)
        action_logits = action_dist.rsample()
        action = torch.tanh(action_logits)
        log_probs = action_dist.log_prob(action_logits).sum(dim=-1)
        # add tanh correction
        log_probs -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_probs

    def to_device(self, device: torch.device) -> 'MBPOAgent':
        self.device = device
        self.env_dynamics_model = self.env_dynamics_model.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        return self