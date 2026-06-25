from __future__ import annotations
import warnings
import torch.nn as nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnn_memory import MemorySRU
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.Transformer import ProprioEncoder, DepthEncoder, CrossAttentionFuseModule


class ActorCriticRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        init_noise_std=1.0,
        noise_std_type: str = "log",
        rnn_hidden_dim: int = 256,
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticSRU] Warning: got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        if rnn_type != "lstm_sru":
            print(f"[ActorCriticSRU] Warning: rnn_type='{rnn_type}' is ignored. ActorCriticSRU always uses LSTM_SRU.")
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        self.in_frames = 1
        self.H, self.W = 48, 64
        self.depth_dim = self.in_frames * self.H * self.W
        self.goal_dim = 3
        self.vel_dim = 3
        self.grav_dim = 3
        token_dim = 64
        H_out, W_out = self.H // 16, self.W // 16

        self.proprio_encoder = ProprioEncoder(goal_out=64, vel_out=32, grav_out=32, token_dim=token_dim)
        self.depth_encoder = DepthEncoder(in_frames=self.in_frames, H=self.H, W=self.W, token_dim=token_dim, base_channels=32)
        self.fusion = CrossAttentionFuseModule(image_dim=token_dim, info_dim=token_dim, num_heads=4, spatial_dims=(self.in_frames, H_out, W_out))

        mlp_input_dim_a = self.fusion.image_dim + token_dim  # 128
        mlp_input_dim_c = self.fusion.image_dim + token_dim  # 128

        self.memory_a = MemorySRU(input_size=mlp_input_dim_a, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = MemorySRU(input_size=mlp_input_dim_c, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        mlp_input_dim = rnn_hidden_dim

        # ----- actor head -----
        actor_layers = [nn.Linear(mlp_input_dim, actor_hidden_dims[0]), activation_fn]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # ----- critic head -----
        critic_layers = [nn.Linear(mlp_input_dim, critic_hidden_dims[0]), activation_fn]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        # Noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        Normal.set_default_validate_args(False)

        print("ActorCritic parameters:")
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def encode(self, observations):
        leading_dims = observations.shape[:-1]
        observations = observations.reshape(-1, observations.shape[-1])
        idx = 0
        depth_flat = observations[:, idx:idx + self.depth_dim]
        idx += self.depth_dim
        goal = observations[:, idx:idx + self.goal_dim]
        idx += self.goal_dim
        vel = observations[:, idx:idx + self.vel_dim]
        idx += self.vel_dim
        grav = observations[:, idx:idx + self.grav_dim]
        idx += self.grav_dim
        depth_flat = depth_flat.reshape(-1, self.in_frames, self.H, self.W)
        _, info_z = self.proprio_encoder(goal, vel, grav)
        depth_5d = self.depth_encoder(depth_flat)
        fused_img = self.fusion(img=depth_5d, info=info_z)
        head_in = torch.cat([fused_img, info_z], dim=-1)
        return head_in.reshape(*leading_dims, -1)

    def act(self, observations, masks=None, hidden_states=None):
        features = self.encode(observations)  # (T,B,D) or (B,D)
        mem_out = self.memory_a(features, masks, hidden_states)  # (1,T,B,H) or (1,B,H) inside Memory
        mem_out = mem_out.squeeze(0)  # (T,B,H) or (B,H)
        mean = self.actor(mem_out)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, observations):
        features = self.encode(observations)
        mem_out = self.memory_a(features)
        mem_out = mem_out.squeeze(0)
        return self.actor(mem_out)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        features = self.encode(critic_observations)
        mem_out = self.memory_c(features, masks, hidden_states)
        mem_out = mem_out.squeeze(0)
        return self.critic(mem_out)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
