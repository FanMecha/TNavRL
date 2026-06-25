from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.Transformer import ProprioEncoder, DepthEncoder, CrossAttentionFuseModule


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
            self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            **kwargs,
    ):

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.in_frames = 1
        self.H, self.W = 48, 64
        self.depth_dim = self.in_frames * self.H * self.W
        self.goal_dim = 3
        self.vel_dim = 3
        self.grav_dim = 3
        token_dim = 64
        H_out, W_out = self.H // 8, self.W // 8

        self.proprio_encoder = ProprioEncoder(goal_out=64, vel_out=32, grav_out=32, token_dim=token_dim)
        self.depth_encoder = DepthEncoder(in_frames=self.in_frames, H=self.H, W=self.W, token_dim=token_dim, base_channels=32)
        self.fusion = CrossAttentionFuseModule(image_dim=token_dim, info_dim=token_dim, num_heads=4, spatial_dims=(self.in_frames, H_out, W_out))

        mlp_input_dim_a = self.fusion.image_dim + token_dim
        mlp_input_dim_c = self.fusion.image_dim + token_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        print("ActorCritic parameters:")
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def encode(self, observations: torch.Tensor):
        idx = 0
        depth_flat = observations[:, idx:idx + self.depth_dim]
        idx += self.depth_dim
        goal = observations[:, idx:idx + self.goal_dim]
        idx += self.goal_dim
        vel = observations[:, idx:idx + self.vel_dim]
        idx += self.vel_dim
        grav = observations[:, idx:idx + self.grav_dim]
        idx += self.grav_dim

        _, info_z = self.proprio_encoder(goal, vel, grav)
        depth_5d = self.depth_encoder(depth_flat)
        fused_img = self.fusion(img=depth_5d, info=info_z)
        head_in = torch.cat([fused_img, info_z], dim=-1)
        return head_in

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        input_actor = self.encode(observations)
        self.update_distribution(input_actor)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        input_actor = self.encode(observations)
        actions_mean = self.actor(input_actor)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        input_actor = self.encode(critic_observations)
        value = self.critic(input_actor)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
