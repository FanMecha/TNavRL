#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_positional_encoding_3d(
    channels: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute 3D positional encoding for given spatial dimensions.

    This code is adapted from: https://github.com/tatp22/multidim-positional-encoding

    Args:
        channels: Number of channels for the encoding.
        D, H, W: Spatial dimensions (depth, height, width).
        device: Device to create tensor on.
        dtype: Data type for the encoding tensor.

    Returns:
        Positional encoding tensor of shape (1, channels, D, H, W).
    """
    org_channels = channels
    channels = int(math.ceil(channels / 6) * 2)
    if channels % 2:
        channels += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device).float() / channels))

    def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    pos_x = torch.arange(D, device=device, dtype=inv_freq.dtype)
    pos_y = torch.arange(H, device=device, dtype=inv_freq.dtype)
    pos_z = torch.arange(W, device=device, dtype=inv_freq.dtype)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
    emb_y = get_emb(sin_inp_y).unsqueeze(1)
    emb_z = get_emb(sin_inp_z)
    emb = torch.zeros((D, H, W, channels * 3), device=device, dtype=dtype)
    emb[:, :, :, :channels] = emb_x
    emb[:, :, :, channels: 2 * channels] = emb_y
    emb[:, :, :, 2 * channels:] = emb_z

    # Convert from (D, H, W, ch) to (1, ch, D, H, W) format (channel-first with batch dim)
    enc = emb[None, :, :, :, :org_channels]  # (1, D, H, W, org_channels)
    enc = enc.permute(0, 4, 1, 2, 3)  # (1, org_channels, D, H, W)
    return enc


class GoalMLP(nn.Module):
    def __init__(self, hidden=64, out_dim=64, max_range=10.0):
        super().__init__()
        self.max_range = float(max_range)
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, goal_with_yaw: torch.Tensor):
        dx, dy, yaw = goal_with_yaw[:, 0], goal_with_yaw[:, 1], goal_with_yaw[:, 2]
        dist = torch.clamp(torch.sqrt(dx * dx + dy * dy), min=1e-6)
        theta = torch.atan2(dy, dx)
        rel_theta = theta - yaw
        g = torch.stack([
            torch.clamp(dx / self.max_range, -1.0, 1.0),
            torch.clamp(dy / self.max_range, -1.0, 1.0),
            torch.clamp(dist / self.max_range, 0.0, 1.0),
            torch.cos(rel_theta),
            torch.sin(rel_theta),
        ], dim=1)
        return self.net(g)


class VelMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=32, max_lin_speed=2.0):
        super().__init__()
        self.register_buffer("max_lin_speed", torch.tensor(float(max_lin_speed)))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, vel: torch.Tensor):
        v = torch.clamp(vel / self.max_lin_speed, -1.0, 1.0)
        return self.net(v)


class GravMLP(nn.Module):
    def __init__(self, use_cosine_axes=True, hidden=32, out_dim=32):
        super().__init__()
        self.use_cosine_axes = use_cosine_axes
        in_dim = 3 if not use_cosine_axes else 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, g_body_unit: torch.Tensor):
        g = F.normalize(g_body_unit, dim=-1, eps=1e-6)
        if self.use_cosine_axes:
            feat = torch.cat([g, torch.abs(g[:, 2:3])], dim=-1)
            tilt = torch.clamp(torch.sqrt((g[:, 0]**2 + g[:, 1]**2)), 0.0, 1.0).unsqueeze(1)
            x = torch.cat([feat, tilt], dim=-1)
        else:
            x = g
        return self.net(x)  # (B, out_dim)


class ProprioEncoder(nn.Module):
    def __init__(self, goal_out=64, vel_out=32, grav_out=32, token_dim=64, max_range=10.0, max_lin_speed=2.0):
        super().__init__()
        self.goal_mlp = GoalMLP(hidden=goal_out, out_dim=goal_out, max_range=max_range)
        self.vel_mlp = VelMLP(in_dim=3, hidden=vel_out, out_dim=vel_out, max_lin_speed=max_lin_speed)
        self.grav_mlp = nn.Sequential(nn.Linear(3, grav_out), nn.SiLU(inplace=True))
        in_dim = goal_out + vel_out + grav_out
        self.fuse = nn.Sequential(nn.Linear(in_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, token_dim))

    def forward(self, goal, vel, grav):
        g = self.goal_mlp(goal)
        v = self.vel_mlp(vel)
        r = self.grav_mlp(grav)
        x = torch.cat([g, v, r], dim=-1)
        z = self.fuse(x)
        state_token = z.unsqueeze(0)
        return state_token, z


class DepthEncoder(nn.Module):
    def __init__(self, in_frames, H, W, token_dim, base_channels):
        super().__init__()
        self.in_frames = in_frames
        self.H = H
        self.W = W
        self.token_dim = token_dim
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.to_token_dim = nn.Conv2d(base_channels * 4, token_dim, kernel_size=1)

    def forward(self, depth: torch.Tensor):
        B = depth.shape[0]
        x = depth.reshape(B * self.in_frames, 1, self.H, self.W)
        feat = self.frame_encoder(x)
        feat = self.to_token_dim(feat)
        _, C, H_out, W_out = feat.shape
        feat_5d = feat.view(B, self.in_frames, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()
        return feat_5d


class CrossAttentionFuseModule(nn.Module):
    """Cross-Attention Module for combining volumetric features with external embeddings.

    Combines self-attention over volumetric features with cross-attention using
    an external info embedding (proprioceptive information).

    Supports a list of 2D feature maps of varying spatial sizes, which are
    zero-padded, stacked along a new depth dimension, and processed with a 3D
    positional encoding. Padding positions are masked out during attention.

    Args:
        image_dim: Number of channels in the given features.
        info_dim: Dimension of the info embedding.
        num_heads: Number of attention heads.
        spatial_dims: Tuple (D, H, W) for positional encoding dimensions.
    """

    def __init__(
        self,
        image_dim: int,
        info_dim: int,
        num_heads: int,
        spatial_dims: tuple,
    ) -> None:
        super().__init__()
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"

        expand_dim = image_dim * 2
        self.image_dim = image_dim

        # Info projection: 2-layer MLP with ELU
        self.info_proj = nn.Sequential(
            nn.Linear(info_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        # Positional encoding
        D, H, W = spatial_dims
        pos_enc = _compute_positional_encoding_3d(image_dim, D, H, W, torch.device("cpu"), torch.float32)
        self.register_buffer("pos_encoding", pos_enc, persistent=True)

        # Self-attention sub-layer
        self.norm1 = nn.LayerNorm(image_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Feed-forward sub-layer
        self.norm2 = nn.LayerNorm(image_dim)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        # Cross-attention sub-layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, img: Union[torch.Tensor, List[torch.Tensor]], info: torch.Tensor) -> torch.Tensor:
        """Process image features with proprioceptive info.

        Args:
            img: Tensor of shape (B, C, H, W) or (B, C, D, H, W), or list of Tensors [(B, C, H_i, W_i), ...]
            info: Tensor of shape (B, info_dim)

        Returns:
            Tensor of shape (B, image_dim)
        """
        # 1. pad & stack if list, or handle 5D tensor directly
        mask: torch.Tensor | None = None
        if isinstance(img, list):
            views = img
            B, C = views[0].shape[:2]
            H_max = max(v.shape[2] for v in views)
            W_max = max(v.shape[3] for v in views)
            padded, masks = [], []
            for v in views:
                _, _, h, w = v.shape
                pad = (0, W_max - w, 0, H_max - h)
                vp = F.pad(v, pad)
                padded.append(vp)
                m = torch.zeros((B, h, w), dtype=torch.bool, device=v.device)
                masks.append(F.pad(m, pad, value=True))
            feats = torch.stack(padded, dim=2)  # (B,C,D,H_max,W_max)
            mask = torch.stack(masks, dim=1)  # (B,D,H_max,W_max)
        elif img.dim() == 5:
            # Already in (B, C, D, H, W) format
            feats = img
        else:
            # 4D tensor: (B, C, H, W) -> (B, C, 1, H, W)
            feats = img.unsqueeze(2)

        # 2. add 3D pos-embed
        B, C, D, H, W = feats.shape
        feats = feats + self.pos_encoding.to(feats.device, feats.dtype)

        # 3. flatten to (B, N, C)
        x = feats.reshape(B, C, D * H * W).permute(0, 2, 1)

        # 4. build padding mask
        key_mask = mask.view(B, D * H * W) if mask is not None else None

        # 5. self-attention (pre-norm + residual)
        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_mask, need_weights=False)
        x = x + sa

        # 6. feed-forward (pre-norm + residual)
        x = x + self.ffn(self.norm2(x))

        # 7. cross-attention with info query
        q = self.info_proj(info).unsqueeze(1)  # (B,1,C)
        ca, _ = self.cross_attn(q, x, x, key_padding_mask=key_mask, need_weights=False)

        return ca.squeeze(1)  # (B, C)