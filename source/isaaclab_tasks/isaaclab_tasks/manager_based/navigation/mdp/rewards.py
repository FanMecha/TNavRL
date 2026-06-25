# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
from isaaclab.utils.math import wrap_to_pi
from isaaclab.managers import SceneEntityCfg
import os
import csv
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """ encourage robot closer to goal point"""
    goal = env.command_manager.get_command(command_name)[:, :2]
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(pos - goal, dim=1)
    if not hasattr(env, "last_distance") or env.last_distance.shape != distance.shape:
        env.last_distance = distance.clone()
    reset_mask = getattr(env, "reset_buf", torch.zeros_like(distance, dtype=torch.bool)).bool()
    keep_mask = ~reset_mask
    reward = torch.zeros_like(distance)
    reward[keep_mask] = env.last_distance[keep_mask] - distance[keep_mask]
    env.last_distance = distance.clone()
    return torch.clamp(reward, min=0)


def reaching_done(env: ManagerBasedRLEnv, dis_threshold: float, command_name: str, yaw_threshold: float) -> torch.Tensor:
    """bool: once robot closer to goal, terminate envs"""
    goal_pos_w = env.command_manager.get_command(command_name)[:, :2]
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(robot_pos_w - goal_pos_w, dim=1)

    goal_heading_w = env.command_manager.get_command(command_name)[:, 3]
    robot_heading_w = env.scene["robot"].data.heading_w
    heading_error = wrap_to_pi(goal_heading_w - robot_heading_w)

    return (distance < dis_threshold) & (torch.abs(heading_error) < yaw_threshold)


def reaching_reward(env: ManagerBasedRLEnv, dis_threshold: float, command_name: str, yaw_threshold: float) -> torch.Tensor:
    """rewards: encourage robot closer to goal point"""
    return reaching_done(env, dis_threshold=dis_threshold, command_name=command_name, yaw_threshold=yaw_threshold).float()


def heading_reward(env: ManagerBasedRLEnv, command_name: str):
    goal = env.command_manager.get_command(command_name)[:, :2]
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal_yaw = torch.atan2(goal[:, 1]-pos[:, 1], goal[:, 0]-pos[:, 0])
    robot_yaw = env.scene["robot"].data.heading_w
    err = (goal_yaw - robot_yaw + math.pi) % (2*math.pi) - math.pi
    reward = torch.clamp(torch.cos(err), min=0.0)
    return torch.clamp(reward, min=0)


def vx_penalty(env: ManagerBasedRLEnv, min_speed: float, max_speed: float):
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    reward = torch.where((vx <= max_speed) & (vx >= min_speed), 0, torch.abs(vx))
    return reward


def vy_penalty(env: ManagerBasedRLEnv, min_speed: float, max_speed: float):
    vy = env.scene["robot"].data.root_lin_vel_b[:, 1]
    reward = torch.where((vy <= max_speed) & (vy >= min_speed), 0, torch.abs(vy))
    return reward


def vz_penalty(env: ManagerBasedRLEnv, min_speed: float, max_speed: float):
    vz = env.scene["robot"].data.root_ang_vel_b[:, 2]
    reward = torch.where((vz <= max_speed) & (vz >= min_speed), 0, torch.abs(vz))
    return reward


def action_rate_l1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L1 kernel."""
    return torch.sum(torch.abs(env.action_manager.action - env.action_manager.prev_action), dim=1)


def forward_turn_exclusive_penalty(env):
    vx = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0])
    wz = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])
    move_mask = vx > 0.1
    turn_mask = wz > 0.1
    return (move_mask & turn_mask).float()