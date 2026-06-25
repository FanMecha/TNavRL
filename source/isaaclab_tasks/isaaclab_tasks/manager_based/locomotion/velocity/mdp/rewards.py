# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.
#
#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.
#
#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward
#
#
# def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Reward long steps taken by the feet for bipeds.
#
#     This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
#     a time in the air.
#
#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
#     in_contact = contact_time > 0.0
#     in_mode_time = torch.where(in_contact, contact_time, air_time)
#     single_stance = torch.sum(in_contact.int(), dim=1) == 1
#     reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
#     reward = torch.clamp(reward, max=threshold)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward
#
#
# def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize feet sliding.
#
#     This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
#     norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
#     agent is penalized only when the feet are in contact with the ground.
#     """
#     # Penalize feet sliding
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset = env.scene[asset_cfg.name]
#
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
#     return reward
#
# def feet_too_near_humanoid(
#     env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
# ) -> torch.Tensor:
#     assert len(asset_cfg.body_ids) == 2
#     asset: Articulation = env.scene[asset_cfg.name]
#     feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
#     distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
#     return (threshold - distance).clamp(min=0)
#
# def track_lin_vel_xy_yaw_frame_exp(
#     env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
#     lin_vel_error = torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
#     )
#     return torch.exp(-lin_vel_error / std**2)
#
#
# def track_ang_vel_z_world_exp(
#     env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     return torch.exp(-ang_vel_error / std**2)

def body_orientation_l2(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

def body_orientation_pitch_l2(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :1]), dim=1)

def body_orientation_roll_l2(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, 1:2]), dim=1)

def track_lin_vel_xy_yaw_frame_exp(
    env:ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env:ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# ********************************** waist ***********************************
def waist_roll_step_coord(
    env:ManagerBasedRLEnv, std: float, k: float, sensor_cfg: SceneEntityCfg, asset_cfg=SceneEntityCfg("robot")
)-> torch.Tensor:
    #
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 读取接触脚
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    # 左脚支撑 → stance = +1，右脚支撑 → –1，其余 0
    stance = torch.where(in_contact[:,0] & ~in_contact[:,1], +1.0,   # 右脚腾空，腰 roll ≈ +θ
             torch.where(in_contact[:,1] & ~in_contact[:,0], -1.0, 0.0))
    # 读取腰 roll 角度
    asset: Articulation = env.scene[asset_cfg.name]
    roll = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    desired = k * stance  # 目标 roll = 0.10 rad × stance
    return torch.exp(-((roll - desired)**2) / std**2)

def waist_yaw_heading_alignment(
    env:ManagerBasedRLEnv, command_name: str, std: float, k: float, asset_cfg=SceneEntityCfg("robot")
)-> torch.Tensor:
    """
        Encourage waist-yaw to align (proportionally) with the commanded heading rate
    """
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]   # 期望 ω_z
    yaw_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    # 让腰角度 ≈ ∫ω_cmd·Δt，近似用 proportional：k = 0.25
    desired = k * yaw_cmd
    reward = torch.exp(-((yaw_pos - desired)**2) / std**2)
    return reward

def waist_stability_l2(
    env:ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot")
)-> torch.Tensor:
    """
        保持腰部关节尽量接近中立位置， 防止过度剧烈的腰部运动
        奖励 = Σ(关节角度²) + 0.05 × Σ(关节速度²)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jpos  = asset.data.joint_pos[:, asset_cfg.joint_ids]          # 角度
    jvel  = asset.data.joint_vel[:, asset_cfg.joint_ids]          # 角速度
    reward = torch.sum(jpos**2, dim=1) + 0.05*torch.sum(jvel**2, dim=1)
    return reward



# *********************************** feet ***********************************
def feet_air_time(
    env:ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env:ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_too_near_humanoid(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:

    assert len(asset_cfg.body_ids) == 2
    asset: RigidObject = env.scene[asset_cfg.name]

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_slide(
    env:ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_stumble(
    env:ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2) > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]), dim=1)

def fly(
    env:ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

def feet_clock_vel(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ Reward for the velocity of the feet during the gait cycle """
    stance_mask = (env.leg_phase[:, :] < 0.6).int()
    # print(stance_mask)
    swing_mask = -1 * (1 - stance_mask)
    # stance_mask = -1, swing_mask = 1 (reverse of feet_clock_frc)
    stance_swing_mask = stance_mask + swing_mask
    stance_swing_mask *= -1
    asset = env.scene[asset_cfg.name]
    max_vel = torch.tensor(0.3, device="cuda")
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    normed_vel = torch.min(body_vel.norm(p=2, dim=-1), max_vel) / max_vel
    rew_normed_vel = normed_vel * stance_swing_mask

    # print(stance_swing_mask, normed_vel, rew_normed_vel.mean(dim=1))

    return rew_normed_vel.mean(dim=1)

# ************************************** ZMP *****************************************
def com_projection_penalty(
    env:ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_sensor")
) -> torch.Tensor:
    """惩罚重心投影超出支撑多边形的距离（L2范数）"""
    # 提取机器人基座和足端传感器数据
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 计算重心投影（假设基座位置为重心）
    com_projection = asset.data.root_pos_w[:, :2]  # x-y平面投影

    # 获取支撑多边形顶点（从传感器接触点提取）
    foot_positions = sensor.data.net_pos_w[sensor_cfg.body_ids, :2]  # 足端x-y位置
    support_polygon = foot_positions  # 假设所有接触地面的足端形成支撑多边形

    # 计算投影到支撑多边形的最小距离（简化实现）
    # 注：实际需使用几何库计算点到多边形距离，此处为示例
    distance = torch.min(torch.norm(com_projection.unsqueeze(1) - support_polygon, dim=1)[0])
    return torch.sum(torch.square(distance), dim=1)

# ***************************** joint ***************************
def joint_vel_torque(
    env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # 索引指定关节
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    # 逐关节计算两种奖励分量 -----------------------------------------------------------------
    # 速度>0时的奖励分量（保留每个关节的贡献，不降维）
    reward_vel_positive = torch.abs(applied_torque * joint_vel)  # 形状: (batch_size, num_joints)
    # 速度<=0时的奖励分量（每个关节的扭矩平方）
    reward_vel_non_positive = torch.square(applied_torque)  # 形状: (batch_size, num_joints)

    # 若速度>0的位置用reward_vel_positive，否则用reward_vel_non_positive
    reward = torch.where(joint_vel > 0.05, reward_vel_positive, reward_vel_non_positive)
    return torch.sum(reward, dim=1)  # 汇总所有关节的奖励

# ************************** phase ***************************

# *************************** task *****************************
def stand_still_body_vel(
    env:ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)-> torch.Tensor:
    """
    当指令几乎为零(‖v_cmd‖<lin_thresh 且 |ω_cmd|<ang_thresh)时，
    惩罚机器人根部仍在动的行为。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # “无速度指令”
    command = env.command_manager.get_command(command_name)
    # print(command)
    stand_still = (torch.norm(command[:, :3], dim=1) <= 0.1)
    # 根部实际速度
    lin_xy  = asset.data.root_lin_vel_b[:, :2]        # shape (N,2)
    ang_yaw = asset.data.root_ang_vel_b[:, 2]         # shape (N,)
    speed_err = torch.sum(lin_xy ** 2, dim=1) + 1.0 * ang_yaw ** 2
    # 仅在 no_cmd 时生效
    reward = torch.where(stand_still, speed_err, torch.zeros_like(speed_err))
    return reward

def stand_still_joint_vel(
    env:ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    stand_still = (torch.norm(command[:, :3], dim=1) <= 0.1)
    jnt_vel_err = 0.01 * torch.sum(asset.data.joint_vel**2, dim=1)
    reward = torch.where(stand_still, jnt_vel_err, torch.zeros_like(jnt_vel_err))
    return reward

def stand_still_joint_pos(
    env:ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)->torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    stand_threshold = (torch.norm(command[:, :3], dim=1) <= 0.1)
    r = torch.exp(-torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1))
    rew = torch.where(stand_threshold, r.clone(), torch.zeros_like(r))
    return rew

def base_euler_xyz(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root euler position in the asset's root frame."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    r, p, w = math_utils.euler_xyz_from_quat(quat)
    euler_xyz = torch.stack([r, p, w], dim=-1)
    euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
    # euler_xyz[:, :] *= 0
    return euler_xyz[:, :2]  # only use xy