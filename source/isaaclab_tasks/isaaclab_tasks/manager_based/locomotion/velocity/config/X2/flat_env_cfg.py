# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .rough_env_cfg import X2RoughEnvCfg
import isaaclab.sim as sim_utils

@configclass
class X2FlatEnvCfg(X2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = "model/office1.usd"

        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(42, 13.2))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.001))



        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.projected_gravity = None
        self.observations.policy.history_length = 15
        self.observations.PrivilegedCfg.history_length = 3

        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # self.events.push_robot = None

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.dof_torques_l2.weight = -3.5e-6  # 对施加在关节上的扭矩进行惩罚
        self.rewards.dof_pos_limits.weight = -2
        self.rewards.dof_pos_default_pitch.weight = -0.1
        self.rewards.dof_pos_default_yaw.weight = -0.5
        self.rewards.dof_pos_default_roll.weight = -0.3
        # self.rewards.dof_pos_default_ankle_roll.weight = -0.1
        self.rewards.base_height_l2.weight = -0.
        self.rewards.feet_too_near.params["threshold"] = 0.22
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.weight = 0.15
        # self.rewards.flat_orientatiom_euler_grav.weight = -1.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (-1, 1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1, 1)




class X2FlatEnvCfg_PLAY(X2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

