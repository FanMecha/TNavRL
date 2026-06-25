# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Droid robots.

The following configurations are available:

* :obj:`X2_CFG`: X2 humanoid robot

Reference: https://github.com/Droidrobotics/Droid_ros
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
# 构建相对路径（假设要从当前脚本所在目录的父目录出发去找到目标文件）
# current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# relative_path = "assets/x02A_350/x02A_350.usd"
# usd_X02A_350_path = os.path.join(current_dir, relative_path)
# # usd_X02A_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Drooid/x02/x02A.usd"

# 改为一次os.path.dirname（根据实际目录结构调整）
current_dir = os.path.dirname(os.path.abspath(__file__))  # 直接获取当前文件所在目录
# relative_path = "../assets/x02A_350/x02A_350.usd"  # 根据实际位置调整相对路径
usd_X2_path = f"//home/user/CODE/TNavRL/source/isaaclab_assets/isaaclab_assets/robots/X2/x2.usd"

X2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_X2_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # ".*_shoulder_pitch": -0.35,  # -15 degrees
            # ".*_shoulder_roll": 0.32,
            # ".*_shoulder_yaw": -0.32,
            # ".*_elbow": 1.920,  # 110 degrees
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": 0.3,  # 30 degrees
            ".*_knee_pitch": -0.6,  # -60 degrees
            ".*_ankle_pitch": 0.3,  # 33 degrees
            # ".*_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee_pitch"],
            effort_limit={
                ".*_hip_yaw":80,
                ".*_hip_roll":80,
                ".*_hip_pitch":80,
                ".*_knee_pitch":80,
            },
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw": 100.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 150.0,
                ".*_knee_pitch": 200.0,
            },
            damping={
                ".*_hip_yaw": 1.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee_pitch": 5.0,
            },
            armature={
                ".*_hip_yaw": 0.05,
                ".*_hip_roll": 0.05,
                ".*_hip_pitch":0.05,
                ".*_knee_pitch": 0.05,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch",
                              # ".*_ankle_roll"
                              ],
            effort_limit={
                ".*_ankle_pitch": 45.0,
                # ".*_ankle_roll": 20.0
                },
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_pitch": 20.0,
                # ".*_ankle_roll": 20.0
                },
            damping={
                ".*_ankle_pitch": 2.0,
                # ".*_ankle_roll": 2.0,
                },
            armature={
                ".*_ankle_pitch": 0.05,
                # ".*_ankle_roll": 0.05
                },
        ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_shoulder_pitch": 40.0,
        #         ".*_shoulder_roll": 40.0,
        #         ".*_shoulder_yaw": 40.0,
        #         ".*_elbow": 40.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch": 10.0,
        #         ".*_shoulder_roll": 10.0,
        #         ".*_shoulder_yaw": 10.0,
        #         ".*_elbow": 10.0,
        #
        #     },
        # ),
    },
)
"""Configuration for the Droid X2 Humanoid robot."""



X2_MINIMAL_CFG = X2_CFG.copy()
X2_MINIMAL_CFG.spawn.usd_path = usd_X2_path
"""Configuration for the Droid X2 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""
