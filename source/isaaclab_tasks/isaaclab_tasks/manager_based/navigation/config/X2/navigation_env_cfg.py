# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torchvision.transforms as transforms
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.X2.flat_env_cfg import X2FlatEnvCfg
LOW_LEVEL_ENV_CFG = X2FlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_base = EventTerm(
        func=mdp.reset_root_state_usd_fixed_height,  # 使用新函数
        mode="reset",
        params={"pose_range": {"x": (-19.0, 19.0),  "y": (-6.0, 6.0), "yaw": (-3.14, 3.14)}, "velocity_range": {}, "fixed_height": 1.0,  "robot_radius": 0.7,  "check_collision": True,  "max_retries": 20},
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"//home/user/CODE/TNavRL/model/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        realsense_depth_image = ObsTerm(func=mdp.depth_image, params={"camera_name": "realsense", "near": 0.1, "far": 5.0}, noise=Unoise(n_min=-0.1, n_max=0.1))
        goal_point = ObsTerm(func=mdp.goal_point, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_xy = ObsTerm(func=mdp.base_lin_xy, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_z = ObsTerm(func=mdp.base_ang_z, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    collision_penalty = RewTerm(func=mdp.illegal_contact,
                                weight=-100.0,
                                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis_link",  "L_hip_roll_link", "L_hip_pitch_link", "L_knee_pitch_link", "R_hip_roll_link", "R_hip_pitch_link", "R_knee_pitch_link",
                                                                                               "L_shoulder_yaw_link", "L_shoulder_pitch_link", "L_elbow_pitch_link", "L_wrist_yaw_link", "R_shoulder_yaw_link", "R_shoulder_pitch_link", "R_elbow_pitch_link", "R_wrist_yaw_link"]), "threshold": 0.1})

    position_reward = RewTerm(func=mdp.position_reward,
                              weight=10.0,
                              params={"command_name": "pose_command"})

    reaching_reward = RewTerm(func=mdp.reaching_reward,
                              weight=100.0,
                              params={"dis_threshold": 0.15, "command_name": "pose_command", "yaw_threshold": math.radians(15)})

    heading_reward = RewTerm(func=mdp.heading_reward,
                             weight=0.1,
                             params={"command_name": "pose_command"})

    vx_penalty = RewTerm(func=mdp.vx_penalty,
                         weight=-1.0,
                         params={"min_speed": 0, "max_speed": 1.0})

    vy_penalty = RewTerm(func=mdp.vy_penalty,
                         weight=-0.5,
                         params={"min_speed": -0.01, "max_speed": 0.01})

    vz_penalty = RewTerm(func=mdp.vz_penalty,
                         weight=-0.5,
                         params={"min_speed": -0.5, "max_speed": 0.5})

    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.2)

    # this reward's params need to modify
    # forward_turn_exclusive_penalty = RewTerm(func=mdp.forward_turn_exclusive_penalty, weight=-0.5)

    # joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-8, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee_pitch','L_ankle_pitch', 'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee_pitch', 'R_ankle_pitch'])})


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(360, 360),
        debug_vis=True,
        check_collision=True,
        collision_radius=0.6,
        max_retries=20,
        center=(0.0, 0.0),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-20.0, 20.0), pos_y=(-6.0, 6.0), pos_z=(1.0, 1.0), heading=(-math.pi, math.pi)),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(func=mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis_link",  "L_hip_roll_link", "L_hip_pitch_link", "L_knee_pitch_link", "R_hip_roll_link", "R_hip_pitch_link", "R_knee_pitch_link",
                                                                                               "L_shoulder_yaw_link", "L_shoulder_pitch_link", "L_elbow_pitch_link", "L_wrist_yaw_link", "R_shoulder_yaw_link", "R_shoulder_pitch_link", "R_elbow_pitch_link", "R_wrist_yaw_link"]), "threshold": 0.1},)

    reached_goal = DoneTerm(func=mdp.reaching_done, params={"dis_threshold": 0.15,  "yaw_threshold": math.radians(15), "command_name": "pose_command"})

@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            self.sim.physx.gpu_collision_stack_size = 536_870_912
            self.sim.physx.gpu_found_lost_pairs_capacity = 67_108_864
            self.sim.physx.gpu_temp_buffer_capacity = 67_108_864
            self.sim.physx.gpu_heap_capacity = 67_108_864


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
