# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.sensors.camera import CameraCfg
import isaaclab.sim as sim_utils
##
# Pre-defined configs
##
from isaaclab_assets.robots.X2.x2 import X2_MINIMAL_CFG  # isort: skip

intrinsicsMatrix = [617.437, 0.0, 317.525, 0.0, 617.689, 240.767, 0.0, 0.0, 1.0]
@configclass
class X2Events(EventCfg):
    robot_joint_friction_and_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.95, 1.05),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           joint_names=[".*"],
                                           scale=0.5,
                                           use_default_offset=True,
                                           preserve_order=False)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.5, n_max=0.5),scale=1)
        base_euler_xy = ObsTerm(func=mdp.base_euler_xyz,noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=['L_hip_yaw','L_hip_roll', 'L_hip_pitch',
                                                                                      'L_knee_pitch','L_ankle_pitch',
                                                                                      'R_hip_yaw','R_hip_roll','R_hip_pitch',
                                                                                      'R_knee_pitch','R_ankle_pitch'],
                                                                         preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=['L_hip_yaw','L_hip_roll', 'L_hip_pitch',
                                                                                      'L_knee_pitch','L_ankle_pitch',
                                                                                      'R_hip_yaw','R_hip_roll','R_hip_pitch',
                                                                                      'R_knee_pitch','R_ankle_pitch'],
                                                                         preserve_order=True)},scale=1)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 15

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for Privileged information."""

        velocity_commands = ObsTerm(clip=[-18, +18], func=mdp.generated_commands,
                                    params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(clip=[-18, +18], func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=['L_hip_yaw','L_hip_roll', 'L_hip_pitch',
                                                                                      'L_knee_pitch','L_ankle_pitch',
                                                                                      'R_hip_yaw','R_hip_roll','R_hip_pitch',
                                                                                      'R_knee_pitch','R_ankle_pitch'],
                                                                         preserve_order=True)}, scale=1)
        joint_vel = ObsTerm(clip=[-18, +18], func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=['L_hip_yaw','L_hip_roll', 'L_hip_pitch',
                                                                                      'L_knee_pitch','L_ankle_pitch',
                                                                                      'R_hip_yaw','R_hip_roll','R_hip_pitch',
                                                                                      'R_knee_pitch','R_ankle_pitch'],
                                                                         preserve_order=True)}, scale=1)
        actions = ObsTerm(clip=[-18, +18], func=mdp.last_action)
        base_lin_vel = ObsTerm(clip=[-18, +18], func=mdp.base_lin_vel, scale=1)
        base_ang_vel = ObsTerm(clip=[-18, +18], func=mdp.base_ang_vel, scale=1)
        base_euler_xy = ObsTerm(clip=[-18, +18], func=mdp.base_euler_xyz, scale=1)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class X2Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # lin_vel_z_l2 = RewTerm(
    #     func=mdp.lin_vel_z_l2,
    #     weight=-1.0
    # )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch_link"),
            "threshold": 0.65,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch_link"),
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch_link"),
                "threshold": 0.2
                }
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    dof_pos_default_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch", ".*_knee_pitch", ".*_ankle_pitch"])}
    )
    dof_pos_default_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw")}
    )
    dof_pos_default_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll")}
    )
    # dof_pos_default_ankle_roll = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_roll")}
    # )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.75}  # 设置目标高度
    )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    # )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_hip_yaw")}
    # )
    flat_orientation_grav=RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_pitch_link"])}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*pelvis_link"), "threshold": 1.0},
    )
    below_limit = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*pelvis_link"), "minimum_height": 0.5},
    )


@configclass
class X2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    cycle_time: float = 1.0
    rewards: X2Rewards = X2Rewards()
    terminations: TerminationsCfg = TerminationsCfg()
    events: X2Events = X2Events()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = X2_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        self.scene.realsense = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis_link",
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.6), rot=(-0.4, 0.6, -0.6, 0.4)),
            pattern_cfg=patterns.PinholeCameraPatternCfg(width=64, height=48, horizontal_aperture=2.3, focal_length=2.2),
            debug_vis=False,
            max_distance=5,
            mesh_prim_paths=["/World/ground"],
            data_types=["distance_to_image_plane"],
            update_period=1/10,
            history_length=1,
        )

        # self.scene.rgb_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/pelvis_link/rgb_camera",
        #     # offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.04), rot=(-0.2, 0.8, -0.8, 0.2)),
        #     offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.6), rot=(-0.4, 0.6, -0.6, 0.4)),
        #     spawn=sim_utils.PinholeCameraCfg(focal_length=2.2, horizontal_aperture=2.3),
        #     width=32, height=24,
        #     data_types=["rgb"],  # ✅ 用 rgb
        #     update_period=1 / 30.0,
        #     history_length=6,
        # )

        # Randomization
        #self.events.push_robot = None
        #self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*"]

        # self.events.
        # startup
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*pelvis_link"),
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add",
            },
        )

        #  随机化摩擦系数
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_pitch_link"),
                "static_friction_range": (0.1, 2.0),  #  0.8
                "dynamic_friction_range": (0.1, 2.0),
                "restitution_range": (0.0, 0.0),  # 弹性系数
                "num_buckets": 64,
            },
        )

        # interval  间隔
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # reset  重置
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.events.robot_joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "distribution": "uniform",  # normal 正态分布   uniform 均匀分布
            },
        )

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*pelvis_link"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -0.0000
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_pos_default_pitch.weight = -1.0
        self.rewards.dof_pos_default_yaw.weight = -1.0
        self.rewards.dof_pos_default_roll.weight = -1.0
        self.rewards.base_height_l2.weight = -1.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class X2RoughEnvCfg_PLAY(X2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        #self.events.push_robot = None
