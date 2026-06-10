"""Booster K1 humanoid task config for the NaVILA-Bench (VLN-CE-Isaac).

Mirrors `h1_matterport_base_cfg.py` but loads the K1 12-DoF locomotion URDF
from booster_assets and mounts the head camera at the K1's head height
(~0.85 m above trunk origin).

Status (autonomous setup, 2026-05-15):
* This config loads K1 in the Matterport scene and exposes the full
  observation / camera plumbing the benchmark needs.
* The shipped trained K1 velocity policy (booster_train) uses a
  47-dim obs with gait_phase + 5-frame history that Isaac Lab 1.1's
  ObservationGroupCfg cannot replicate directly. A benchmark-style
  K1 policy needs to be trained separately; until then the
  ``k1_base`` rsl_rl checkpoint dir contains a zero-action stub
  (`scripts/zero_policy_to_rslrl.py`) so the rest of the pipeline
  (NaVILA -> velocity command -> Isaac env -> SR/NE/SPL) can be
  exercised end-to-end.
"""

import os
import math

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.matterport.config import MatterportImporterCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns

import omni.isaac.vlnce.vlnce.mdp as mdp
from omni.isaac.vlnce.utils import ASSETS_DIR

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# ---------------------------------------------------------------------------
# K1 URDF -> Isaac Sim Articulation
# ---------------------------------------------------------------------------

K1_LOCOMOTION_URDF = os.path.expanduser(
    "~/Projects/k1_research/booster/booster_assets/robots/K1/K1_locomotion.urdf"
)

# Default standing pose (matches booster_train BOOSTER_K1_LOCOMOTION_CFG).
K1_DEFAULT_JOINT_POS = {
    ".*_Hip_Pitch": -0.15,
    ".*_Hip_Roll": 0.0,
    ".*_Hip_Yaw": 0.0,
    ".*_Knee_Pitch": 0.30,
    ".*_Ankle_Pitch": -0.15,
    ".*_Ankle_Roll": 0.0,
}

K1_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=K1_LOCOMOTION_URDF,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # K1 locomotion URDF default standing height
        joint_pos=K1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Booster-proven gains (legs 100/2, ankles 50/1) + real motor effort limits, matching
    # k1_matterport_vision_cfg. Was 350/10 + 250/5 (old v2/v3 lineage) which made the policy
    # over-stiff and oscillate; NOT on the production path (the vision task runs) but fixed so an
    # ablation/--task=k1_matterport_base run doesn't silently get wrong dynamics (audit rank 13).
    actuators={
        "hips_knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                              ".*_Knee_Pitch"],
            effort_limit={
                ".*_Hip_Pitch": 68.0,
                ".*_Hip_Roll": 76.0,
                ".*_Hip_Yaw": 38.3,
                ".*_Knee_Pitch": 112.0,
            },
            velocity_limit=20.0,
            stiffness=100.0,
            damping=2.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit=38.3,
            velocity_limit=17.59,
            stiffness=50.0,
            damping=1.0,
        ),
    },
)


@configclass
class K1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "k1_base"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        class_name="ActorCritic",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# MDP — mirrors H1 base config except for body-name regexes (K1 names)
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward terms — kept minimal since this env is used for NaVILA
    EVALUATION ONLY (the policy isn't trained inside the benchmark)."""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class ActionsCfg:
    """K1 12-DoF leg joint position offset action (head/arms are fixed in URDF)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraObsCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("rgb_camera"),
                    "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VizCameraObsCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("viz_rgb_camera"),
                    "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    camera_obs: CameraObsCfg = CameraObsCfg()
    viz_camera_obs: VizCameraObsCfg = VizCameraObsCfg()


@configclass
class CurriculumCfg:
    pass  # no terrain curriculum


@configclass
class TerminationsCfg:
    """No contact-based termination — URDF-loaded K1 doesn't expose
    contact reporter API in Isaac Lab 1.1 (`activate_contact_sensors`
    arg isn't on UrdfFileCfg). For benchmark eval the high-level NaVILA
    policy controls episode end via `set_stop_called`; we keep only
    time_out + bad_orientation."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.3},  # was 0.8 (fired mid-stride, killed ~61% of episodes); match vision_cfg (rank 13)
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Matterport scene + K1 robot."""

    terrain = MatterportImporterCfg(
        prim_path="/World/matterport",
        terrain_type="matterport",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        obj_filepath=os.path.join(ASSETS_DIR, "matterport_usd/5q7pvUzZiYa/5q7pvUzZiYa.usd"),
        groundplane=False,
    )

    robot = K1_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (9.1, 3.8, 0.55)
    robot.init_state.rot = (0.7, 0.0, 0.0, 0.0)

    # contact_forces sensor omitted — see TerminationsCfg docstring.

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1000.0),
    )
    disk_1 = AssetBaseCfg(
        prim_path="/World/disk_1",
        spawn=sim_utils.DiskLightCfg(color=(1.0, 1.0, 1.0),
                                      intensity=10000.0, radius=50.0),
    )
    disk_2 = AssetBaseCfg(
        prim_path="/World/disk_2",
        spawn=sim_utils.DiskLightCfg(color=(1.0, 1.0, 1.0),
                                      intensity=10000.0, radius=50.0),
    )
    disk_1.init_state.pos = (0, 0, 2.6)
    disk_2.init_state.pos = (-1, 0, 2.6)

    # K1-ACCURATE camera = the DEPLOYED BoosterMipi head cam (NOT the bench ZED),
    # ground-truthed on-robot 2026-06-08. HFOV~90deg / VFOV~58deg, 16:9, level
    # (pitch_compensation=0.0). With merge_fixed_joints=True the head is collapsed
    # into Trunk; real head is ~0.25m above Trunk origin -> ~0.78m eye at standing.
    # horizontal_aperture = 24.0*1280/643.9(fx) = 47.71 -> HFOV 90deg.
    rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk/rgb_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.25),
                                    rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=47.71),
        width=1280,
        height=720,
        data_types=["rgb", "distance_to_image_plane"],
    )
    viz_rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk/viz_rgb_camera",
        offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 0.85),
                                    rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=100.0),
        width=512,
        height=512,
        data_types=["rgb"],
    )


@configclass
class K1MatterportBaseCfg(ManagerBasedRLEnvCfg):
    """Configuration for K1 velocity-tracking in Matterport scenes."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz policy at decimation=4 -> 200 Hz physics. Matches H1 cfg.
        self.decimation = 4
        self.episode_length_s = 200000.0
        self.sim.render_interval = 4
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        # NaVILA commands are direct yaw rate, not heading.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.viewer.eye = (5, 12, 5)
        self.viewer.lookat = (5, 0, 0.0)
