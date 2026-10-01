"""K1 vision-based Matterport task for NaVILA-Bench.

Mirrors `h1_matterport_vision_cfg.py` but loads the K1 12-DoF locomotion
URDF (no arms / head merged into Trunk) and includes a height-scanner so
the trained `k1_vision` policy can use perceptive obs (NaVILA Table IV
"Vision" row).

The policy observation MUST match the training observation order
exactly. We mirror legged-loco/k1_low_vision_cfg.py:PolicyCfg:
    base_lin_vel | base_ang_vel | projected_gravity | velocity_commands
    | joint_pos | joint_vel | last_action | height_scan
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
# K1 ArticulationCfg (must match legged-loco training config exactly)
# ---------------------------------------------------------------------------

K1_LOCOMOTION_URDF = os.path.expanduser(
    "~/Projects/k1_research/booster/booster_assets/robots/K1/K1_locomotion.urdf"
)

K1_DEFAULT_JOINT_POS = {
    ".*_Hip_Pitch": -0.15,
    ".*_Hip_Roll": 0.0,
    ".*_Hip_Yaw": 0.0,
    ".*_Knee_Pitch": 0.30,
    ".*_Ankle_Pitch": -0.15,
    ".*_Ankle_Roll": 0.0,
}

# Per-joint armature constants copied verbatim from
# booster_train/assets/robots/booster.py — these encode motor-model rotor
# inertia and were used during training. The eval policy was optimized
# under these inertial values; missing them at eval = different effective
# joint dynamics.
ARMATURE_6416 = 0.095625    # Knee_Pitch
ARMATURE_6408 = 0.0478125   # Hip_Pitch
ARMATURE_4315 = 0.0339552   # Hip_Roll
ARMATURE_4310 = 0.0282528   # Hip_Yaw
ARMATURE_ANKLE = 2.0 * ARMATURE_4310  # 0.0565056, ankle pitch/roll

K1_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=K1_LOCOMOTION_URDF,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
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
        pos=(0.0, 0.0, 0.55),
        joint_pos=K1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Actuator params MUST match the policy's TRAINING config. For model_8700
    # (experiment k1_navila, navila_env_cfg.py) that is the Booster-PROVEN deploy
    # config (booster_deploy/tasks/locomotion K1WalkControllerCfg, == the real
    # robot): legs(hips+knees) Kp=100 Kd=2, ankles Kp=50 Kd=1, with the REAL motor
    # effort/velocity limits (E6408 68/14.66, E4315 76/12.57, E4310 38.3/17.59,
    # E6416 112/12.57; ankle=E4310 38.3/17.59). The prior 350/250 gains + 30-60 Nm
    # limits were the OLD v2/v3 policy lineage and made model_8700 over-stiff and
    # torque-clipped (~3.5-5x torque). Fixed 2026-06-08 to proven values.
    #
    # Training uses DelayedImplicitActuatorCfg (booster_train custom class):
    # ImplicitActuator + 0-4-step (0-20 ms) random delay buffer on joint
    # setpoints, used as a domain-randomization technique for sim-to-real.
    # Isaac Lab 1.1 has no equivalent; we use plain ImplicitActuatorCfg
    # here. The policy was trained to be robust to this delay, so feeding
    # un-delayed commands at eval is acceptable — the delay was meant to
    # make the policy robust to latency, not require it.
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
            velocity_limit={
                ".*_Hip_Pitch": 14.66,
                ".*_Hip_Roll": 12.57,
                ".*_Hip_Yaw": 17.59,
                ".*_Knee_Pitch": 12.57,
            },
            stiffness=100.0,
            damping=2.0,
            armature={
                ".*_Hip_Pitch": ARMATURE_6408,
                ".*_Hip_Roll": ARMATURE_4315,
                ".*_Hip_Yaw": ARMATURE_4310,
                ".*_Knee_Pitch": ARMATURE_6416,
            },
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit=38.3,
            velocity_limit=17.59,
            stiffness=50.0,
            damping=1.0,
            armature=ARMATURE_ANKLE,
        ),
    },
)


@configclass
class K1VisionRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 200
    experiment_name = "k1_vision_rough"  # matches legged-loco training
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
# MDP -- same observation order as training (k1_low_vision_cfg.PolicyCfg)
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Minimal eval-only rewards (we are not training inside the benchmark)."""
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
    """K1 12-DoF leg joint position offset action."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Vision policy obs — order MUST match k1_low_vision_cfg PolicyCfg."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
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
    pass  # no curriculum in eval


@configclass
class TerminationsCfg:
    """Relaxed bad_orientation (1.3 rad ~75°) so it only fires for actual
    falls, not normal walking body sway. The 0.8 rad threshold in the
    original eval config was firing mid-stride and killing 61% of episodes
    inside 2 sim-seconds.

    NOTE: We do NOT add base_contact on Trunk here even though training
    uses it. Reason: with merge_fixed_joints=True the K1's Trunk body
    INCLUDES the merged arms/head/hands. In Matterport scenes the K1
    often spawns close to a wall or piece of furniture, and the merged-arm
    geometry clips into the wall → contact force ~1N → base_contact fires
    and env auto-resets in a loop, leaving K1 standing-still while NaVILA
    keeps saying 'move forward 75 cm'. Observed in fix_v1 ep0/1/2 where
    NaVILA emitted 100+ 'move forward' commands but path_length was 0.054m."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.3},
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
    """Matterport scene + K1 + height scanner + cameras."""

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

    # Height scanner -- raycaster casts down from 20m above Trunk onto the
    # matterport mesh. attach_yaw_only=True keeps the grid world-aligned.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/matterport"],
    )

    # Contact sensor on the K1 body (needed by base_contact termination).
    # K1_ARTICULATION_CFG sets `activate_contact_sensors=True` on the URDF
    # spawner, so this works in Isaac Lab 1.1 despite the older
    # k1_matterport_base note.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

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
    # ground-truthed on-robot 2026-06-08 (perception_info.yaml Camera: BoosterMipi;
    # soccer vision.yaml fx=643.9 cx=649.0 @1280x720, distortion "rs d455"; mount
    # pitch_compensation=0.0 -> level). HFOV~90deg / VFOV~58deg, 16:9. The real head
    # (ZED Head_2 chain) is ~0.25m above Trunk origin -> ~0.78m eye at standing.
    # horizontal_aperture = focal(24.0)*width(1280)/fx(643.9) = 47.71 -> HFOV 90deg.
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
class K1MatterportVisionCfg(ManagerBasedRLEnvCfg):

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 200000.0
        self.sim.render_interval = 4
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = 4 * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.viewer.eye = (5, 12, 5)
        self.viewer.lookat = (5, 0, 0.0)
