"""CORRECTED robot model for still images only (copied config; the sweep's k1_matterport_vision_cfg.py is untouched).

Starts from a copy of the evaluated K1_ARTICULATION_CFG (k1_matterport_vision_cfg.py, sweep-time) and changes only
the robot model:
  * URDF K1_locomotion.urdf -> K1_22dof.urdf (the training model of model_14498; head and arm joints revolute,
    so nothing is merged into the Trunk)
  * fix_base=True (still image: the Trunk is held at a given pose)
  * head/arm joints held by PD at the TRAINING pose of model_14498 (params/env.yaml of run 2026-06-10_14-30-20):
    Shoulder_Pitch 0.2, Shoulder_Roll -1.25 (L) / +1.25 (R), Elbow_Pitch 0, Elbow_Yaw -0.5 (L) / +0.5 (R), head 0;
    actuators as in training: arms stiffness 20, damping 2, effort 14, velocity 33.51, armature 0.001;
    head stiffness 4, damping 1, effort 6, velocity 7.85, armature 0.001.
Leg joints, gains, limits and armatures are unchanged from the evaluated config.
"""
import os

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.vlnce.config.k1.k1_matterport_vision_cfg import K1_ARTICULATION_CFG, K1_DEFAULT_JOINT_POS

K1_22DOF_URDF = os.path.expanduser("~/Projects/k1_research/booster/booster_assets/robots/K1/K1_22dof.urdf")

TRAINING_UPPER_BODY_POSE = {
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.25,
    "Right_Shoulder_Roll": 1.25,
    ".*_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Right_Elbow_Yaw": 0.5,
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
}


def make_corrected_robot_cfg(root_pos, root_rot_wxyz):
    cfg = K1_ARTICULATION_CFG.copy()
    cfg.spawn = cfg.spawn.replace(asset_path=K1_22DOF_URDF, fix_base=True)
    cfg.init_state = cfg.init_state.replace(pos=tuple(root_pos), rot=tuple(root_rot_wxyz),
                                            joint_pos={**K1_DEFAULT_JOINT_POS, **TRAINING_UPPER_BODY_POSE})
    cfg.actuators = dict(cfg.actuators)
    cfg.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
        effort_limit=14.0, velocity_limit=33.51, stiffness=20.0, damping=2.0, armature=0.001)
    cfg.actuators["head"] = ImplicitActuatorCfg(
        joint_names_expr=[".*Head.*"], effort_limit=6.0, velocity_limit=7.85, stiffness=4.0, damping=1.0, armature=0.001)
    return cfg
