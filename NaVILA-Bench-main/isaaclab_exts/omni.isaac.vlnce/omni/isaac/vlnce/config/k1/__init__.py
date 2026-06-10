import gymnasium as gym

from .k1_matterport_base_cfg import K1MatterportBaseCfg, K1RoughPPORunnerCfg
from .k1_matterport_vision_cfg import K1MatterportVisionCfg, K1VisionRoughPPORunnerCfg


gym.register(
    id="k1_matterport_base",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1MatterportBaseCfg,
        "rsl_rl_cfg_entry_point": K1RoughPPORunnerCfg,
    },
)

gym.register(
    id="k1_matterport_vision",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1MatterportVisionCfg,
        "rsl_rl_cfg_entry_point": K1VisionRoughPPORunnerCfg,
    },
)
