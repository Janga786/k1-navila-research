"""FAKE parse_env_cfg (offline test harness). Not Isaac."""
from types import SimpleNamespace as NS


def parse_env_cfg(task, num_envs=1):
    return NS(scene=NS(
        terrain=NS(obj_filepath=None, origins=None),
        robot=NS(init_state=NS(rot=(1.0, 0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.55))),
        disk_1=NS(init_state=NS(pos=None)), disk_2=NS(init_state=NS(pos=None)),
        rgb_camera=NS(offset=NS(pos=(0.10, 0.0, 0.25)), width=1280, height=720,
                      spawn=NS(horizontal_aperture=47.71))))
