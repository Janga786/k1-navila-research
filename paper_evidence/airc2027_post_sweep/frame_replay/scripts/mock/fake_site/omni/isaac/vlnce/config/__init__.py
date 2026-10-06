"""FAKE task registration (offline test harness). Not Isaac."""
import gymnasium as gym

gym.register(id="k1_matterport_vision", entry_point="omni.isaac.vlnce.utils:FakeEnv", disable_env_checker=True)
