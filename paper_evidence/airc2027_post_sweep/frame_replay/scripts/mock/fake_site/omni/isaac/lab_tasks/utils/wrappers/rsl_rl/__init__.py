"""FAKE RslRlVecEnvWrapper (offline test harness). Not Isaac."""


class RslRlVecEnvWrapper:
    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        extras["observations"] = obs_dict
        return obs_dict["policy"], rew, (terminated | truncated).long(), extras

    def close(self):
        pass
