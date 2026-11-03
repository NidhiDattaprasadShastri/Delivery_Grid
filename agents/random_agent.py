import numpy as np
from env import N_ACTIONS


class RandomAgent:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(0, N_ACTIONS))

    def observe(self, *args, **kwargs):
        pass  # stateless, nothing to learn
