import numpy as np
from collections import defaultdict
from env import N_ACTIONS


class QLearningAgent:

    def __init__(self, alpha=0.15, gamma=0.97, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay_episodes=800, seed=None):
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = np.random.default_rng(seed)
        self.episode_count = 0

    def act(self, obs, greedy=False):
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        qvals = self.q[obs]
        # break ties randomly instead of always picking the first max
        best = np.flatnonzero(qvals == qvals.max())
        return int(self.rng.choice(best))

    def observe(self, obs, action, reward, next_obs, terminated):
        target = reward
        if not terminated:
            target += self.gamma * self.q[next_obs].max()
        td_error = target - self.q[obs][action]
        self.q[obs][action] += self.alpha * td_error

    def end_episode(self):
        self.episode_count += 1
        frac = min(1.0, self.episode_count / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @property
    def num_states_seen(self):
        return len(self.q)
