from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

EMPTY, WALL, PACKAGE, DEPOT = 0, 1, 2, 3

UP, DOWN, LEFT, RIGHT, DROPOFF = 0, 1, 2, 3, 4
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
N_ACTIONS = 5


@dataclass
class EnvConfig:
    grid_size: int = 8
    battery_capacity: int = 50
    max_steps: int = 150
    p_wind: float = 0.15         
    step_cost: float = -0.01
    crash_penalty: float = -0.1
    pickup_bonus: float = 1.0
    delivery_bonus: float = 10.0
    battery_death_penalty: float = -5.0
    invalid_dropoff_penalty: float = -0.05
    holding_bonus_per_step: float = 0.4    
    reward_mode: str = "sparse"          
    full_observability: bool = True
    partial_obs_window: int = 3           
    battery_bucket_size: int = 10
    seed: Optional[int] = None
    walls: Tuple[Tuple[int, int], ...] = field(default_factory=lambda: (
        (2, 2), (2, 3), (2, 4), (5, 5), (5, 6), (5, 3), (4, 1),
    ))
    agent_start: Tuple[int, int] = (0, 0)
    package_pos: Tuple[int, int] = (3, 6)
    depot_pos: Tuple[int, int] = (7, 7)


class GridDeliveryEnv:
    def __init__(self, config: EnvConfig = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build_grid()
        self.reset()

    def _build_grid(self):
        n = self.cfg.grid_size
        grid = np.zeros((n, n), dtype=np.int8)
        for (r, c) in self.cfg.walls:
            grid[r, c] = WALL
        grid[self.cfg.package_pos] = PACKAGE
        grid[self.cfg.depot_pos] = DEPOT
        self.static_grid = grid 

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


    def reset(self) -> Any:
        self.agent_pos = tuple(self.cfg.agent_start)
        self.battery = self.cfg.battery_capacity
        self.holding = False
        self.package_delivered = False
        self.package_available = True 
        self.steps = 0
        self.last_crash = False
        self.trajectory = []  
        return self._get_obs()

    def step(self, action: int):
        assert 0 <= action < N_ACTIONS, f"invalid action {action}"
        assert self.battery > 0 or self.agent_pos == self.cfg.depot_pos, \
            "step() called after episode should have ended (battery depleted)"
        self.steps += 1

        reward = self.cfg.step_cost
        crashed = False
        wind_triggered = False
        just_picked_up = False
        delivered_now = False

        executed_action = action
        if action == DROPOFF:
            if self.holding and self.agent_pos == self.cfg.depot_pos:
                self.holding = False
                self.package_delivered = True
                delivered_now = True
                reward += self.cfg.delivery_bonus
            else:
                reward += self.cfg.invalid_dropoff_penalty
        else:
            if self.rng.random() < self.cfg.p_wind:
                executed_action = int(self.rng.integers(0, 4))
                wind_triggered = True

            dr, dc = ACTION_DELTAS[executed_action]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if not self._in_bounds(nr, nc) or self.static_grid[nr, nc] == WALL:
                crashed = True
                nr, nc = self.agent_pos
            self.agent_pos = (nr, nc)
            self.last_crash = crashed
            if crashed:
                reward += self.cfg.crash_penalty

            if (not self.holding and self.package_available
                    and self.agent_pos == self.cfg.package_pos):
                self.holding = True
                self.package_available = False
                just_picked_up = True
                reward += self.cfg.pickup_bonus

        if self.agent_pos == self.cfg.depot_pos:
            self.battery = self.cfg.battery_capacity
        else:
            self.battery = max(0, self.battery - 1)

        if self.cfg.reward_mode == "shaped_holding" and self.holding:
            reward += self.cfg.holding_bonus_per_step

        terminated = self.package_delivered
        truncated = False
        if self.battery <= 0 and self.agent_pos != self.cfg.depot_pos:
            truncated = True
            reward += self.cfg.battery_death_penalty
        if self.steps >= self.cfg.max_steps:
            truncated = True

        info = {
            "crashed": crashed,
            "wind_triggered": wind_triggered,
            "just_picked_up": just_picked_up,
            "delivered_now": delivered_now,
            "battery": self.battery,
            "steps": self.steps,
        }
        obs = self._get_obs()
        self.trajectory.append((action, executed_action, reward, info))
        return obs, reward, terminated, truncated, info

    def _battery_bucket(self) -> int:
        return self.battery // self.cfg.battery_bucket_size

    def _get_obs(self):
        holding_flag = int(self.holding)
        batt = self._battery_bucket()
        if self.cfg.full_observability:
            return (self.agent_pos[0], self.agent_pos[1], holding_flag, batt)
        k = self.cfg.partial_obs_window // 2
        window = []
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
                if not self._in_bounds(r, c):
                    window.append(-1)  # out-of-bounds marker
                elif not self.package_available and (r, c) == self.cfg.package_pos:
                    window.append(EMPTY) 
                else:
                    window.append(int(self.static_grid[r, c]))
        return tuple(window) + (holding_flag, batt)

    def _in_bounds(self, r, c):
        n = self.cfg.grid_size
        return 0 <= r < n and 0 <= c < n

    def render(self) -> str:
        n = self.cfg.grid_size
        rows = []
        for r in range(n):
            row = []
            for c in range(n):
                cell = "."
                if self.static_grid[r, c] == WALL:
                    cell = "#"
                elif (r, c) == self.cfg.depot_pos:
                    cell = "D"
                elif (r, c) == self.cfg.package_pos and self.package_available:
                    cell = "P"
                if (r, c) == self.agent_pos:
                    cell = "A" if not self.holding else "@"
                row.append(cell)
            rows.append(" ".join(row))
        header = f"step={self.steps} battery={self.battery} holding={self.holding} delivered={self.package_delivered}"
        return header + "\n" + "\n".join(rows)
