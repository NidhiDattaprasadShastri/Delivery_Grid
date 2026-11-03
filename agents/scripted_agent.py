from env import UP, DOWN, LEFT, RIGHT, DROPOFF


class ScriptedAgent:

    def __init__(self, env):
        self.env = env
        self._last_crashed_dir = None

    def act(self, obs):
        env = self.env
        if env.holding and env.agent_pos == env.cfg.depot_pos:
            return DROPOFF
        if env.holding:
            target = env.cfg.depot_pos
        else:
            target = env.cfg.package_pos
        return self._greedy_move_toward(env.agent_pos, target, env)

    def _greedy_move_toward(self, pos, target, env):
        r, c = pos
        tr, tc = target
        dr, dc = tr - r, tc - c
        candidates = []
        if abs(dr) >= abs(dc):
            if dr != 0:
                candidates.append(DOWN if dr > 0 else UP)
            if dc != 0:
                candidates.append(RIGHT if dc > 0 else LEFT)
        else:
            if dc != 0:
                candidates.append(RIGHT if dc > 0 else LEFT)
            if dr != 0:
                candidates.append(DOWN if dr > 0 else UP)
        if not candidates:
            candidates = [UP, DOWN, LEFT, RIGHT]

        for a in candidates + [UP, DOWN, LEFT, RIGHT]:
            nr, nc = self._apply(pos, a)
            if env._in_bounds(nr, nc) and env.static_grid[nr, nc] != 1:
                return a
        return candidates[0]

    @staticmethod
    def _apply(pos, action):
        deltas = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
        dr, dc = deltas[action]
        return pos[0] + dr, pos[1] + dc

    def observe(self, *args, **kwargs):
        pass  
