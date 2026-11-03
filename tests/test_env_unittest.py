import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import GridDeliveryEnv, EnvConfig, UP, DOWN, LEFT, RIGHT, DROPOFF


def make_env(**overrides):
    cfg = EnvConfig(p_wind=0.0, **overrides)  
    return GridDeliveryEnv(cfg)


class TestGridDeliveryEnv(unittest.TestCase):

    def test_reset_state(self):
        env = make_env()
        obs = env.reset()
        self.assertEqual(env.agent_pos, env.cfg.agent_start)
        self.assertEqual(env.battery, env.cfg.battery_capacity)
        self.assertFalse(env.holding)
        self.assertFalse(env.package_delivered)
        self.assertEqual(obs, (0, 0, 0, env.cfg.battery_capacity // env.cfg.battery_bucket_size))

    def test_deterministic_move_no_wind(self):
        env = make_env(agent_start=(1, 1))
        env.reset()
        obs, reward, term, trunc, info = env.step(DOWN)
        self.assertEqual(env.agent_pos, (2, 1))
        self.assertFalse(info["crashed"])

    def test_wall_collision_stays_in_place_and_penalizes(self):
        env = make_env(agent_start=(1, 2))
        env.reset()
        obs, reward, term, trunc, info = env.step(DOWN)
        self.assertTrue(info["crashed"])
        self.assertEqual(env.agent_pos, (1, 2))
        self.assertAlmostEqual(reward, env.cfg.step_cost + env.cfg.crash_penalty)

    def test_out_of_bounds_collision(self):
        env = make_env(agent_start=(0, 0))
        env.reset()
        obs, reward, term, trunc, info = env.step(UP)
        self.assertTrue(info["crashed"])
        self.assertEqual(env.agent_pos, (0, 0))

    def test_pickup_is_automatic_and_one_time(self):
        env = make_env(agent_start=(3, 5), package_pos=(3, 6))
        env.reset()
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.agent_pos, (3, 6))
        self.assertTrue(env.holding)
        self.assertFalse(env.package_available)
        self.assertTrue(info["just_picked_up"])
        self.assertAlmostEqual(reward, env.cfg.step_cost + env.cfg.pickup_bonus)

        obs2, reward2, term2, trunc2, info2 = env.step(LEFT)
        self.assertFalse(info2["just_picked_up"])

    def test_arriving_at_depot_does_not_auto_deliver(self):
        env = make_env(agent_start=(7, 6), depot_pos=(7, 7))
        env.reset()
        env.holding = True
        env.package_available = False
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.agent_pos, (7, 7))
        self.assertFalse(term)
        self.assertFalse(info["delivered_now"])
        self.assertTrue(env.holding) 

    def test_explicit_dropoff_terminates_with_bonus(self):
        env = make_env(agent_start=(7, 7), depot_pos=(7, 7))
        env.reset()
        env.holding = True
        env.package_available = False
        obs, reward, term, trunc, info = env.step(DROPOFF)
        self.assertTrue(term)
        self.assertTrue(info["delivered_now"])
        self.assertFalse(env.holding)
        self.assertAlmostEqual(reward, env.cfg.step_cost + env.cfg.delivery_bonus)

    def test_dropoff_fails_off_depot_or_without_package(self):
        env = make_env(agent_start=(0, 0))
        env.reset()
        obs, reward, term, trunc, info = env.step(DROPOFF)
        self.assertFalse(term)
        self.assertAlmostEqual(reward, env.cfg.step_cost + env.cfg.invalid_dropoff_penalty)

    def test_battery_depletes_and_truncates(self):
        env = make_env(battery_capacity=2, agent_start=(0, 0), max_steps=100)
        env.reset()
        obs, r, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.battery, 1)
        self.assertFalse(trunc)
        obs, r, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.battery, 0)
        self.assertTrue(trunc)
        self.assertAlmostEqual(r, env.cfg.step_cost + env.cfg.battery_death_penalty)

    def test_battery_recharges_at_depot(self):
        env = make_env(agent_start=(7, 6), depot_pos=(7, 7), battery_capacity=10)
        env.reset()
        env.battery = 1
        obs, r, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.agent_pos, (7, 7))
        self.assertEqual(env.battery, env.cfg.battery_capacity)

    def test_max_steps_truncates(self):
        env = make_env(max_steps=3, agent_start=(0, 0))
        env.reset()
        for i in range(3):
            obs, r, term, trunc, info = env.step(RIGHT)
        self.assertTrue(trunc)

    def test_shaped_holding_mode_adds_bonus_each_step_while_holding(self):
        env = make_env(reward_mode="shaped_holding", agent_start=(3, 5), package_pos=(3, 6))
        env.reset()
        env.step(RIGHT)  
        obs, r2, term, trunc, info = env.step(LEFT)
        expected = env.cfg.step_cost + env.cfg.holding_bonus_per_step
        self.assertAlmostEqual(r2, expected)

    def test_sparse_mode_gives_no_holding_bonus(self):
        env = make_env(reward_mode="sparse", agent_start=(3, 5), package_pos=(3, 6))
        env.reset()
        env.step(RIGHT)
        obs, r2, term, trunc, info = env.step(LEFT)
        self.assertAlmostEqual(r2, env.cfg.step_cost)

    def test_wind_can_override_action(self):
        cfg = EnvConfig(p_wind=1.0, agent_start=(4, 4), seed=0)
        env = GridDeliveryEnv(cfg)
        env.reset()
        obs, r, term, trunc, info = env.step(UP)
        self.assertTrue(info["wind_triggered"])

    def test_partial_observability_hides_far_away_package(self):
        cfg = EnvConfig(p_wind=0.0, full_observability=False, partial_obs_window=3,
                         agent_start=(0, 0), package_pos=(6, 6))
        env = GridDeliveryEnv(cfg)
        obs = env.reset()
        window_cells = obs[:9]
        self.assertNotIn(2, window_cells)

    def test_partial_observability_reveals_nearby_package(self):
        cfg = EnvConfig(p_wind=0.0, full_observability=False, partial_obs_window=3,
                         agent_start=(3, 5), package_pos=(3, 6))
        env = GridDeliveryEnv(cfg)
        obs = env.reset()
        window_cells = obs[:9]
        self.assertIn(2, window_cells)

    def test_reproducibility_with_seed(self):
        cfg1 = EnvConfig(seed=42, p_wind=0.3)
        cfg2 = EnvConfig(seed=42, p_wind=0.3)
        env1, env2 = GridDeliveryEnv(cfg1), GridDeliveryEnv(cfg2)
        env1.reset(); env2.reset()
        actions = [RIGHT, DOWN, RIGHT, DOWN, RIGHT, UP, LEFT, DOWN]
        traj1 = [env1.step(a) for a in actions]
        traj2 = [env2.step(a) for a in actions]
        for (o1, r1, t1, tr1, i1), (o2, r2, t2, tr2, i2) in zip(traj1, traj2):
            self.assertEqual(o1, o2)
            self.assertAlmostEqual(r1, r2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
