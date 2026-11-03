import argparse
import pickle
import numpy as np
from env import GridDeliveryEnv, EnvConfig
from agents import QLearningAgent


def train(reward_mode, full_observability, episodes, seed=0, verbose=True):
    cfg = EnvConfig(reward_mode=reward_mode, full_observability=full_observability, seed=seed)
    env = GridDeliveryEnv(cfg)
    agent = QLearningAgent(seed=seed, epsilon_decay_episodes=int(episodes * 0.6))

    episode_returns = []
    episode_delivered = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated)
            obs = next_obs
            ep_return += reward
            done = terminated or truncated
        agent.end_episode()
        episode_returns.append(ep_return)
        episode_delivered.append(1 if env.package_delivered else 0)

        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            recent = episode_returns[-max(1, episodes // 10):]
            recent_delivered = episode_delivered[-max(1, episodes // 10):]
            print(f"  ep {ep+1:5d}/{episodes}  avg_return={np.mean(recent):7.3f}  "
                  f"delivery_rate={np.mean(recent_delivered):.2f}  eps={agent.epsilon:.3f}  "
                  f"states_seen={agent.num_states_seen}")

    return agent, episode_returns, episode_delivered


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sparse", "shaped_holding"], default="sparse")
    ap.add_argument("--obs", choices=["full", "partial"], default="full")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"Training Q-learning | reward_mode={args.mode} | observability={args.obs} | "
          f"episodes={args.episodes}")
    agent, returns, delivered = train(
        reward_mode=args.mode,
        full_observability=(args.obs == "full"),
        episodes=args.episodes,
        seed=args.seed,
    )

    with open(args.out, "wb") as f:
        pickle.dump({
            "q": dict(agent.q),
            "returns": returns,
            "delivered": delivered,
            "config": {"mode": args.mode, "obs": args.obs, "episodes": args.episodes, "seed": args.seed},
        }, f)
    print(f"Saved to {args.out}")
