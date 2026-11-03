import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import GridDeliveryEnv, EnvConfig, DROPOFF
from agents import RandomAgent, ScriptedAgent, QLearningAgent

N_EVAL_EPISODES = 200


def load_q_agent(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent(epsilon_start=0.0, epsilon_end=0.0)  
    for k, v in data["q"].items():
        agent.q[k] = v
    return agent, data["config"]


def run_episodes(env, agent, n_episodes, is_q_agent, seed_base=1000):
    returns, delivered, steps_list, crashes_list, battery_deaths = [], [], [], [], []
    for i in range(n_episodes):
        env.seed(seed_base + i)
        obs = env.reset()
        done = False
        ep_return = 0.0
        n_crashes = 0
        while not done:
            if is_q_agent:
                action = agent.act(obs, greedy=True)
            else:
                action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            n_crashes += int(info["crashed"])
            done = terminated or truncated
        returns.append(ep_return)
        delivered.append(int(env.package_delivered))
        steps_list.append(env.steps)
        crashes_list.append(n_crashes)
        battery_deaths.append(int(truncated and not env.package_delivered and env.battery <= 0))
    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "delivery_rate": float(np.mean(delivered)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_crashes_per_ep": float(np.mean(crashes_list)),
        "battery_death_rate": float(np.mean(battery_deaths)),
        "n_episodes": n_episodes,
    }


def record_sample_episode(env, agent, is_q_agent, path, max_render_steps=60):
    obs = env.reset()
    lines = []
    done = False
    step = 0
    while not done and step < max_render_steps:
        lines.append(env.render())
        if is_q_agent:
            action = agent.act(obs, greedy=True)
        else:
            action = agent.act(obs)
        action_name = ["UP", "DOWN", "LEFT", "RIGHT", "DROPOFF"][action]
        obs, reward, terminated, truncated, info = env.step(action)
        lines.append(f">>> action={action_name}  reward={reward:+.3f}  "
                      f"holding={env.holding}  delivered={env.package_delivered}")
        lines.append("")
        done = terminated or truncated
        step += 1
    lines.append(env.render())
    lines.append(f"[episode ended after {env.steps} steps; delivered={env.package_delivered}]")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    results = {}

    cfg = EnvConfig(reward_mode="sparse", full_observability=True, seed=1)
    env = GridDeliveryEnv(cfg)

    random_agent = RandomAgent(seed=1)
    scripted_agent = ScriptedAgent(env)
    q_agent, q_cfg = load_q_agent("results/q_sparse_full.pkl")

    results["full_sparse"] = {
        "random": run_episodes(env, random_agent, N_EVAL_EPISODES, is_q_agent=False),
        "scripted": run_episodes(env, scripted_agent, N_EVAL_EPISODES, is_q_agent=False),
        "q_learning": run_episodes(env, q_agent, N_EVAL_EPISODES, is_q_agent=True),
    }

    cfg2 = EnvConfig(reward_mode="shaped_holding", full_observability=True, seed=2)
    env2 = GridDeliveryEnv(cfg2)
    random_agent2 = RandomAgent(seed=2)
    scripted_agent2 = ScriptedAgent(env2)
    q_agent2, q_cfg2 = load_q_agent("results/q_shaped_full.pkl")

    results["full_shaped_holding"] = {
        "random": run_episodes(env2, random_agent2, N_EVAL_EPISODES, is_q_agent=False),
        "scripted": run_episodes(env2, scripted_agent2, N_EVAL_EPISODES, is_q_agent=False),
        "q_learning": run_episodes(env2, q_agent2, N_EVAL_EPISODES, is_q_agent=True),
    }

    record_sample_episode(env2, q_agent2, is_q_agent=True,
                           path="results/sample_episode_shaped_holding.txt")

    cfg3 = EnvConfig(reward_mode="sparse", full_observability=False, seed=3)
    env3 = GridDeliveryEnv(cfg3)
    random_agent3 = RandomAgent(seed=3)
    q_agent3, q_cfg3 = load_q_agent("results/q_sparse_partial.pkl")
    scripted_agent3 = ScriptedAgent(env3)

    results["partial_sparse"] = {
        "random": run_episodes(env3, random_agent3, N_EVAL_EPISODES, is_q_agent=False),
        "scripted": run_episodes(env3, scripted_agent3, N_EVAL_EPISODES, is_q_agent=False),
        "q_learning": run_episodes(env3, q_agent3, N_EVAL_EPISODES, is_q_agent=True),
    }

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    make_plots(results)
    print("\nWrote results/results.json, results/comparison.png, "
          "results/sample_episode_shaped_holding.txt")


def make_plots(results):
    conditions = ["full_sparse", "full_shaped_holding", "partial_sparse"]
    cond_labels = ["Full obs.\nsparse reward", "Full obs.\nshaped_holding reward\n(reward hacking test)",
                   "Partial obs.\nsparse reward"]
    agent_names = ["random", "scripted", "q_learning"]
    agent_labels = ["Random", "Scripted", "Q-learning"]
    colors = ["#B0B0B0", "#5B9BD5", "#ED7D31"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("delivery_rate", "Delivery success rate", (0, 1.05)),
               ("avg_return", "Average episode return", None),
               ("avg_steps", "Average steps per episode", None)]

    x = np.arange(len(conditions))
    width = 0.25

    for ax_idx, (metric_key, title, ylim) in enumerate(metrics):
        ax = axes[ax_idx]
        for i, agent in enumerate(agent_names):
            vals = [results[c][agent][metric_key] for c in conditions]
            ax.bar(x + (i - 1) * width, vals, width, label=agent_labels[i], color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.axhline(0, color="black", linewidth=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("GridDeliveryEnv: baseline comparison across three conditions\n"
                  "(200 eval episodes per cell, greedy policies)", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
