# GridDeliveryEnv — a Reinforcement Learning Environment for Testing Reward Design

**Problem statement selected:** Build a Reinforcement Learning Environment (Problem Statement 3), out of the four options offered (agent harness, data collection tool, RL environment, agent/RL research experiment).

---

## Why I chose this problem statement over the other three

All four options were reasonable choices, but I picked the RL Environment track for three concrete reasons:

**1. It is fully self-contained.** The agent-harness and research-experiment options are centered on a language model doing live reasoning and calling tools mid-task — the project's own wording ("tools," "planning," "retries," and a reference to the "Pi agent harness") makes that clear. The data-collection option is centered on capturing real interaction data (teleoperation, gameplay, sensor logs). An RL environment, by contrast, can be entirely designed, coded, trained and evaluated by hand with no dependency on an external AI service, hardware, or dataset. That gave me a project I could fully build, run and verify end-to-end myself, with nothing outside my control that could fail on someone else's server between when I build it and when it gets reviewed.

**2. The option's own text explicitly supports a simple, non-LLM approach.** Problem Statement 3 explicitly lists *"comparing scripted, random, learned, or model-based policies"* as a valid direction — meaning small hand-coded logic and small learned agents (not large language models) are exactly what's expected here. That is not true of the other three options, which assume an AI model is doing the actual "thinking" inside the system.

**3. It let me build toward a specific, well-understood problem: reward hacking.** Problem Statement 3 explicitly names *"reward design or reward hacking"* as a suggested direction. Reward hacking is a well-documented AI safety concept: it's what happens when an AI system finds a way to maximize the score it's given without actually completing the real task the score was meant to represent. Because it's a specific, well-defined phenomenon (rather than a vague, open-ended agent behavior), I could design an experiment that clearly demonstrates it, measure it with real numbers, and describe it precisely in a short technical write-up — rather than producing a fuzzier result that would be harder to characterize confidently in a short document.

---

## What the project actually is, in plain terms

Picture a small, 8×8 grid, like a chessboard. On it:

- A **delivery drone** (the AI agent) starts in a corner.
- One **package** sits somewhere on the board.
- One **depot** (drop-off point) sits somewhere else.
- Some **walls** block direct paths, so the drone has to navigate around them.
- The drone has a **battery** that drains by 1 every move and only recharges back to full while it's sitting on the depot.
- 15% of the time, a random **gust of wind** overrides the drone's intended move with a random direction instead, so the world isn't perfectly predictable.

The drone's job: pick up the package, then carry it to the depot and choose to explicitly drop it off — all before the battery runs out.

I then tested **three different "brains"** for the drone and two different **reward setups** (the scoring rules that tell the drone what's "good"), to see how each brain behaves and whether either reward setup could be gamed.

---

## Project structure

```
grid_delivery/
├── README.md                   ← this file
├── TECHNICAL_NOTE.md            ← write-up of design decisions, results, and takeaways
├── env/
│   ├── __init__.py
│   └── grid_delivery_env.py     ← the environment itself: the grid, rules, rewards, physics
├── agents/
│   ├── __init__.py
│   ├── random_agent.py          ← baseline brain #1: picks moves completely at random
│   ├── scripted_agent.py        ← baseline brain #2: simple hand-written rules
│   └── q_learning_agent.py      ← baseline brain #3: learns through trial and error
├── tests/
│   ├── __init__.py
│   └── test_env_unittest.py     ← automated checks that the environment behaves correctly
├── train.py                     ← trains the Q-learning agent and saves what it learned
├── evaluate.py                  ← runs all three brains and produces the final results
```

---

## How the pieces connect (the flow)

1. **`env/grid_delivery_env.py`** defines the *world* — the grid layout, what happens when the drone tries to move, when it picks up the package, when it's allowed to drop it off, how the battery behaves, when wind kicks in, and what reward number gets handed out after each action. This file doesn't know anything about "agents" or "learning" — it's just the rules of the world.

2. **`agents/*.py`** each define a *brain* that can look at the current situation and decide what to do next:
   - `random_agent.py` — no logic at all, just picks randomly.
   - `scripted_agent.py` — simple hand-written "if this, do that" rules.
   - `q_learning_agent.py` — starts knowing nothing, and gradually learns a strategy through repeated practice, guided only by the reward numbers it receives.

3. **`train.py`** is only needed for the Q-learning brain (the other two don't need practice — they're either random or pre-programmed). It runs the Q-learning agent through thousands of practice episodes in the environment, and saves what it learned into a file (a "Q-table") inside `results/`.

4. **`evaluate.py`** is the final step: it loads all three brains (including the already-trained Q-learning ones) and runs each of them through 200 test episodes, across three different conditions (normal reward, flawed reward, and limited-visibility reward), then reports how well each one did — success rate, average score, average time taken, etc. It also saves one full step-by-step replay of an interesting episode, so the results aren't just numbers — you can actually see what the agent did.

5. **`tests/test_env_unittest.py`** doesn't involve any agent at all — it directly checks that the environment's rules behave correctly (walls block movement, battery drains and recharges correctly, the package can only be picked up once, etc.), independent of whether any agent ever learns anything.

---

## How to run it

### 1. Get the project open
Unzip the project folder, then open it in VS Code: **File → Open Folder** → select the project folder.

### 2. Create and activate a virtual environment
A virtual environment keeps this project's Python packages separate from everything else on your computer.

```bash
python -m venv venv
```

Then activate it — pick the line matching your terminal type:

```bash
# Git Bash / bash:
source venv/Scripts/activate
```

You'll know it worked because your terminal prompt will now start with `(venv)`.

### 3. Install the two required packages

```bash
pip install numpy matplotlib
```

These are the only two external packages this project needs — `numpy` for numerical/array operations and randomness, `matplotlib` for generating the results chart.

### 4. Run the automated tests

```bash
python -m unittest tests.test_env_unittest -v
```

This should print 17 tests, all ending in "ok," confirming the environment's rules work correctly.

### 5. Run the evaluation

```bash
python evaluate.py
```

This reproduces all the results using the already-trained brains included in `results/` — you don't need to retrain anything to see the results. It prints a summary and regenerates `results/results.json`, `results/comparison.png`, and `results/sample_episode_shaped_holding.txt`.

### 6. (Optional) Retrain the Q-learning brains from scratch

Only needed if you want to reproduce the *training* process itself, not just the evaluation:

```bash
python train.py --mode sparse         --obs full    --episodes 2000 --out results/q_sparse_full.pkl
python train.py --mode shaped_holding --obs full    --episodes 2000 --out results/q_shaped_full.pkl
python train.py --mode sparse         --obs partial --episodes 4000 --out results/q_sparse_partial.pkl
```

Then re-run `python evaluate.py` to regenerate the results using the freshly trained brains.

