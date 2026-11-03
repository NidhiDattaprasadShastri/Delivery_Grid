# Technical Note

## Why this problem

I chose the RL Environment track over the agent-harness, data-collection,
and research-experiment options because it's the one I could build, train,
and fully evaluate end-to-end without an external LLM API, hardware, or a
pre-existing dataset — everything runs offline with NumPy and Matplotlib,
which let me spend the budget on substance rather than plumbing.

I built the environment around a question more interesting to me than
"can a tabular agent solve a gridworld": **can a small, individually
reasonable reward term make an agent worse at the actual task while
giving it a higher score?** The environment itself (delivery drone,
package, depot, battery, wind) is a standard sequential-decision problem;
the real design choice is the `shaped_holding` reward mode, which adds a
per-step bonus for holding the package — meant to look like sensible
shaping ("reward progress toward pickup") but which is not
potential-based, and is therefore exploitable.

## Decisions and tradeoffs

**Explicit DROPOFF action.** My first version made delivery automatic on
arrival at the depot. Once I started training I realized this made the
exploit I wanted to demonstrate *impossible*: if arriving at the depot
always ends the episode, an agent can never sit there farming a per-step
"holding" bonus. I changed delivery to a fifth explicit action, which is
what let the interesting behavior emerge. This is a real example of a
spec bug that only became visible by trying to break my own reward
function, not by inspection.

**Full vs. partial observability, and baseline scope.** The Q-learning
agent supports both (it just hashes whatever tuple the environment hands
it), but the scripted baseline reads privileged environment state
directly and doesn't respect the observability flag. That's an
intentional scope cut — a scripted agent that plans under partial
observability (belief tracking, frontier search) was out of budget — but
it means the scripted-vs-learned comparison under partial observability
isn't apples-to-apples, which I flag rather than paper over.

**Tabular Q-learning, not deep RL.** State spaces here are ~330–490
visited states, so a Q-table converges reliably in a couple thousand
episodes with no function-approximation instability to debug. I'd rather
have a fully-understood, fully-converged tabular result in this budget
than a plausible-but-uncertain deep-RL one.

**Reward-hacking magnitude was itself a tuning problem.** My first
attempt at a hackable reward (+0.05/step for holding) wasn't actually
exploited: under the agent's discount factor (γ=0.97), the infinite
discounted sum of that bonus was still smaller than the one-time delivery
bonus (+10), so the optimal policy correctly delivered immediately. I
raised the bonus to +0.4/step until its discounted value plausibly
exceeded the delivery bonus. That's a finding in its own right: whether a
shaped reward is exploitable isn't just a qualitative property of the
term — it depends on its magnitude relative to the discount factor and
the terminal bonus, which is easy to get wrong in either direction.

## What worked

All three baselines behave as expected under the well-specified (sparse)
reward: Random never delivers (0% success, always battery-truncated);
Scripted reaches 100% delivery in the fewest steps (17.4 avg); Q-learning
also reaches 100% delivery in both full and partial observability
(26.2 and 25.1 avg steps), taking a bit longer than the scripted heuristic
because it's learning the value function rather than following a
hand-coded shortest-path rule, and because it has to route around the
stochastic wind rather than assuming clean control.

Under the misspecified `shaped_holding` reward, the Q-learning agent's
average return jumps from 10.67 to 22.00 (+106%) — while its delivery
rate *drops* slightly, from 100% to 96.5%, and 3.5% of episodes now end
in battery death. Looking at the actual episode replay
(`results/sample_episode_shaped_holding.txt`), the agent doesn't simply
delay delivery — it discovered that calling DROPOFF *away* from the depot
silently fails without moving the agent, and uses that as a free
"stand still and farm the bonus" action: 30+ consecutive steps of
`DROPOFF` at a fixed off-depot cell, each yielding the holding bonus,
before finally delivering near the end of the episode. That's an
emergent exploit I did not design in explicitly — it fell out of the
interaction between the reward term and an action whose failure mode I
hadn't thought to make more costly.

## What didn't work / limitations

The scripted agent's privileged observability limits how much the
partial-observability comparison says about scripted vs. learned
policies specifically under uncertainty. The tabular approach won't
scale past this grid size or state cardinality — the right choice for
this problem's size, but untested for scaling by construction. I also
only tested one misspecified reward (holding bonus); I don't have
evidence on which other shaping classes are similarly exploitable versus
safe — true potential-based shaping (provably policy-invariant) would
have been the natural contrasting condition and next baseline.

## Evaluation method

Numbers come from 200 held-out episodes per (agent, condition) cell, with
fixed distinct seeds per condition and agents run greedily (epsilon=0 for
Q-learning). Metrics: delivery rate, average return, average episode
length, crashes/episode, battery-death rate — in `results/results.json`
and `results/comparison.png`. The 17 unit tests in
`tests/test_env_unittest.py` check transition and reward correctness in
both reward modes, battery mechanics, wind, and observability
independent of any learned agent.

## What I'd build or test next

A true potential-based shaping variant as a control — theory says it
shouldn't be exploitable, and I'd like to confirm that empirically
side-by-side with `shaped_holding`. I'd also extend the scripted baseline
to respect partial observability (e.g. a simple belief-state agent) so
that comparison stops being "learned vs. privileged."

## Closing question: what would I want to explore or build next at Iacon Autonomics?

I'd want to push on exactly the seam this exercise surfaced: **automated
detection of reward misspecification**, rather than relying on a human
noticing it in a replay after the fact. For a gym running many
environments and reward variants, the failure mode I found here — higher
return via an action's *unintended* side effect rather than its intended
one — seems cheap to detect programmatically: flag policies whose return
is high but whose task-success rate is flat or falling, and flag
environments where a "no-op-like" action exists that no reward term
explicitly penalizes. More broadly, I'd want to know whether this
generalizes past gridworlds — whether the same audit (compare return
against an independent success signal, watch for divergence) is a useful
cheap safety net across the harness, data-collection, and RL-environment
tracks this assignment offered, since misspecified reward/success
criteria seems like a recurring risk across all of them, not just RL
environments.