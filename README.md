# State-Aware Memory Retrieval for Long-Horizon Planning

*Reliability-aware Planning with Optimus-style Memory:
State-Aware Memory Retrieval for Long-Horizon Planning* (G. Mahale, J. Lazarus).

We test whether adding **state similarity** to **task similarity** when retrieving
past experiences from an Optimus-1-style experience pool reduces wasted actions
and replanning in a Minecraft-like crafting environment.

```
R(e) = alpha * S_task(T_curr, T_exp) + beta * S_state(S_curr, S_exp)
S_state = w_i * S_inventory + w_o * S_objects + w_p * S_progress + w_v * S_visual
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce all results and figures

```bash
python scripts/run_all.py          # ~4 minutes on a laptop, CPU only
python scripts/run_all.py --quick  # ~10 s smoke run
pytest                             # sanity tests
```

Outputs: `results/*.csv` (per-episode logs, summary statistics, A-vs-B tests)
and `figures/*.png`.

## Layout

| File | Purpose |
|---|---|
| `src/env.py` | CraftWorld: gridworld with a Minecraft-style tech tree and precondition failures |
| `src/state.py` | The four state-similarity components and their weighted combination (Eq. 2) |
| `src/oracle.py` | Greedy closed-loop planner used to build the experience pool and as replanning fallback |
| `src/memory.py` | Experience pool, TF-IDF task similarity, and retrieval scoring (Eq. 1) |
| `src/agent.py` | Retrieve → replay abstracted plan → replan on failure; per-episode metrics |
| `src/experiments.py` | Main comparison, per-component ablation, beta sweep, pool-size sweep |
| `src/plots.py` | Report figures |
| `tests/` | Property tests for the environment and retrieval |

## Retrieval configurations

| Name | alpha | beta | State weights |
|---|---|---|---|
| `A_task_only` (baseline) | 1.0 | 0.0 | – |
| `B_state_aware` | 0.5 | 0.5 | equal |
| `B_inventory` / `B_progress` / `B_objects` / `B_visual` | 0.5 | 0.5 | single component |

## Metrics (per episode)

`success`, `steps`, `excess_steps` (vs. the oracle from the same start),
`failed_actions` (precondition failures), `replans`, `oracle_fallback`.