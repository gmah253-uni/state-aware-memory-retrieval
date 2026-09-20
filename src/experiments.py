"""
Experiment runner. Reproduces every result in the report:

    python -m src.experiments            # everything (a few minutes)
    python -m src.experiments --quick    # small smoke run

Design (see Methodology):
  * Factors: retrieval config x task x initial-inventory preset x test world seed,
    repeated over several independently generated experience pools (pool seed).
  * Test worlds use seeds disjoint from pool worlds, so no test episode is ever
    an exact replay of a stored experience.
  * Trivial (task already satisfied) combinations are skipped.
  * Three experiments:
      main       - all configs in memory.CONFIGS (A vs B and per-component ablation)
      beta_sweep - alpha/beta trade-off with equal component weights
      pool_size  - how the benefit of state-awareness scales with pool size
  * summarise() computes mean/std per config and a Welch t-test + Mann-Whitney U
    for A vs B on excess steps (the primary metric), written to results/summary.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.agent import MemoryAgent
from src.env import CraftWorld
from src.memory import CONFIGS, RetrievalConfig, build_pool
from src.oracle import INVENTORY_PRESETS, PRESETS, TASKS, make_episode
from src.state import StateWeights

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TEST_SEED_BASE = 500_000   # far away from pool world seeds


def _is_trivial(task: str, preset: str) -> bool:
    return INVENTORY_PRESETS[preset].get(task, 0) > 0


def run_grid(configs: Dict[str, RetrievalConfig], pool_seeds: List[int],
             n_test: int, n_per_combo: int, tag: str) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for ps in pool_seeds:
        pool = build_pool(n_per_combo=n_per_combo, seed=ps)
        for name, cfg in configs.items():
            for task in TASKS:
                for preset in PRESETS:
                    if _is_trivial(task, preset):
                        continue
                    for i in range(n_test):
                        seed = TEST_SEED_BASE + 1000 * ps + 10 * i
                        rng = np.random.default_rng(seed)
                        env = CraftWorld()
                        text = make_episode(env, task, preset, seed=seed, rng=rng)
                        agent = MemoryAgent(pool, cfg, rng=np.random.default_rng(seed + 1))
                        r = agent.run_episode(env, text, preset, seed).as_dict()
                        r.update(pool_seed=ps, pool_size=len(pool), n_per_combo=n_per_combo,
                                 alpha=cfg.alpha, beta=cfg.beta, experiment=tag)
                        rows.append(r)
        print(f"  [{tag}] pool seed {ps} done ({len(rows)} episodes, {time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_main(pool_seeds, n_test, n_per_combo):
    df = run_grid(CONFIGS, pool_seeds, n_test, n_per_combo, "main")
    df.to_csv(RESULTS / "main.csv", index=False)
    return df


def exp_beta_sweep(pool_seeds, n_test, n_per_combo):
    betas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    cfgs = {f"beta_{b:.2f}": RetrievalConfig(f"beta_{b:.2f}", alpha=1 - b, beta=b,
                                              weights=StateWeights())
            for b in betas}
    df = run_grid(cfgs, pool_seeds, n_test, n_per_combo, "beta_sweep")
    df.to_csv(RESULTS / "beta_sweep.csv", index=False)
    return df


def exp_pool_size(pool_seeds, n_test):
    frames = []
    for npc in [1, 2, 4, 8]:
        cfgs = {k: CONFIGS[k] for k in ["A_task_only", "B_state_aware"]}
        frames.append(run_grid(cfgs, pool_seeds, n_test, npc, f"pool_size_{npc}"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "pool_size.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

METRICS = ["success", "steps", "excess_steps", "failed_actions", "replans", "oracle_fallback"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config")[METRICS]
    summ = g.mean().round(3)
    summ.columns = [f"{c}_mean" for c in summ.columns]
    sd = g.std().round(3)
    sd.columns = [f"{c}_std" for c in sd.columns]
    out = pd.concat([summ, sd], axis=1)
    out["n"] = g.size()
    return out.reset_index()


def ab_tests(df: pd.DataFrame, metric: str = "excess_steps") -> pd.DataFrame:
    """A vs B, overall and per preset. Welch t-test and Mann-Whitney U."""
    rows = []
    groups = [("all", df)] + [(p, d) for p, d in df.groupby("preset")]
    for label, d in groups:
        a = d[d.config == "A_task_only"][metric].to_numpy()
        b = d[d.config == "B_state_aware"][metric].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(preset=label, metric=metric, n_A=len(a), n_B=len(b),
                         mean_A=a.mean(), mean_B=b.mean(), diff=a.mean() - b.mean(),
                         welch_t=t, p_welch=p_t, mannwhitney_U=u, p_mw=p_u))
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    args = ap.parse_args()

    if args.quick:
        pool_seeds, n_test, npc = [0], 3, 2
    else:
        pool_seeds, n_test, npc = [0, 1, 2], 20, 4

    print("Running main experiment...")
    main = exp_main(pool_seeds, n_test, npc)
    print("\nRunning beta sweep...")
    exp_beta_sweep(pool_seeds, n_test, npc)
    print("\nRunning pool-size sweep...")
    exp_pool_size(pool_seeds, n_test)

    summ = summarise(main)
    summ.to_csv(RESULTS / "summary.csv", index=False)
    tests = pd.concat([ab_tests(main, m) for m in ["excess_steps", "replans", "failed_actions"]])
    tests.to_csv(RESULTS / "ab_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Main experiment: per-config means ===")
    print(summ[["config", "n", "success_mean", "excess_steps_mean", "replans_mean",
                "failed_actions_mean", "oracle_fallback_mean"]].to_string(index=False))
    print("\n=== A vs B, excess steps ===")
    print(tests[tests.metric == "excess_steps"].to_string(index=False))