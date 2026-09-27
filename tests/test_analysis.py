"""The post-hoc analysis must work on any run, including the one-pool --quick run."""

import numpy as np
import pandas as pd

from src.analysis import gap_by_task_preset, paired_tests, retrieval_categories


def _frame(n_pools):
    rows = []
    rng = np.random.default_rng(0)
    for ps in range(n_pools):
        for task in ["furnace", "iron_ingot"]:
            for preset in ["empty", "has_wood"]:
                for w in range(4):
                    seed = 500_000 + 1000 * ps + 10 * w
                    a = int(rng.integers(0, 20)); b = int(rng.integers(0, 5))
                    for cfg, ex in [("A_task_only", a), ("B_state_aware", b)]:
                        rows.append(dict(config=cfg, task_item=task, preset=preset, world_seed=seed, pool_seed=ps,
                                         excess_steps=ex, replans=ex // 5, failed_actions=ex // 6))
    return pd.DataFrame(rows)


def test_paired_tests_single_pool_has_no_pool_interval_but_runs():
    pt = paired_tests(_frame(1))
    assert "diff_pool0" in pt.columns and "diff_pool1" not in pt.columns
    assert pt.n_pools.eq(1).all() and pt.ci95_pools_low.isna().all()
    assert (pt.p_paired_t.between(0, 1)).all()


def test_paired_tests_three_pools_matches_direct_difference():
    df = _frame(3)
    pt = paired_tests(df)
    row = pt[(pt.preset == "all") & (pt.metric == "excess_steps")].iloc[0]
    a = df[df.config == "A_task_only"].excess_steps.mean(); b = df[df.config == "B_state_aware"].excess_steps.mean()
    assert abs(row["diff"] - (a - b)) < 1e-6
    assert row.ci95_pools_low <= row["diff"] <= row.ci95_pools_high


def test_gap_table_has_all_tasks_and_presets():
    gap = gap_by_task_preset(_frame(2))
    assert set(gap.index) == {"furnace", "iron_ingot"} and list(gap.columns) == ["empty", "has_wood"]
