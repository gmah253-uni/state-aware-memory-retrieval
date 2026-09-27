"""
Post-hoc analysis of the logged episodes in results/*.csv.

    python -m src.analysis

Everything here is derived from the logs plus a deterministic re-run of the
FIRST retrieval of every logged episode (the logs record only which preset the
retrieved experience came from, not which task or experience). Outputs:

  results/paired_tests.csv          A vs B on per-episode differences (the two
                                    configurations run the identical episodes):
                                    paired t-test, Wilcoxon signed-rank, the mean
                                    difference in each pool, and a 95% CI over the
                                    60 distinct worlds (a world seed fixes the grid
                                    for every task and preset, so episodes that share
                                    a seed are not independent)
  results/gap_by_task_preset.csv    mean A-B difference in excess steps for every
                                    (task, preset) pair
  results/first_retrievals.csv      the first retrieval of every logged episode of
                                    every configuration: retrieved task, preset,
                                    phrasing, scores, exact-phrasing flag
  results/retrieval_categories.csv  those retrievals grouped by same/other task and
                                    same/other preset with mean outcomes per group
  results/phrasing_similarity.csv   TF-IDF cosine between every pair of task phrasings
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.env import CraftWorld
from src.memory import CONFIGS, build_pool
from src.oracle import TASK_PHRASES, make_episode

RESULTS = Path("results")
KEY = ["task_item", "preset", "world_seed", "pool_seed"]
PRESET_ORDER = ["empty", "has_wood", "has_planks", "has_wood_pick", "has_stone_pick", "has_furnace"]
METRICS = ["excess_steps", "replans", "failed_actions"]


# ---------------------------------------------------------------------------
# Paired A vs B statistics
# ---------------------------------------------------------------------------

def paired_tests(main: pd.DataFrame, a: str = "A_task_only", b: str = "B_state_aware") -> pd.DataFrame:
    A = main[main.config == a].set_index(KEY).sort_index()
    B = main[main.config == b].set_index(KEY).sort_index()
    assert A.index.equals(B.index), "A and B must cover the same episodes"
    rows = []
    groups = [("all", A.index)] + [(p, A.index[A.index.get_level_values("preset") == p])
                                   for p in PRESET_ORDER if p in A.index.get_level_values("preset")]
    for label, idx in groups:
        for m in METRICS:
            x, y = A.loc[idx, m].to_numpy(float), B.loc[idx, m].to_numpy(float)
            d = x - y
            t, p_t = stats.ttest_rel(x, y)
            try:
                w, p_w = stats.wilcoxon(x, y, zero_method="wilcox")
            except ValueError:          # all differences zero
                w, p_w = np.nan, np.nan
            frame = pd.DataFrame({"d": d, "pool": idx.get_level_values("pool_seed"),
                                  "world": idx.get_level_values("world_seed")})
            by_pool = frame.groupby("pool").d.mean()
            by_world = frame.groupby("world").d.mean()
            se = by_world.std(ddof=1) / np.sqrt(len(by_world))
            rows.append(dict(preset=label, metric=m, n=len(d), mean_A=x.mean(), mean_B=y.mean(),
                             diff=d.mean(), share_B_better=np.mean(d > 0), share_equal=np.mean(d == 0),
                             share_A_better=np.mean(d < 0), paired_t=t, p_paired_t=p_t,
                             wilcoxon_W=w, p_wilcoxon=p_w,
                             **{f"diff_pool{int(k)}": v for k, v in by_pool.items()},
                             n_worlds=len(by_world), diff_worlds_mean=by_world.mean(),
                             ci95_low=by_world.mean() - 1.96 * se, ci95_high=by_world.mean() + 1.96 * se))
    return pd.DataFrame(rows).round(4)


def gap_by_task_preset(main: pd.DataFrame) -> pd.DataFrame:
    A = main[main.config == "A_task_only"].set_index(KEY).sort_index()
    B = main[main.config == "B_state_aware"].set_index(KEY).sort_index()
    d = (A.excess_steps - B.excess_steps).reset_index()
    pv = d.pivot_table(index="task_item", columns="preset", values="excess_steps", aggfunc="mean")
    return pv.reindex(columns=[p for p in PRESET_ORDER if p in pv.columns]).round(2)


# ---------------------------------------------------------------------------
# What did each episode retrieve first?
# ---------------------------------------------------------------------------

def first_retrievals(main: pd.DataFrame) -> pd.DataFrame:
    pools = {}
    rows = []
    for _, r in main.iterrows():
        ps, npc = int(r.pool_seed), int(r.n_per_combo)
        if (ps, npc) not in pools:
            pools[(ps, npc)] = build_pool(n_per_combo=npc, seed=ps)
        pool = pools[(ps, npc)]
        seed = int(r.world_seed)
        env = CraftWorld()
        text = make_episode(env, r.task_item, r.preset, seed=seed, rng=np.random.default_rng(seed))
        top = pool.retrieve(text, env.get_state(), CONFIGS[r.config], k=1,
                            rng=np.random.default_rng(seed + 1))[0]
        e = top.experience
        rows.append(dict(config=r.config, task_item=r.task_item, preset=r.preset, world_seed=seed,
                         pool_seed=ps, task_text=text, retrieved_task=e.task_item,
                         retrieved_preset=e.preset, retrieved_text=e.task_text,
                         retrieved_id=e.id, s_task=round(top.s_task, 4), s_state=round(top.s_state, 4),
                         exact_phrase=e.task_text == text, same_task=e.task_item == r.task_item,
                         same_preset=e.preset == r.preset, excess_steps=r.excess_steps,
                         replans=r.replans, failed_actions=r.failed_actions,
                         oracle_fallback=bool(r.oracle_fallback)))
    return pd.DataFrame(rows)


def retrieval_categories(fr: pd.DataFrame) -> pd.DataFrame:
    cat = np.where(fr.same_task & fr.same_preset, "same task, same preset",
          np.where(fr.same_task & ~fr.same_preset, "same task, other preset",
          np.where(~fr.same_task & fr.same_preset, "other task, same preset", "other task, other preset")))
    fr = fr.assign(category=cat)
    g = fr.groupby(["config", "category"])
    out = g.agg(n=("excess_steps", "size"), excess_steps=("excess_steps", "mean"),
                replans=("replans", "mean"), failed_actions=("failed_actions", "mean"),
                oracle_fallback=("oracle_fallback", "mean"), exact_phrase=("exact_phrase", "mean"))
    out["share"] = out.n / out.groupby(level=0).n.transform("sum")
    return out.reset_index().round(4)


def phrasing_similarity() -> pd.DataFrame:
    tf = build_pool(n_per_combo=4, seed=0)._index()
    phr = [(t, p) for t, ps in TASK_PHRASES.items() for p in ps]
    rows = [dict(phrase_a=p1, phrase_b=p2, same_task=t1 == t2,
                 cosine=round(float(tf.vector(p1) @ tf.vector(p2)), 4))
            for (t1, p1), (t2, p2) in itertools.combinations(phr, 2)]
    return pd.DataFrame(rows).sort_values("cosine", ascending=False)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main = pd.read_csv(RESULTS / "main.csv")

    pt = paired_tests(main)
    pt.to_csv(RESULTS / "paired_tests.csv", index=False)
    print("=== paired A vs B (excess steps) ===")
    print(pt[pt.metric == "excess_steps"][["preset", "n", "diff", "paired_t", "p_paired_t", "p_wilcoxon",
                                           "diff_pool0", "diff_pool1", "diff_pool2", "ci95_low", "ci95_high"]].to_string(index=False))
    print("max p over all presets and metrics: paired t %.2e, wilcoxon %.2e" % (pt.p_paired_t.max(), pt.p_wilcoxon.max()))

    gap = gap_by_task_preset(main)
    gap.to_csv(RESULTS / "gap_by_task_preset.csv")
    print("\n=== A-B gap in excess steps by task and preset ===")
    print(gap.to_string())

    fr = first_retrievals(main)
    fr.to_csv(RESULTS / "first_retrievals.csv", index=False)
    rc = retrieval_categories(fr)
    rc.to_csv(RESULTS / "retrieval_categories.csv", index=False)
    print("\n=== first retrievals by category ===")
    print(rc[rc.config.isin(["A_task_only", "B_state_aware"])].to_string(index=False))

    sim = phrasing_similarity()
    sim.to_csv(RESULTS / "phrasing_similarity.csv", index=False)
    same, cross = sim[sim.same_task].cosine, sim[~sim.same_task].cosine
    print(f"\nTF-IDF cosine: same-task phrasings {same.min():.2f}-{same.max():.2f}; "
          f"different tasks up to {cross.max():.2f}")
