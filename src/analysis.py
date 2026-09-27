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
  results/worst_case.txt            ranked candidates and action traces of the two worst
                                    state-aware episodes (Sec. IV-D narrates the second)
  results/timing.txt                cost of the state components and of one retrieval
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
            # pool-level interval: the pool is the coarsest unit of dependence (t with n_pools-1 df)
            if len(by_pool) >= 2:
                se_pool = by_pool.std(ddof=1) / np.sqrt(len(by_pool))
                tcrit = stats.t.ppf(0.975, len(by_pool) - 1)
                ci_pool = (by_pool.mean() - tcrit * se_pool, by_pool.mean() + tcrit * se_pool)
            else:
                ci_pool = (np.nan, np.nan)
            rows.append(dict(preset=label, metric=m, n=len(d), mean_A=x.mean(), mean_B=y.mean(),
                             diff=d.mean(), share_B_better=np.mean(d > 0), share_equal=np.mean(d == 0),
                             share_A_better=np.mean(d < 0), paired_t=t, p_paired_t=p_t,
                             wilcoxon_W=w, p_wilcoxon=p_w,
                             **{f"diff_pool{int(k)}": v for k, v in by_pool.items()},
                             n_worlds=len(by_world), diff_worlds_mean=by_world.mean(),
                             ci95_worlds_low=by_world.mean() - 1.96 * se, ci95_worlds_high=by_world.mean() + 1.96 * se,
                             n_pools=len(by_pool), ci95_pools_low=ci_pool[0], ci95_pools_high=ci_pool[1]))
    df = pd.DataFrame(rows)
    # keep p-values at full precision; round everything else
    return df.round({c: 4 for c in df.columns if df[c].dtype.kind == "f" and not c.startswith("p_")})


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
# The narrated worst case of Sec. IV-D, and the cost of a retrieval
# ---------------------------------------------------------------------------

def worst_case(main: pd.DataFrame, config: str = "B_state_aware", k: int = 8, rank: int = 0) -> str:
    """Re-run the logged episode of `config` with the (rank+1)-th largest excess and
    return a text report: the ranked candidates of its first retrieval and the action
    trace. The report narrates rank 1 (an iron-pickaxe episode); rank 0 is an
    iron-ingot episode from the same start that fails in the same way."""
    from src.agent import MemoryAgent, abstract_plan
    r = main[main.config == config].sort_values("excess_steps", ascending=False).iloc[rank]
    ps, seed = int(r.pool_seed), int(r.world_seed)
    pool = build_pool(n_per_combo=int(r.n_per_combo), seed=ps)
    env = CraftWorld()
    text = make_episode(env, r.task_item, r.preset, seed=seed, rng=np.random.default_rng(seed))
    state = env.get_state()
    lines = [f"episode: config={config} task='{text}' preset={r.preset} world_seed={seed} pool_seed={ps}",
             f"logged: steps={r.steps} oracle_steps={r.oracle_steps} excess={r.excess_steps} "
             f"failed={r.failed_actions} replans={r.replans} fallback={r.oracle_fallback}",
             f"start inventory: {state.inventory}; progress: {sorted(state.progress)}; nearby: {state.nearby_objects}",
             "", f"top-{k} candidates of the first retrieval:"]
    for i, c in enumerate(pool.retrieve(text, state, CONFIGS[config], k=k, rng=np.random.default_rng(seed + 1)), 1):
        e = c.experience
        comps = {n: round(v, 2) for n, v in c.components.items()}
        lines.append(f" {i}. score={c.score:.3f} s_task={c.s_task:.2f} s_state={c.s_state:.2f} {comps} "
                     f"'{e.task_text}' from {e.preset}; plan={abstract_plan(e.actions)}")
    # replay with the inventory logged at every retrieval
    env2 = CraftWorld()
    text2 = make_episode(env2, r.task_item, r.preset, seed=seed, rng=np.random.default_rng(seed))
    log = []
    original = pool.retrieve
    def logged_retrieve(*a, **kw):
        log.append((env2.step_count, dict(env2.inventory)))
        return original(*a, **kw)
    pool.retrieve = logged_retrieve
    try:
        res = MemoryAgent(pool, CONFIGS[config], rng=np.random.default_rng(seed + 1)).run_episode(env2, text2, r.preset, seed)
    finally:
        pool.retrieve = original
    lines += ["", f"replay: steps={res.steps} excess={res.excess_steps} failed={res.failed_actions} "
                  f"replans={res.replans} fallback={res.oracle_fallback}",
              "inventory at each retrieval (step, inventory):"] + [f"  {s}: {inv}" for s, inv in log]
    lines += ["non-movement actions (action, ok):",
              "  " + ", ".join(f"{a}{'' if ok else '(FAILED)'}" for a, ok in env2.action_log if a not in ("up", "down", "left", "right"))]
    return "\n".join(lines)


def timing(n: int = 2000) -> str:
    """Wall-clock cost of the state components and of one full retrieval."""
    import time
    from src.state import state_components
    from src.oracle import INVENTORY_PRESETS
    env = CraftWorld()
    s1 = env.reset("iron_pickaxe", seed=1)
    s2 = env.reset("iron_pickaxe", initial_inventory=INVENTORY_PRESETS["has_furnace"], seed=2)
    t0 = time.perf_counter()
    for _ in range(n):
        state_components(s1, s2)
    per_candidate = (time.perf_counter() - t0) / n * 1e6
    pool = build_pool(n_per_combo=4, seed=0)
    text = make_episode(env, "iron_pickaxe", "has_furnace", seed=3)
    state = env.get_state()
    t0 = time.perf_counter()
    for _ in range(200):
        pool.retrieve(text, state, CONFIGS["B_state_aware"], k=1, rng=np.random.default_rng(0))
    per_retrieval = (time.perf_counter() - t0) / 200 * 1e3
    return (f"state components: {per_candidate:.1f} microseconds per candidate (mean of {n} calls)\n"
            f"full retrieval over {len(pool)} experiences: {per_retrieval:.2f} ms")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main = pd.read_csv(RESULTS / "main.csv")

    pt = paired_tests(main)
    pt.to_csv(RESULTS / "paired_tests.csv", index=False)
    print("=== paired A vs B (excess steps) ===")
    pool_cols = [c for c in pt.columns if c.startswith("diff_pool")]
    cols = ["preset", "n", "diff", "paired_t", "p_paired_t", "p_wilcoxon", *pool_cols,
            "ci95_worlds_low", "ci95_worlds_high", "ci95_pools_low", "ci95_pools_high"]
    print(pt[pt.metric == "excess_steps"][cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
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

    wc = "\n\n".join(f"### rank {i + 1} ###\n" + worst_case(main, rank=i) for i in range(2))
    (RESULTS / "worst_case.txt").write_text(wc + "\n")
    print("\n=== two worst state-aware episodes (results/worst_case.txt) ===")
    print("\n".join(wc.splitlines()[:4]))

    tm = timing()
    (RESULTS / "timing.txt").write_text(tm + "\n")
    print("\n" + tm)
