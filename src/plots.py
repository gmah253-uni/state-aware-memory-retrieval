"""
Generate report figures from results/*.csv into figures/*.png.

    python -m src.plots

Figures:
  fig1_config_comparison.png  - excess steps & replans per retrieval config (95% CI)
  fig2_by_preset.png          - A vs B excess steps for each initial-state preset
  fig3_beta_sweep.png         - effect of the task/state trade-off beta
  fig4_pool_size.png          - benefit of state-awareness vs pool size
  fig5_distributions.png      - full distribution of excess steps, A vs B
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})

CONFIG_ORDER = ["A_task_only", "B_state_aware", "B_inventory", "B_progress", "B_objects", "B_visual"]
CONFIG_LABEL = {"A_task_only": "A\ntask-only", "B_state_aware": "B\nall (equal)",
                "B_inventory": "B\ninventory", "B_progress": "B\nprogress",
                "B_objects": "B\nobjects", "B_visual": "B\nvisual"}
PRESET_ORDER = ["empty", "has_wood", "has_planks", "has_wood_pick", "has_stone_pick", "has_furnace"]
COL_A, COL_B = "#888888", "#2a6f97"


def mean_ci(x: pd.Series):
    x = x.to_numpy(dtype=float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


# ---------------------------------------------------------------------------

def fig1_config_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for ax, metric, ylabel in [(axes[0], "excess_steps", "Excess steps vs oracle"),
                               (axes[1], "replans", "Replanning events")]:
        means, cis, colors = [], [], []
        for c in CONFIG_ORDER:
            m, ci = mean_ci(df[df.config == c][metric])
            means.append(m); cis.append(ci)
            colors.append(COL_A if c.startswith("A") else COL_B)
        ax.bar(range(len(CONFIG_ORDER)), means, yerr=cis, color=colors, capsize=3)
        ax.set_xticks(range(len(CONFIG_ORDER)))
        ax.set_xticklabels([CONFIG_LABEL[c] for c in CONFIG_ORDER], fontsize=7.5)
        ax.set_ylabel(ylabel)
    _save(fig, "fig1_config_comparison.png")


def fig2_by_preset(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    presets = [p for p in PRESET_ORDER if p in df.preset.unique()]
    x = np.arange(len(presets)); w = 0.38
    for off, cfg, col, lab in [(-w / 2, "A_task_only", COL_A, "A: task-only"),
                               (w / 2, "B_state_aware", COL_B, "B: state-aware")]:
        ms, cis = zip(*[mean_ci(df[(df.config == cfg) & (df.preset == p)].excess_steps) for p in presets])
        ax.bar(x + off, ms, w, yerr=cis, color=col, capsize=2, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(presets, rotation=20)
    ax.set_ylabel("Excess steps vs oracle"); ax.set_xlabel("Initial-state preset")
    ax.legend(frameon=False)
    _save(fig, "fig2_by_preset.png")


def fig3_beta_sweep(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    g = df.groupby("beta")
    for ax, metric, ylabel in [(axes[0], "excess_steps", "Excess steps"),
                               (axes[1], "replans", "Replanning events"),
                               (axes[2], "failed_actions", "Failed (redundant) actions")]:
        betas = sorted(df.beta.unique())
        ms, cis = zip(*[mean_ci(g.get_group(b)[metric]) for b in betas])
        ax.errorbar(betas, ms, yerr=cis, marker="o", color=COL_B, capsize=3)
        ax.set_xlabel(r"$\beta$ (state weight, $\alpha = 1-\beta$)"); ax.set_ylabel(ylabel)
    _save(fig, "fig3_beta_sweep.png")


def fig4_pool_size(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    for cfg, col, lab in [("A_task_only", COL_A, "A: task-only"), ("B_state_aware", COL_B, "B: state-aware")]:
        d = df[df.config == cfg].groupby("pool_size").excess_steps
        sizes = sorted(d.groups)
        ms, cis = zip(*[mean_ci(d.get_group(s)) for s in sizes])
        ax.errorbar(sizes, ms, yerr=cis, marker="o", color=col, capsize=3, label=lab)
    ax.set_xlabel("Experience pool size"); ax.set_ylabel("Excess steps vs oracle")
    ax.legend(frameon=False)
    _save(fig, "fig4_pool_size.png")


def fig5_distributions(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    data = [df[df.config == c].excess_steps for c in ["A_task_only", "B_state_aware"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black"))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["A: task-only", "B: state-aware"])
    for patch, col in zip(bp["boxes"], [COL_A, COL_B]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_ylabel("Excess steps vs oracle")
    _save(fig, "fig5_distributions.png")


if __name__ == "__main__":
    main = pd.read_csv(RESULTS / "main.csv")
    fig1_config_comparison(main)
    fig2_by_preset(main)
    fig5_distributions(main)
    if (RESULTS / "beta_sweep.csv").exists():
        fig3_beta_sweep(pd.read_csv(RESULTS / "beta_sweep.csv"))
    if (RESULTS / "pool_size.csv").exists():
        fig4_pool_size(pd.read_csv(RESULTS / "pool_size.csv"))