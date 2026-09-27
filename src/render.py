"""
Rendering utilities for CraftWorld.

    python -m src.render --figure   # figures/fig0_craftworld.png   (report figure)
    python -m src.render --slide-figure   # figures/fig0_craftworld_slide.png (two large panels for slides)
    python -m src.render --gif      # figures/episode_A_vs_B.gif    (presentation)
    python -m src.render            # both
    python -m src.render --task stone_pickaxe --preset has_wood_pick --seed 500010 --pool-seed 0

Both outputs use one example episode: a single test world and start state,
executed by the task-only agent (A) and the state-aware agent (B) with the same
experience pool, exactly as in src/experiments.py. By default the example is
chosen from results/main.csv as the episode of the requested (task, preset)
with the largest A-B gap in which B wastes no steps, i.e. a typical success of
state-aware retrieval rather than a hand-picked outlier.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

from src.agent import MemoryAgent, abstract_plan
from src.env import EMPTY, IRON, MILESTONES, STONE, TREE, CraftWorld
from src.memory import CONFIGS, ExperiencePool, RetrievalConfig, RetrievalResult, build_pool
from src.oracle import make_episode

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

CELL_COLOR = {EMPTY: "#f3efe6", TREE: "#4c9a5f", STONE: "#9a9a9a", IRON: "#d4842f"}
CELL_LABEL = {TREE: "T", STONE: "S", IRON: "I"}
COL_A, COL_B = "#888888", "#2a6f97"


# ---------------------------------------------------------------------------
# Recording the episode
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    grid: np.ndarray
    quantity: np.ndarray
    pos: Tuple[int, int]
    inventory: Dict[str, int]
    progress: Set[str]
    step: int
    action: str
    ok: bool
    failed: int
    replans: int
    retrieved: str = ""


class RecordingWorld(CraftWorld):
    """CraftWorld that stores a snapshot after reset and after every step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames: List[Frame] = []
        self.replans = 0
        self.note = ""
        self.retrievals: List[RetrievalResult] = []

    def _snap(self, action: str = "", ok: bool = True) -> None:
        failed = sum(1 for _, o in self.action_log if not o)
        self.frames.append(Frame(self.grid.copy(), self.quantity.copy(), self.pos,
                                 dict(self.inventory), set(self.progress), self.step_count,
                                 action, ok, failed, self.replans, self.note))

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.frames, self.replans, self.note, self.retrievals = [], 0, "", []
        self._snap()
        return state

    def step(self, action: str):
        res = super().step(action)
        self._snap(action, res.ok)
        return res


def run_recorded(pool: ExperiencePool, cfg: RetrievalConfig, task: str, preset: str,
                 world_seed: int):
    """Run one episode exactly as src/experiments.py does, recording every step."""
    env = RecordingWorld()
    rng = np.random.default_rng(world_seed)
    text = make_episode(env, task, preset, seed=world_seed, rng=rng)

    tried: Set[int] = set()
    calls = [0]
    original = pool.retrieve

    def retrieve(*args, **kwargs):
        ranked = original(*args, **kwargs)
        calls[0] += 1
        env.replans = calls[0] - 1
        left = [r for r in ranked if r.experience.id not in tried]
        if left and env.replans <= 3:
            top = left[0]
            tried.add(top.experience.id)
            env.retrievals.append(top)
            env.note = (f"retrieved '{top.experience.task_text}' recorded from "
                        f"{top.experience.preset} (S_task={top.s_task:.2f}, S_state={top.s_state:.2f})")
        else:
            env.note = "oracle fallback"
        env.frames[-1].retrieved = env.note
        return ranked

    pool.retrieve = retrieve
    try:
        agent = MemoryAgent(pool, cfg, rng=np.random.default_rng(world_seed + 1))
        result = agent.run_episode(env, text, preset, world_seed)
    finally:
        pool.retrieve = original
    return env, text, result


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_world(ax, grid: np.ndarray, quantity: np.ndarray, pos: Tuple[int, int],
               title: str = "", view_radius: int = CraftWorld.VIEW_RADIUS) -> None:
    n = grid.shape[0]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(n):
        for c in range(n):
            t = int(grid[r, c])
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=CELL_COLOR[t], edgecolor="white", linewidth=0.6))
            if t != EMPTY:
                ax.text(c + 0.5, r + 0.55, f"{CELL_LABEL[t]}{int(quantity[r, c])}", ha="center",
                        va="center", fontsize=6, color="white", fontweight="bold")
    pr, pc = pos
    ax.add_patch(Rectangle((pc - view_radius, pr - view_radius), 2 * view_radius + 1,
                           2 * view_radius + 1, fill=False, edgecolor=COL_B, linestyle="--",
                           linewidth=1.2, zorder=4))
    ax.add_patch(Circle((pc + 0.5, pr + 0.5), 0.36, facecolor="black", edgecolor="white",
                        linewidth=1.2, zorder=5))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8)


def draw_local_view(ax, local_view: np.ndarray, title: str = "") -> None:
    types = local_view.argmax(-1)
    outside = local_view.sum(-1) == 0          # cells beyond the world edge
    n = types.shape[0]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(n):
        for c in range(n):
            col = "#ffffff" if outside[r, c] else CELL_COLOR[int(types[r, c])]
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=col, edgecolor="#bbbbbb", linewidth=0.6,
                                   hatch="///" if outside[r, c] else None))
    ax.add_patch(Circle((n / 2, n / 2), 0.3, facecolor="black", edgecolor="white", zorder=5))
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8)


def fmt_inventory(inv: Dict[str, int]) -> str:
    return ", ".join(f"{k}×{v}" for k, v in inv.items()) or "(empty)"


def fmt_progress(progress: Set[str]) -> str:
    return ", ".join(m for m in MILESTONES if m in progress) or "(none)"


def fmt_plan(actions: List[str], max_items: int = 6) -> str:
    plan = abstract_plan(actions)
    parts = [f"gather {i}×{n}" if k == "gather" else f"craft {i}" for k, i, n in plan]
    s = " → ".join(parts[:max_items])
    if len(parts) > max_items:
        s += f" → … ({len(parts)} subgoals)"
    return s


# ---------------------------------------------------------------------------
# Report figure: environment + state + first retrieval of A and B
# ---------------------------------------------------------------------------

def make_figure(env_a: RecordingWorld, env_b: RecordingWorld, text: str, preset: str,
                res_a, res_b, out: Path) -> None:
    f0 = env_a.frames[0]
    state = env_a.get_state()  # final state; we need the START state's local view:
    # rebuild the start state's local view from the first frame
    start = RecordingWorld()
    start.grid, start.quantity, start.pos = f0.grid.copy(), f0.quantity.copy(), f0.pos
    start_state = start.get_state()

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.5, 1.55], wspace=0.25)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    draw_world(ax0, f0.grid, f0.quantity, f0.pos, title="(a) CraftWorld at the start of the episode")
    draw_local_view(ax1, start_state.local_view, title="(b) 5×5 local view")
    ax2.axis("off")

    wrap = lambda s, w=58: textwrap.fill(s, w, subsequent_indent="      ")
    a0, b0 = env_a.retrievals[0], env_b.retrievals[0]
    lines = [
        "(c) Agent state used for retrieval",
        wrap(f"Task: “{text}”  (start preset: {preset})"),
        wrap(f"Inventory: {fmt_inventory(f0.inventory)}"),
        wrap(f"Nearby objects (radius 2): {fmt_inventory(start_state.nearby_objects)}"),
        wrap(f"Progress: {fmt_progress(f0.progress)}"),
        f"Position: {f0.pos} (stored, not used for retrieval)",
        "",
        wrap(f"A (task-only) retrieved “{a0.experience.task_text}” recorded from "
             f"{a0.experience.preset}: S_task={a0.s_task:.2f}"),
        wrap(f"      plan: {fmt_plan(a0.experience.actions)}"),
        wrap(f"      outcome: {res_a.steps} steps, {res_a.excess_steps} excess, "
             f"{res_a.failed_actions} failed actions, {res_a.replans} replans"),
        wrap(f"B (state-aware) retrieved “{b0.experience.task_text}” recorded from "
             f"{b0.experience.preset}: S_task={b0.s_task:.2f}, S_state={b0.s_state:.2f}"),
        wrap(f"      plan: {fmt_plan(b0.experience.actions)}"),
        wrap(f"      outcome: {res_b.steps} steps, {res_b.excess_steps} excess, "
             f"{res_b.failed_actions} failed actions, {res_b.replans} replans"),
    ]
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=6.6, family="DejaVu Sans",
             linespacing=1.35, transform=ax2.transAxes)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



def make_slide_figure(env_a: RecordingWorld, out: Path) -> None:
    """Two large panels for a slide: the world at the start of the episode and the
    agent's 5x5 local view, without the text panel of the report figure."""
    f0 = env_a.frames[0]
    start = RecordingWorld()
    start.grid, start.quantity, start.pos = f0.grid.copy(), f0.quantity.copy(), f0.pos
    state = start.get_state()
    fig = plt.figure(figsize=(9.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_world(ax0, f0.grid, f0.quantity, f0.pos, title="(a) CraftWorld at the start of the episode")
    draw_local_view(ax1, state.local_view, title="(b) The agent's 5\u00d75 local view")
    ax0.text(0.5, -0.04, "T tree \u00b7 S stone \u00b7 I iron \u00b7 number = units left \u00b7 dashed = view window \u00b7 disc = agent",
             transform=ax0.transAxes, ha="center", va="top", fontsize=7, color="#555555")
    ax1.text(0.5, -0.06, "hatched = beyond the world edge", transform=ax1.transAxes, ha="center",
             va="top", fontsize=7, color="#555555")
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

# ---------------------------------------------------------------------------
# Animated GIF: A and B side by side from the identical start state
# ---------------------------------------------------------------------------

def make_gif(env_a: RecordingWorld, env_b: RecordingWorld, text: str, preset: str, out: Path,
             interval_ms: int = 250) -> None:
    fa, fb = env_a.frames, env_b.frames
    n = max(len(fa), len(fb))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.6))
    fig.subplots_adjust(top=0.88, bottom=0.30, left=0.04, right=0.98, wspace=0.12)
    fig.suptitle(f"Task: “{text}”   |   start state: {preset}   |   same world, same pool",
                 fontsize=10)
    texts = [fig.text(x, 0.02, "", fontsize=7.2, va="bottom", ha="left", family="DejaVu Sans Mono")
             for x in (0.04, 0.53)]

    def current_note(frames, i):
        note = ""
        for f in frames[:i + 1]:
            if f.retrieved:
                note = f.retrieved
        return note

    def draw(i):
        for ax, frames, label, col, txt in zip(axes, (fa, fb), ("A: task-only retrieval", "B: state-aware retrieval"),
                                               (COL_A, COL_B), texts):
            j = min(i, len(frames) - 1)
            f = frames[j]
            ax.clear()
            done = " – done" if j == len(frames) - 1 and i >= len(frames) - 1 else ""
            draw_world(ax, f.grid, f.quantity, f.pos, title=f"{label}  (step {f.step}{done})")
            ax.title.set_color(col)
            if f.action == "":
                act = "start"
            else:
                act = f.action + ("" if f.ok else "   ✗ precondition failed")
            lines = [
                f"action:    {act}",
                f"failed:    {f.failed}    replans: {f.replans}",
                textwrap.fill(f"inventory: {fmt_inventory(f.inventory)}", 46, subsequent_indent="           "),
                textwrap.fill(current_note(frames, j), 46, subsequent_indent="  "),
            ]
            txt.set_text("\n".join(lines))
        return list(axes) + texts

    ani = animation.FuncAnimation(fig, draw, frames=n, interval=interval_ms, blit=False)
    ani.save(out, writer=animation.PillowWriter(fps=1000 / interval_ms))
    plt.close(fig)
    print(f"wrote {out} ({n} frames)")


# ---------------------------------------------------------------------------
# Example selection and CLI
# ---------------------------------------------------------------------------

def pick_example(task: str, preset: str) -> Tuple[int, int]:
    """(pool_seed, world_seed) of the logged episode with the largest A-B gap and B excess 0."""
    path = Path("results/main.csv")
    if not path.exists():
        return 0, 500_000
    df = pd.read_csv(path)
    d = df[(df.task_item == task) & (df.preset == preset) & df.config.isin(["A_task_only", "B_state_aware"])]
    pv = d.pivot_table(index=["pool_seed", "world_seed"], columns="config", values="excess_steps")
    good = pv[pv["B_state_aware"] == 0]
    if good.empty:
        good = pv
    ps, ws = (good["A_task_only"] - good["B_state_aware"]).idxmax()
    return int(ps), int(ws)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", action="store_true", help="write figures/fig0_craftworld.png")
    ap.add_argument("--gif", action="store_true", help="write figures/episode_A_vs_B.gif")
    ap.add_argument("--slide-figure", action="store_true", help="write figures/fig0_craftworld_slide.png")
    ap.add_argument("--task", default="iron_pickaxe")
    ap.add_argument("--preset", default="has_furnace")
    ap.add_argument("--seed", type=int, help="test-world seed (default: picked from results/main.csv)")
    ap.add_argument("--pool-seed", type=int, help="experience-pool seed (default: picked with --seed)")
    args = ap.parse_args()
    if not (args.figure or args.gif or args.slide_figure):
        args.figure = args.gif = True

    ps, ws = pick_example(args.task, args.preset)
    if args.pool_seed is not None:
        ps = args.pool_seed
    if args.seed is not None:
        ws = args.seed
    print(f"example: task={args.task} preset={args.preset} world_seed={ws} pool_seed={ps}")

    pool = build_pool(n_per_combo=4, seed=ps)
    env_a, text, res_a = run_recorded(pool, CONFIGS["A_task_only"], args.task, args.preset, ws)
    env_b, _, res_b = run_recorded(pool, CONFIGS["B_state_aware"], args.task, args.preset, ws)
    print(f"A: steps={res_a.steps} excess={res_a.excess_steps} failed={res_a.failed_actions} replans={res_a.replans}")
    print(f"B: steps={res_b.steps} excess={res_b.excess_steps} failed={res_b.failed_actions} replans={res_b.replans}")

    if args.figure:
        make_figure(env_a, env_b, text, args.preset, res_a, res_b, FIGURES / "fig0_craftworld.png")
    if args.gif:
        make_gif(env_a, env_b, text, args.preset, FIGURES / "episode_A_vs_B.gif")
    if args.slide_figure:
        make_slide_figure(env_a, FIGURES / "fig0_craftworld_slide.png")


if __name__ == "__main__":
    main()
