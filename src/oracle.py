"""
Oracle planner for CraftWorld.

Role in the project: the oracle is NOT the agent under test. It is used to
  1. generate the experience pool (successful trajectories from many random
     start states, which the memory module later retrieves from), and
  2. provide the fallback "replanning" behaviour when a retrieved plan fails.

It is a greedy, closed-loop goal decomposer over the tech tree:
  next_action() recurses from the task item down to the deepest missing
  ingredient, then returns ONE action (move / gather / craft). It is called
  again after every step, so it adapts to depleted nodes and partial progress.
This mirrors how Optimus-1's planner re-plans from current state, but with no
LLM in the loop so experiments are fast and fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env import (GATHER, GATHER_ITEMS, RECIPES, TOOL_TIER, CraftWorld,
                     State, TREE, STONE, IRON)

# Inverse lookups
ITEM_TO_CELL = {item: cell for cell, (item, _) in GATHER.items()}
TIER_TO_TOOL = {tier: tool for tool, tier in TOOL_TIER.items()}

FAIL = "__fail__"   # returned when the task is unachievable (resource exhausted)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _move_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> str:
    dr, dc = target[0] - pos[0], target[1] - pos[1]
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def next_action(env: CraftWorld, item: str, qty: int = 1) -> Optional[str]:
    """Return the next single action needed to obtain `qty` of `item`.

    Returns None if the agent already has enough, FAIL if impossible.
    """
    inv = env.inventory
    if inv.get(item, 0) >= qty:
        return None

    if item in RECIPES:
        ingredients, tools, out_qty = RECIPES[item]
        # Tools first (e.g. furnace for iron_ingot)
        for tool in tools:
            a = next_action(env, tool, 1)
            if a is not None:
                return a
        # Then ingredients, scaled by how many crafts are still needed
        crafts_needed = math.ceil((qty - inv.get(item, 0)) / out_qty)
        for ing, n in ingredients.items():
            a = next_action(env, ing, n * crafts_needed)
            if a is not None:
                return a
        return f"craft:{item}"

    if item in GATHER_ITEMS:
        cell = ITEM_TO_CELL[item]
        _, tier = GATHER[cell]
        # Need a tool of sufficient tier before gathering
        if env._best_tool_tier() < tier:
            a = next_action(env, TIER_TO_TOOL[tier], 1)
            if a is not None:
                return a
        node = env.find_nearest(cell)
        if node is None:
            return FAIL
        if env.pos == node:
            return "gather"
        return _move_toward(env.pos, node)

    return FAIL


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    task_item: str
    task_text: str
    initial_state: State
    actions: List[str]
    success: bool
    steps: int
    failed_actions: int


def run_oracle(env: CraftWorld, task_text: str = "") -> Trajectory:
    """Drive the environment with the oracle until the task is done."""
    initial = env.get_state()
    actions: List[str] = []
    failed = 0
    while True:
        a = next_action(env, env.task_item, 1)
        if a is None or a == FAIL:
            break
        if a == "gather":
            a = "gather:" + GATHER[int(env.grid[env.pos])][0]   # record what was gathered
        res = env.step(a)
        actions.append(a)
        failed += (not res.ok)
        if res.done:
            break
    return Trajectory(
        task_item=env.task_item,
        task_text=task_text or TASK_PHRASES[env.task_item][0],
        initial_state=initial,
        actions=actions,
        success=env.task_complete(),
        steps=env.step_count,
        failed_actions=failed,
    )


# ---------------------------------------------------------------------------
# Scenario generation (shared by pool building and test episodes)
# ---------------------------------------------------------------------------

# Several natural-language phrasings per task, so task-text similarity is not
# trivially exact-match (this is what the TF-IDF task retrieval operates on).
TASK_PHRASES: Dict[str, List[str]] = {
    "wooden_pickaxe": ["craft a wooden pickaxe", "make a wooden pickaxe",
                       "obtain a wooden pickaxe", "build a wood pickaxe"],
    "stone_pickaxe":  ["craft a stone pickaxe", "make a stone pickaxe",
                       "obtain a stone pickaxe", "build a pickaxe from stone"],
    "furnace":        ["craft a furnace", "build a furnace",
                       "make a furnace from cobblestone", "obtain a furnace"],
    "iron_ingot":     ["smelt an iron ingot", "obtain an iron ingot",
                       "make iron ingot", "produce an iron ingot"],
    "iron_pickaxe":   ["craft an iron pickaxe", "make an iron pickaxe",
                       "obtain an iron pickaxe", "build a pickaxe from iron"],
}
TASKS = list(TASK_PHRASES)

# Initial-inventory presets: how far along the tech tree the agent starts.
# These are the knob that creates "same task, different state".
INVENTORY_PRESETS: Dict[str, Dict[str, int]] = {
    "empty":        {},
    "has_wood":     {"wood": 3},
    "has_planks":   {"planks": 6, "stick": 2},
    "has_wood_pick": {"wooden_pickaxe": 1, "stick": 2},
    "has_stone_pick": {"stone_pickaxe": 1, "stick": 4, "cobblestone": 4},
    "has_furnace":  {"stone_pickaxe": 1, "furnace": 1, "planks": 4, "stick": 2},
}
PRESETS = list(INVENTORY_PRESETS)


def make_episode(env: CraftWorld, task_item: str, preset: str, seed: int,
                 rng: Optional[np.random.Generator] = None) -> str:
    """Reset env for (task, preset, world seed) and return a task phrasing."""
    rng = rng or np.random.default_rng(seed)
    env.reset(task_item, initial_inventory=INVENTORY_PRESETS[preset], seed=seed)
    return str(rng.choice(TASK_PHRASES[task_item]))


if __name__ == "__main__":
    # Sanity check: every task should be solvable from every preset.
    env = CraftWorld(seed=0)
    print(f"{'task':15s} {'preset':15s} {'success':8s} {'steps':6s} failed_actions")
    for task in TASKS:
        for preset in PRESETS:
            text = make_episode(env, task, preset, seed=42)
            tr = run_oracle(env, text)
            print(f"{task:15s} {preset:15s} {str(tr.success):8s} {tr.steps:<6d} {tr.failed_actions}")