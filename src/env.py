"""
CraftWorld: a small Minecraft-style crafting gridworld.

Design goals (relevant to the report's Methodology section):
  * Long-horizon tasks: crafting an iron pickaxe from scratch takes ~40-70 steps
    through a tech tree (wood -> planks -> sticks -> wooden pickaxe -> cobblestone ...).
  * Rich, varying state: inventory, position, nearby resources, local "visual" view,
    and completed subgoals all change during execution and across episodes.
  * Precondition failures: gathering without the right tool or crafting without
    ingredients FAILS (like Minecraft), so a plan retrieved from a mismatched state
    produces observable wasted actions and replanning events.

Nothing here is learned; it is a deterministic simulator driven by a seeded RNG.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# World definition
# ---------------------------------------------------------------------------

# Cell types on the grid
EMPTY, TREE, STONE, IRON = 0, 1, 2, 3
CELL_NAMES = {EMPTY: "empty", TREE: "tree", STONE: "stone", IRON: "iron"}
RESOURCE_CELLS = [TREE, STONE, IRON]
MIN_TOTAL = {TREE: 8, STONE: 14, IRON: 5}   # minimum units of each resource per world

# What each cell yields when gathered, and the minimum tool required.
# Tool tiers: 0 = hand, 1 = wooden_pickaxe, 2 = stone_pickaxe, 3 = iron_pickaxe
GATHER = {
    TREE: ("wood", 0),
    STONE: ("cobblestone", 1),
    IRON: ("iron_ore", 2),
}
TOOL_TIER = {"wooden_pickaxe": 1, "stone_pickaxe": 2, "iron_pickaxe": 3}

# Recipes: item -> (ingredients consumed, tools required but not consumed, quantity produced)
RECIPES: Dict[str, Tuple[Dict[str, int], Set[str], int]] = {
    "planks":         ({"wood": 1}, set(), 4),
    "stick":          ({"planks": 2}, set(), 4),
    "wooden_pickaxe": ({"planks": 3, "stick": 2}, set(), 1),
    "stone_pickaxe":  ({"cobblestone": 3, "stick": 2}, set(), 1),
    "furnace":        ({"cobblestone": 8}, set(), 1),
    "iron_ingot":     ({"iron_ore": 1, "planks": 1}, {"furnace"}, 1),
    "iron_pickaxe":   ({"iron_ingot": 3, "stick": 2}, set(), 1),
}

# Milestones used for the "progress" state component (ordered by tech tree depth).
MILESTONES = ["wood", "planks", "stick", "wooden_pickaxe", "cobblestone",
              "stone_pickaxe", "furnace", "iron_ore", "iron_ingot", "iron_pickaxe"]

# Movement actions
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# All action strings the environment accepts
def all_actions() -> List[str]:
    return list(MOVES) + ["gather"] + [f"craft:{item}" for item in RECIPES]


# ---------------------------------------------------------------------------
# State snapshot (what gets stored in memory and compared at retrieval time)
# ---------------------------------------------------------------------------

@dataclass
class State:
    position: Tuple[int, int]
    inventory: Dict[str, int]
    nearby_objects: Dict[str, int]      # resource type -> count within VIEW_RADIUS
    local_view: np.ndarray              # one-hot (2r+1, 2r+1, 4) "visual" observation
    progress: Set[str]                  # milestones achieved so far
    step: int

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "inventory": dict(self.inventory),
            "nearby_objects": dict(self.nearby_objects),
            "local_view": self.local_view.copy(),
            "progress": set(self.progress),
            "step": self.step,
        }


@dataclass
class StepResult:
    ok: bool                 # did the action's preconditions hold?
    reason: str              # "" if ok, else why it failed
    done: bool               # task complete or step budget exhausted
    success: bool            # task complete
    state: State


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CraftWorld:
    VIEW_RADIUS = 2  # local view is (2r+1) x (2r+1)

    def __init__(self, size: int = 10, max_steps: int = 150, seed: Optional[int] = None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((size, size), dtype=int)
        self.quantity = np.zeros((size, size), dtype=int)  # remaining units per node
        self.pos = (0, 0)
        self.inventory: Counter = Counter()
        self.progress: Set[str] = set()
        self.step_count = 0
        self.task_item = ""
        self.action_log: List[Tuple[str, bool]] = []

    # ----- episode setup ---------------------------------------------------

    def reset(self, task_item: str, initial_inventory: Optional[Dict[str, int]] = None,
              density: float = 0.12, seed: Optional[int] = None) -> State:
        """Build a random world and place the agent.

        `initial_inventory` is how experiments create "same task, different state"
        conditions (e.g. start already holding a wooden pickaxe).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        assert task_item in RECIPES or task_item in GATHER_ITEMS, f"unknown task item {task_item}"
        self.task_item = task_item
        self.step_count = 0
        self.action_log = []

        # Random resource layout. Trees most common, iron rarest.
        self.grid[:] = EMPTY
        self.quantity[:] = 0
        n_cells = self.size * self.size
        n_nodes = max(6, int(density * n_cells))
        cells = self.rng.choice(n_cells, size=n_nodes, replace=False)
        types = self.rng.choice(RESOURCE_CELLS, size=n_nodes, p=[0.5, 0.35, 0.15])
        for c, t in zip(cells, types):
            r, col = divmod(int(c), self.size)
            self.grid[r, col] = t
            self.quantity[r, col] = int(self.rng.integers(2, 5))
        # Guarantee enough of each resource for the deepest task (iron pickaxe from
        # scratch needs ~4 wood, 11 cobblestone, 3 iron ore), so every task is solvable.
        for t, min_total in MIN_TOTAL.items():
            while self.quantity[self.grid == t].sum() < min_total:
                r, col = self.rng.integers(0, self.size, size=2)
                if self.grid[r, col] == EMPTY:
                    self.grid[r, col] = t
                    self.quantity[r, col] = int(self.rng.integers(2, 5))

        self.pos = tuple(int(x) for x in self.rng.integers(0, self.size, size=2))
        self.inventory = Counter(initial_inventory or {})
        self.progress = {m for m in MILESTONES if self.inventory.get(m, 0) > 0}
        return self.get_state()

    # ----- observation -----------------------------------------------------

    def get_state(self) -> State:
        r = self.VIEW_RADIUS
        pr, pc = self.pos
        view = np.zeros((2 * r + 1, 2 * r + 1, 4), dtype=np.float32)
        nearby: Counter = Counter()
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = pr + dr, pc + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    t = int(self.grid[rr, cc])
                    view[dr + r, dc + r, t] = 1.0
                    if t != EMPTY:
                        nearby[CELL_NAMES[t]] += 1
        return State(
            position=self.pos,
            inventory=dict(self.inventory),
            nearby_objects=dict(nearby),
            local_view=view,
            progress=set(self.progress),
            step=self.step_count,
        )

    def task_complete(self) -> bool:
        return self.inventory.get(self.task_item, 0) > 0

    # ----- dynamics --------------------------------------------------------

    def _best_tool_tier(self) -> int:
        return max([TOOL_TIER[t] for t in TOOL_TIER if self.inventory.get(t, 0) > 0] + [0])

    def step(self, action: str) -> StepResult:
        """Apply one action. Failed preconditions cost a step but change nothing."""
        ok, reason = True, ""

        if action in MOVES:
            dr, dc = MOVES[action]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                self.pos = (nr, nc)
            else:
                ok, reason = False, "wall"

        elif action.startswith("gather"):
            t = int(self.grid[self.pos])
            if t == EMPTY:
                ok, reason = False, "nothing_here"
            else:
                item, tier = GATHER[t]
                if self._best_tool_tier() < tier:
                    ok, reason = False, f"need_tool_tier_{tier}"
                else:
                    self.inventory[item] += 1
                    self.progress.add(item)
                    self.quantity[self.pos] -= 1
                    if self.quantity[self.pos] <= 0:
                        self.grid[self.pos] = EMPTY

        elif action.startswith("craft:"):
            item = action.split(":", 1)[1]
            if item not in RECIPES:
                ok, reason = False, "unknown_recipe"
            else:
                ingredients, tools, qty = RECIPES[item]
                missing = [k for k, v in ingredients.items() if self.inventory.get(k, 0) < v]
                missing_tools = [t for t in tools if self.inventory.get(t, 0) < 1]
                if missing or missing_tools:
                    ok, reason = False, "missing_" + ",".join(missing + missing_tools)
                else:
                    for k, v in ingredients.items():
                        self.inventory[k] -= v
                        if self.inventory[k] == 0:
                            del self.inventory[k]
                    self.inventory[item] += qty
                    self.progress.add(item)
        else:
            ok, reason = False, "unknown_action"

        self.step_count += 1
        self.action_log.append((action, ok))
        success = self.task_complete()
        done = success or self.step_count >= self.max_steps
        return StepResult(ok=ok, reason=reason, done=done, success=success, state=self.get_state())

    # ----- helpers used by the oracle planner ------------------------------

    def find_nearest(self, cell_type: int) -> Optional[Tuple[int, int]]:
        """Manhattan-nearest remaining node of the given type, or None."""
        rows, cols = np.where(self.grid == cell_type)
        if len(rows) == 0:
            return None
        d = np.abs(rows - self.pos[0]) + np.abs(cols - self.pos[1])
        i = int(np.argmin(d))
        return (int(rows[i]), int(cols[i]))

    def render(self) -> str:
        chars = {EMPTY: ".", TREE: "T", STONE: "S", IRON: "I"}
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                row.append("@" if (r, c) == self.pos else chars[int(self.grid[r, c])])
            lines.append(" ".join(row))
        return "\n".join(lines) + f"\ninv={dict(self.inventory)} step={self.step_count}"


GATHER_ITEMS = {"wood", "cobblestone", "iron_ore"}


if __name__ == "__main__":
    # Smoke test: random actions for a few steps.
    env = CraftWorld(seed=0)
    s = env.reset("wooden_pickaxe", seed=0)
    print(env.render())
    rng = np.random.default_rng(1)
    acts = all_actions()
    for _ in range(10):
        a = acts[rng.integers(len(acts))]
        res = env.step(a)
        print(f"{a:22s} ok={res.ok} reason={res.reason}")
    print(env.render())