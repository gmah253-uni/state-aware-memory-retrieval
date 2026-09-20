"""
State similarity functions (Eq. 2 of the proposal).

    S_state = w_i * S_inventory + w_o * S_objects + w_p * S_progress + w_v * S_visual

Each component returns a value in [0, 1]. Design choices:
  * inventory  -> weighted Jaccard over item counts. Counts matter (3 planks vs 1
                  plank) so plain set overlap would be too coarse.
  * objects    -> weighted Jaccard over nearby resource counts. What is reachable
                  nearby determines which plan is executable.
  * progress   -> Jaccard over milestone sets. Captures "where in the tech tree"
                  the agent is, independent of exact inventory numbers.
  * visual     -> cosine similarity over the flattened one-hot local view.
                  Mirrors JARVIS-1's visual-state ranking, but on a symbolic
                  observation rather than pixels.

All four are cheap (no learned models), which keeps retrieval cost low
(Challenge 4 in the review: avoid expensive high-dimensional state matching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.env import State


# ---------------------------------------------------------------------------
# Component similarities
# ---------------------------------------------------------------------------

def weighted_jaccard(a: Dict[str, int], b: Dict[str, int]) -> float:
    """sum(min) / sum(max) over the union of keys. 1.0 if both empty."""
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den > 0 else 1.0


def set_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def s_inventory(cur: State, exp: State) -> float:
    return weighted_jaccard(cur.inventory, exp.inventory)


def s_objects(cur: State, exp: State) -> float:
    return weighted_jaccard(cur.nearby_objects, exp.nearby_objects)


def s_progress(cur: State, exp: State) -> float:
    return set_jaccard(cur.progress, exp.progress)


def s_visual(cur: State, exp: State) -> float:
    return cosine(cur.local_view, exp.local_view)


COMPONENTS = {
    "inventory": s_inventory,
    "objects": s_objects,
    "progress": s_progress,
    "visual": s_visual,
}


# ---------------------------------------------------------------------------
# Combination (Eq. 2)
# ---------------------------------------------------------------------------

@dataclass
class StateWeights:
    inventory: float = 0.25
    objects: float = 0.25
    progress: float = 0.25
    visual: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return {"inventory": self.inventory, "objects": self.objects,
                "progress": self.progress, "visual": self.visual}

    def normalised(self) -> "StateWeights":
        d = self.as_dict()
        tot = sum(d.values())
        if tot == 0:
            return StateWeights(0, 0, 0, 0)
        return StateWeights(**{k: v / tot for k, v in d.items()})


def state_similarity(cur: State, exp: State, weights: StateWeights) -> float:
    """Eq. (2). Weights are normalised so the result stays in [0, 1]."""
    w = weights.normalised().as_dict()
    return sum(w[name] * fn(cur, exp) for name, fn in COMPONENTS.items())


def state_components(cur: State, exp: State) -> Dict[str, float]:
    """All four component scores, useful for logging and the ablation plots."""
    return {name: fn(cur, exp) for name, fn in COMPONENTS.items()}


if __name__ == "__main__":
    from src.env import CraftWorld

    env = CraftWorld(seed=0)
    a = env.reset("stone_pickaxe", seed=1)                                  # empty inventory
    b = env.reset("stone_pickaxe", initial_inventory={"wooden_pickaxe": 1,
                                                     "stick": 2}, seed=1)    # same world, tool in hand
    c = env.reset("stone_pickaxe", seed=2)                                  # different world

    w = StateWeights()
    print("same world, different inventory :", state_components(a, b), round(state_similarity(a, b, w), 3))
    print("different world, empty inventory:", state_components(a, c), round(state_similarity(a, c, w), 3))