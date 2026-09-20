"""
Experience pool and retrieval (Eq. 1 of the proposal).

    R(e) = alpha * S_task(T_curr, T_exp) + beta * S_state(S_curr, S_exp)

An Experience is an abstracted trajectory in the spirit of Optimus-1's
Abstracted Multimodal Experience Pool: task text, the state it started from,
the action plan that was executed, and whether it succeeded.

Task similarity uses TF-IDF cosine over the task phrasing. This is a deliberate
stand-in for the embedding-based retrieval used by Optimus-1 / JARVIS-1: cheap,
deterministic, and sufficient to distinguish "craft a stone pickaxe" from
"craft an iron pickaxe" while still scoring different phrasings of the same
task as similar.

The baseline (Model A) is alpha=1, beta=0: retrieval ignores state entirely.
The proposed method (Model B) sets beta>0 and uses the Eq. 2 weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env import CraftWorld, State
from src.oracle import (PRESETS, TASKS, Trajectory, make_episode, run_oracle)
from src.state import StateWeights, state_components, state_similarity


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    id: int
    task_item: str
    task_text: str
    state: State            # state at the START of the trajectory
    actions: List[str]      # the plan that was executed
    success: bool
    steps: int
    preset: str = ""        # bookkeeping only; never used for retrieval

    @classmethod
    def from_trajectory(cls, id: int, tr: Trajectory, preset: str = "") -> "Experience":
        return cls(id=id, task_item=tr.task_item, task_text=tr.task_text,
                   state=tr.initial_state, actions=list(tr.actions),
                   success=tr.success, steps=tr.steps, preset=preset)


# ---------------------------------------------------------------------------
# Minimal TF-IDF
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


class TfIdf:
    def __init__(self, docs: List[str]):
        self.vocab: Dict[str, int] = {}
        toks = [_tokens(d) for d in docs]
        for t in toks:
            for w in t:
                self.vocab.setdefault(w, len(self.vocab))
        n, v = len(docs), len(self.vocab)
        df = np.zeros(v)
        for t in toks:
            for w in set(t):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1.0     # smoothed idf
        self.matrix = np.stack([self.vector(d) for d in docs]) if docs else np.zeros((0, v))

    def vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for w in _tokens(text):
            if w in self.vocab:
                vec[self.vocab[w]] += 1
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def similarities(self, text: str) -> np.ndarray:
        """Cosine similarity of `text` against every indexed document."""
        return self.matrix @ self.vector(text)


# ---------------------------------------------------------------------------
# Retrieval configuration and pool
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    name: str
    alpha: float = 1.0                 # weight on task similarity
    beta: float = 0.0                  # weight on state similarity
    weights: StateWeights = field(default_factory=StateWeights)

    def score(self, s_task: float, s_state: float) -> float:
        tot = self.alpha + self.beta
        return (self.alpha * s_task + self.beta * s_state) / tot if tot > 0 else 0.0


# The two configurations from the experimental plan, plus ablations.
CONFIGS: Dict[str, RetrievalConfig] = {
    "A_task_only":     RetrievalConfig("A_task_only", alpha=1.0, beta=0.0),
    "B_state_aware":   RetrievalConfig("B_state_aware", alpha=0.5, beta=0.5),
    "B_inventory":     RetrievalConfig("B_inventory", 0.5, 0.5, StateWeights(1, 0, 0, 0)),
    "B_objects":       RetrievalConfig("B_objects", 0.5, 0.5, StateWeights(0, 1, 0, 0)),
    "B_progress":      RetrievalConfig("B_progress", 0.5, 0.5, StateWeights(0, 0, 1, 0)),
    "B_visual":        RetrievalConfig("B_visual", 0.5, 0.5, StateWeights(0, 0, 0, 1)),
}


@dataclass
class RetrievalResult:
    experience: Experience
    score: float
    s_task: float
    s_state: float
    components: Dict[str, float]


class ExperiencePool:
    def __init__(self):
        self.experiences: List[Experience] = []
        self._tfidf: Optional[TfIdf] = None

    def add(self, exp: Experience) -> None:
        self.experiences.append(exp)
        self._tfidf = None            # invalidate index

    def _index(self) -> TfIdf:
        if self._tfidf is None:
            self._tfidf = TfIdf([e.task_text for e in self.experiences])
        return self._tfidf

    def __len__(self) -> int:
        return len(self.experiences)

    def retrieve(self, task_text: str, state: State, cfg: RetrievalConfig,
                 k: int = 1, successful_only: bool = True,
                 rng: Optional[np.random.Generator] = None) -> List[RetrievalResult]:
        """Rank all experiences by Eq. 1 and return the top-k.

        Ties are broken randomly (seeded via `rng`) so that a task-only
        configuration, which scores every phrasing-identical experience equally,
        does not systematically favour whichever experience was stored first.
        """
        s_task_all = self._index().similarities(task_text)
        tiebreak = (rng.permutation(len(self.experiences)) if rng is not None
                    else np.arange(len(self.experiences)))
        results: List[RetrievalResult] = []
        for e, s_task in zip(self.experiences, s_task_all):
            if successful_only and not e.success:
                continue
            comps = state_components(state, e.state)
            s_state = state_similarity(state, e.state, cfg.weights) if cfg.beta > 0 else 0.0
            results.append(RetrievalResult(e, cfg.score(float(s_task), s_state),
                                           float(s_task), s_state, comps))
        results.sort(key=lambda r: (-round(r.score, 9), tiebreak[r.experience.id]))
        return results[:k]


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def build_pool(n_per_combo: int = 4, seed: int = 0, size: int = 10) -> ExperiencePool:
    """Oracle-generated pool: n_per_combo worlds for every (task, preset) pair.

    Trivial episodes (task already satisfied at start) are skipped: they carry
    an empty plan and would never be useful to retrieve.
    """
    rng = np.random.default_rng(seed)
    env = CraftWorld(size=size)
    pool = ExperiencePool()
    world_seed = 10_000 * (seed + 1)
    for task in TASKS:
        for preset in PRESETS:
            for _ in range(n_per_combo):
                world_seed += 1
                text = make_episode(env, task, preset, seed=world_seed, rng=rng)
                tr = run_oracle(env, text)
                if tr.steps == 0:
                    continue
                pool.add(Experience.from_trajectory(len(pool), tr, preset))
    return pool


if __name__ == "__main__":
    pool = build_pool(n_per_combo=4, seed=0)
    print(f"pool size: {len(pool)} experiences\n")

    # Query: stone pickaxe, agent already holds a wooden pickaxe.
    env = CraftWorld()
    text = make_episode(env, "stone_pickaxe", "has_wood_pick", seed=777)
    state = env.get_state()
    print(f"query: '{text}'  inventory={state.inventory}\n")

    for name in ["A_task_only", "B_state_aware"]:
        cfg = CONFIGS[name]
        print(f"--- {name} (alpha={cfg.alpha}, beta={cfg.beta}) ---")
        for r in pool.retrieve(text, state, cfg, k=3):
            e = r.experience
            print(f"  score={r.score:.3f} task={r.s_task:.2f} state={r.s_state:.2f} "
                  f"| '{e.task_text}' from preset={e.preset:14s} plan_len={len(e.actions)}")
        print()