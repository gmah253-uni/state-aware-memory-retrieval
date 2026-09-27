"""
Memory-augmented planning agent (the system under test).

Loop, per episode:
  1. Retrieve the top-scoring experience for (task text, current state) with the
     configured retrieval rule (Eq. 1).
  2. Abstract its stored action list into a SUBGOAL PLAN, e.g.
        [("gather","wood",3), ("craft","planks"), ("craft","stick"), ...]
     This matches Optimus-1, whose experience pool stores task *plans*, with a
     separate low-level controller executing each subgoal.
  3. Execute subgoals with a simple controller (navigate to nearest node and
     gather / craft). Plans are followed WITHOUT verification: a subgoal that is
     already satisfied is still executed. This is deliberate - it isolates the
     effect of retrieval quality, which is what the experiment is about.
  4. If a subgoal fails (precondition unmet) or the plan ends without success,
     the failed plan is counted as a REPLANNING EVENT (`replans`). While
     replans <= max_replans the agent re-retrieves using the *current* state,
     excluding experiences already tried; once the counter exceeds max_replans
     (with max_replans=3: after the FOURTH failed plan) it falls back to the
     oracle so every episode terminates with a well-defined outcome. Fallback
     episodes therefore log replans = max_replans + 1.

Metrics logged per episode (as listed in the proposal's experimental plan):
  success, steps, failed_actions (redundant), replans, excess_steps
  (steps minus the oracle's steps from the identical start), oracle_fallback.
"""

from __future__ import annotations

import copy

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from src.env import GATHER, CraftWorld
from src.memory import ExperiencePool, RetrievalConfig, RetrievalResult
from src.oracle import ITEM_TO_CELL, _move_toward, next_action, run_oracle, FAIL

Subgoal = Tuple[str, str, int]   # ("gather", item, n) or ("craft", item, 1)


# ---------------------------------------------------------------------------
# Plan abstraction
# ---------------------------------------------------------------------------

def abstract_plan(actions: List[str]) -> List[Subgoal]:
    """Compress a raw action list into subgoals; movement is dropped."""
    plan: List[Subgoal] = []
    for a in actions:
        if a.startswith("gather:"):
            item = a.split(":", 1)[1]
            if plan and plan[-1][0] == "gather" and plan[-1][1] == item:
                plan[-1] = ("gather", item, plan[-1][2] + 1)
            else:
                plan.append(("gather", item, 1))
        elif a.startswith("craft:"):
            plan.append(("craft", a.split(":", 1)[1], 1))
    return plan


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    config: str
    task_item: str
    task_text: str
    preset: str
    world_seed: int
    success: bool
    steps: int
    failed_actions: int
    replans: int
    oracle_fallback: bool
    oracle_steps: int
    excess_steps: int
    first_retrieved_preset: str
    first_s_task: float
    first_s_state: float

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MemoryAgent:
    def __init__(self, pool: ExperiencePool, cfg: RetrievalConfig,
                 max_replans: int = 3, max_subgoal_moves: int = 40,
                 rng: Optional[np.random.Generator] = None):
        self.pool = pool
        self.cfg = cfg
        self.rng = rng
        self.max_replans = max_replans
        self.max_subgoal_moves = max_subgoal_moves

    # ----- low-level controller -------------------------------------------

    def _execute_subgoal(self, env: CraftWorld, sg: Subgoal) -> Tuple[bool, bool, int]:
        """Returns (subgoal_ok, episode_done, failed_action_count)."""
        kind, item, n = sg
        failed = 0
        if kind == "craft":
            res = env.step(f"craft:{item}")
            failed += (not res.ok)
            return res.ok, res.done, failed

        # gather: navigate to nearest node of that type, then gather, n times
        cell = ITEM_TO_CELL[item]
        for _ in range(n):
            moves = 0
            while True:
                node = env.find_nearest(cell)
                if node is None:
                    return False, env.step_count >= env.max_steps, failed
                if env.pos == node:
                    break
                res = env.step(_move_toward(env.pos, node))
                moves += 1
                if res.done or moves > self.max_subgoal_moves:
                    return False, res.done, failed
            res = env.step(f"gather:{item}")
            failed += (not res.ok)
            if not res.ok:
                return False, res.done, failed
            if res.done:
                return True, True, failed
        return True, False, failed

    # ----- main loop ---------------------------------------------------------

    def run_episode(self, env: CraftWorld, task_text: str, preset: str,
                    world_seed: int) -> EpisodeResult:
        # Oracle reference from the identical start state
        oracle_steps = run_oracle(copy.deepcopy(env), task_text).steps

        tried: set = set()
        replans = 0
        failed_actions = 0
        fallback = False
        first: Optional[RetrievalResult] = None
        done = env.task_complete()

        while not done:
            # --- retrieve ---
            cands = self.pool.retrieve(task_text, env.get_state(), self.cfg, k=len(self.pool), rng=self.rng)
            cands = [c for c in cands if c.experience.id not in tried]
            if not cands or replans > self.max_replans:
                fallback = True
                break
            top = cands[0]
            if first is None:
                first = top
            tried.add(top.experience.id)

            # --- execute abstracted plan ---
            plan_ok = True
            for sg in abstract_plan(top.experience.actions):
                ok, done, f = self._execute_subgoal(env, sg)
                failed_actions += f
                if done or not ok:
                    plan_ok = ok
                    break
            if done:
                break
            # plan finished (or broke) without completing the task -> replan
            replans += 1

        # --- oracle fallback so the episode has a well-defined outcome ---
        if fallback and not env.task_complete():
            while env.step_count < env.max_steps:
                a = next_action(env, env.task_item, 1)
                if a is None or a == FAIL:
                    break
                if a == "gather":
                    a = "gather:" + GATHER[int(env.grid[env.pos])][0]
                res = env.step(a)
                failed_actions += (not res.ok)
                if res.done:
                    break

        return EpisodeResult(
            config=self.cfg.name,
            task_item=env.task_item,
            task_text=task_text,
            preset=preset,
            world_seed=world_seed,
            success=env.task_complete(),
            steps=env.step_count,
            failed_actions=failed_actions,
            replans=replans,
            oracle_fallback=fallback,
            oracle_steps=oracle_steps,
            excess_steps=env.step_count - oracle_steps,
            first_retrieved_preset=first.experience.preset if first else "",
            first_s_task=first.s_task if first else 0.0,
            first_s_state=first.s_state if first else 0.0,
        )


if __name__ == "__main__":
    from src.memory import CONFIGS, build_pool
    from src.oracle import make_episode

    pool = build_pool(n_per_combo=4, seed=0)
    print(f"pool size: {len(pool)}\n")
    print(f"{'config':15s} {'task':14s} {'preset':14s} {'succ':5s} {'steps':5s} "
          f"{'orc':4s} {'excess':6s} {'fail':4s} {'replan':6s} retrieved_from")
    for task, preset in [("stone_pickaxe", "has_wood_pick"), ("stone_pickaxe", "empty"),
                         ("iron_pickaxe", "has_furnace"), ("iron_pickaxe", "empty")]:
        for name in ["A_task_only", "B_state_aware", "B_inventory"]:
            env = CraftWorld()
            text = make_episode(env, task, preset, seed=777)
            r = MemoryAgent(pool, CONFIGS[name]).run_episode(env, text, preset, 777)
            print(f"{name:15s} {task:14s} {preset:14s} {str(r.success):5s} {r.steps:<5d} "
                  f"{r.oracle_steps:<4d} {r.excess_steps:<6d} {r.failed_actions:<4d} "
                  f"{r.replans:<6d} {r.first_retrieved_preset}")