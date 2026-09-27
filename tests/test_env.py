"""
Sanity tests. Run from the project root with:  pytest

These check the properties the report relies on, so any later change to
env.py / state.py / memory.py that breaks an assumption is caught immediately.
"""

import numpy as np
import pytest

from src.env import CraftWorld, MIN_TOTAL, RECIPES, TREE, STONE, IRON
from src.oracle import INVENTORY_PRESETS, PRESETS, TASKS, make_episode, run_oracle
from src.state import StateWeights, state_components, state_similarity
from src.memory import CONFIGS, build_pool


# ---------------------------------------------------------------- environment

def test_every_task_solvable_from_every_preset():
    env = CraftWorld()
    for seed in range(20):
        for task in TASKS:
            for preset in PRESETS:
                make_episode(env, task, preset, seed=seed)
                tr = run_oracle(env)
                assert tr.success, f"{task} from {preset} seed {seed} unsolvable"
                assert tr.failed_actions == 0, "oracle should never take a failing action"


def test_every_task_solvable_on_1000_worlds():
    """Backs the claim in the report: from an empty inventory every task is solvable
    on 1,000 random worlds (world generation tops up scarce resources)."""
    env = CraftWorld()
    for seed in range(1000):
        for task in TASKS:
            env.reset(task, seed=seed)
            assert run_oracle(env).success, f"{task} unsolvable on world {seed}"


def test_world_has_minimum_resources():
    env = CraftWorld()
    for seed in range(50):
        env.reset("iron_pickaxe", seed=seed)
        for cell, min_total in MIN_TOTAL.items():
            assert env.quantity[env.grid == cell].sum() >= min_total


def test_craft_without_ingredients_fails_and_costs_a_step():
    env = CraftWorld()
    env.reset("wooden_pickaxe", seed=0)
    res = env.step("craft:planks")
    assert not res.ok and res.reason == "missing_wood"
    assert env.step_count == 1
    assert env.inventory == {}


def test_gather_requires_tool():
    env = CraftWorld()
    env.reset("stone_pickaxe", seed=0)
    env.pos = env.find_nearest(STONE)
    assert not env.step("gather").ok                       # bare hands
    env.inventory["wooden_pickaxe"] = 1
    assert env.step("gather").ok                           # with tool
    assert env.inventory["cobblestone"] == 1


def test_resource_nodes_deplete():
    env = CraftWorld()
    env.reset("wooden_pickaxe", seed=0)
    env.pos = env.find_nearest(TREE)
    q = env.quantity[env.pos]
    for _ in range(q):
        assert env.step("gather").ok
    assert env.grid[env.pos] == 0                          # node gone
    assert not env.step("gather").ok


def test_step_budget_terminates_episode():
    env = CraftWorld(max_steps=5)
    env.reset("iron_pickaxe", seed=0)
    for _ in range(5):
        res = env.step("up")
    assert res.done and not res.success


# ---------------------------------------------------------------- similarity

def test_identical_state_scores_one():
    env = CraftWorld()
    s = env.reset("stone_pickaxe", initial_inventory={"wood": 2}, seed=3)
    comps = state_components(s, s)
    assert all(abs(v - 1.0) < 1e-9 for v in comps.values())
    assert abs(state_similarity(s, s, StateWeights()) - 1.0) < 1e-9


def test_similarity_bounded():
    env = CraftWorld()
    states = [env.reset(t, initial_inventory=INVENTORY_PRESETS[p], seed=i)
              for i, (t, p) in enumerate([(t, p) for t in TASKS for p in PRESETS])]
    for a in states[:10]:
        for b in states[10:20]:
            for v in state_components(a, b).values():
                assert -1e-9 <= v <= 1 + 1e-9
            assert 0 <= state_similarity(a, b, StateWeights()) <= 1 + 1e-9


def test_weights_normalise():
    w = StateWeights(2, 0, 0, 0).normalised()
    assert w.inventory == 1.0 and w.objects == 0.0


# ---------------------------------------------------------------- retrieval

def test_pool_builds_and_skips_trivial():
    pool = build_pool(n_per_combo=1, seed=0)
    assert len(pool) > 0
    assert all(e.steps > 0 and e.success for e in pool.experiences)


def test_task_only_ignores_state():
    pool = build_pool(n_per_combo=2, seed=0)
    env = CraftWorld()
    text = make_episode(env, "stone_pickaxe", "has_wood_pick", seed=99)
    top = pool.retrieve(text, env.get_state(), CONFIGS["A_task_only"], k=1)[0]
    assert top.s_state == 0.0 and top.s_task == pytest.approx(top.score)


def test_state_aware_prefers_matching_inventory():
    pool = build_pool(n_per_combo=4, seed=0)
    env = CraftWorld()
    text = make_episode(env, "iron_pickaxe", "has_furnace", seed=99)
    top = pool.retrieve(text, env.get_state(), CONFIGS["B_inventory"], k=1)[0]
    assert top.experience.preset == "has_furnace"


def test_retrieval_is_reproducible():
    pool = build_pool(n_per_combo=2, seed=0)
    env = CraftWorld()
    text = make_episode(env, "furnace", "empty", seed=5)
    s = env.get_state()
    r1 = pool.retrieve(text, s, CONFIGS["A_task_only"], k=3, rng=np.random.default_rng(1))
    r2 = pool.retrieve(text, s, CONFIGS["A_task_only"], k=3, rng=np.random.default_rng(1))
    assert [r.experience.id for r in r1] == [r.experience.id for r in r2]