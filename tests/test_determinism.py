import numpy as np

from especies.state import create_world
from especies.step import run


def test_same_seed_same_simulation(cfg):
    a, b = create_world(cfg, seed=7), create_world(cfg, seed=7)
    run(a, 300)
    run(b, 300)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.genes, b.genes)
    assert a.births_total == b.births_total


def test_different_seed_different_simulation(cfg):
    a, b = create_world(cfg, seed=1), create_world(cfg, seed=2)
    run(a, 100)
    run(b, 100)
    assert not np.array_equal(a.pos, b.pos)


def test_negative_seed_picks_a_random_reproducible_seed(cfg):
    from dataclasses import replace
    w = create_world(replace(cfg, sim=replace(cfg.sim, seed=-1)))
    assert w.seed >= 0
    again = create_world(cfg, seed=w.seed)
    assert np.array_equal(w.pos, again.pos)
