"""Invariantes que deben valer siempre, tick tras tick: atrapan regresiones numéricas."""
import numpy as np
import pytest

from especies.state import create_world
from especies.step import step


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 42])
def test_world_stays_consistent(cfg, seed):
    w = create_world(cfg, seed=seed)
    for t in range(1500):
        step(w)
        if t % 50:
            continue
        a = w.alive_idx()
        assert np.isfinite(w.pos[a]).all() and np.isfinite(w.vel[a]).all()
        assert np.isfinite(w.energy[a]).all()
        assert ((w.pos[a] >= 0) & (w.pos[a] < w.size)).all()
        assert (w.energy[a] <= w.reserve[a] + 1e-9).all()
        assert (w.energy[a] > 0).all()                       # los sin energía ya murieron
        assert (w.terrain.grass >= -1e-9).all()
        assert (w.terrain.grass <= w.terrain.grass_max + 1e-9).all()
        assert len(np.unique(w.uid[a])) == len(a)            # uid únicos entre vivos
        targets = w.target[a]
        assert (targets >= -1).all()
