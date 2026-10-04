from dataclasses import replace

import numpy as np
import pytest

from especies.config import load_config
from especies.state import World, create_world


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def small_cfg(cfg):
    """Mundo chico y rápido para tests de integración."""
    return replace(cfg, sim=replace(cfg.sim, capacity=400))


@pytest.fixture
def make_world(cfg):
    """Fábrica de mundos para tests: solo quedan vivas las primeras `n` criaturas y
    cada columna que se pase por nombre se escribe en ellas.

        w = make_world(n=2, pos=[[10, 10], [12, 10]], energy=50.0)
    """
    def _make(*, n: int = 0, seed: int = 0, config=None, **columns) -> World:
        w = create_world(config or cfg, seed=seed)
        w.alive[:] = False
        s = np.arange(n)
        w.alive[s] = True
        for name, values in columns.items():
            getattr(w, name)[s] = values
        return w
    return _make
