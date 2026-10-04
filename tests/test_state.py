import numpy as np
from scipy.spatial import cKDTree

from especies.state import Action, create_world, spawn


def test_spawn_wraps_tiny_negative_positions(cfg):
    """np.mod(-1e-14, 1600) da 1600.0, y el KD-tree con boxsize rechaza ese punto."""
    w = create_world(cfg, seed=0)
    w.alive[:] = False
    n = 2
    born = spawn(
        w,
        species=np.zeros(n, dtype=int),
        genes=np.ones((n, len(cfg.species[0].genes))),
        instinct=np.ones((n, Action.N)),
        pos=np.array([[-1e-14, 500.0], [800.0, -1e-14]]),
        energy=None,
        age=np.zeros(n),
        generation=np.zeros(n, dtype=int),
        parent_a=np.full(n, -1),
        parent_b=np.full(n, -1),
    )
    assert born == n
    pos = w.pos[w.alive]
    assert (pos >= 0).all() and (pos < w.size).all()
    cKDTree(pos, boxsize=w.size)   # antes lanzaba ValueError
