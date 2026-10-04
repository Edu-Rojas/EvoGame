import numpy as np

from especies.physics import separate
from especies.state import create_world


def test_bodies_on_the_same_point_split_apart(cfg):
    w = create_world(cfg, seed=0)
    s = w.alive_idx()[:2]
    w.alive[:] = False
    w.alive[s] = True
    w.pos[s] = [500.0, 500.0]
    separate(w)
    assert np.linalg.norm(w.pos[s[0]] - w.pos[s[1]]) > 0.1
