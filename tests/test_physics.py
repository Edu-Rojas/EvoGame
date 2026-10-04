import numpy as np

from especies.physics import move, separate
from especies.state import Action


def test_bodies_on_the_same_point_split_apart(make_world):
    w = make_world(n=2, pos=[500.0, 500.0])
    separate(w)
    assert np.linalg.norm(w.pos[0] - w.pos[1]) > 0.1


def test_separation_pulls_a_crowd_apart(make_world):
    crowd = np.array([400.0, 400.0]) + np.random.default_rng(0).normal(0, 0.5, (30, 2))
    w = make_world(n=30, pos=crowd)
    for _ in range(200):
        separate(w)
    d = np.linalg.norm(w.pos[:30][:, None] - w.pos[:30][None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1.0   # se abrieron (antes estaban todos casi en el mismo punto)


def test_move_heads_toward_the_mate_and_stays_in_the_world(make_world):
    w = make_world(n=2, pos=[[1595.0, 500.0], [5.0, 500.0]], vel=0.0, age=0.0,
                   action=Action.MATE, target=[1, 0])
    for _ in range(5):
        move(w)
    assert (w.pos[:2] >= 0).all() and (w.pos[:2] < w.size).all()
    # 0 cruza el borde derecho hacia 1 en vez de recorrer todo el mundo
    assert w.vel[0, 0] > 0 and w.vel[1, 0] < 0
