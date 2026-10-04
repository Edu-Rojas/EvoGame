import numpy as np
from scipy.spatial import cKDTree

from especies.genes import Gene
from especies.state import N_ACTIONS, create_world, spawn


def test_spawn_wraps_tiny_negative_positions(cfg):
    """np.mod(-1e-14, 1600) da 1600.0, y el KD-tree con boxsize rechaza ese punto."""
    w = create_world(cfg, seed=0)
    w.alive[:] = False
    n = 2
    born = spawn(
        w,
        species=np.zeros(n, dtype=int),
        genes=np.ones((n, len(cfg.species[0].genes))),
        instinct=np.ones((n, N_ACTIONS)),
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


def test_never_exceeds_capacity(small_cfg):
    from especies.step import run
    w = create_world(small_cfg, seed=3)
    for _ in range(1500):
        run(w, 1)
        assert w.alive.sum() <= small_cfg.sim.capacity


def test_founders_respect_species_counts(cfg):
    w = create_world(cfg, seed=0)
    for sp_id, sp in enumerate(cfg.species):
        assert (w.alive & (w.species == sp_id)).sum() == sp.count


def test_creature_info_is_raw_data(cfg):
    from dataclasses import asdict

    from especies.metrics import describe_creature
    w = create_world(cfg, seed=0)
    info = describe_creature(w, int(w.alive_idx()[0]))
    d = asdict(info)                       # serializable tal cual para la API
    assert isinstance(d["energy"], float) and d["action"] in {"explore", "eat", "mate"}
    assert set(d["genes"]) == {g.key for g in Gene}
