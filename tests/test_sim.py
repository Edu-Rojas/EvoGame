import numpy as np
import pytest

from especies.genes import Gene
from especies.metabolism import eat
from especies.geometry import torus_delta, wrap
from especies.reproduction import litter_size
from especies.state import Action, create_world
from especies.step import run


def test_torus_delta_takes_shortest_path():
    size = np.array([100.0, 100.0])
    d = torus_delta(np.array([[95.0, 50.0]]), np.array([[5.0, 50.0]]), size)
    assert d[0, 0] == pytest.approx(10.0)   # cruza el borde en vez de ir 90 hacia atrás


def test_wrap_never_returns_the_edge():
    size = np.array([1600.0, 1000.0])
    p = wrap(np.array([[-1e-17, 1000.0], [1600.0, -0.5]]), size)
    assert (p >= 0).all() and (p < size).all()


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


def test_litter_size(cfg):
    w = create_world(cfg, seed=0)
    assert {litter_size(1.0, 1.0, w) for _ in range(200)} == {1}
    assert {litter_size(10.0, 10.0, w) for _ in range(200)} == {4}
    mid = [litter_size(5.5, 5.5, w) for _ in range(4000)]
    assert np.mean(mid) == pytest.approx(2.5, abs=0.05)   # redondeo al azar


def test_scarce_grass_is_shared(cfg):
    w = create_world(cfg, seed=0)
    t = w.terrain
    w.alive[:] = False
    s = np.array([0, 1])
    cell = 0
    w.alive[s] = True
    w.energy[s] = 0.0
    w.reserve[s] = 1000.0
    w.plant_eff[s] = 1.0
    w.radius[s] = 3.0
    w.pos[s] = t.center_of(np.array([cell, cell]))
    w.action[s] = Action.EAT
    w.target[s] = cell
    t.grass[cell] = 2.0               # alcanza para 2, quieren 4 cada una
    eat(w)
    assert w.energy[0] == pytest.approx(1.0)
    assert w.energy[1] == pytest.approx(1.0)
    assert t.grass[cell] == pytest.approx(0.0)


def test_every_species_reproduces(cfg):
    """Regresión: la separación de cuerpos llegó a impedir que se aparearan los grandes."""
    w = create_world(cfg, seed=42)
    best = np.zeros(len(cfg.species), dtype=int)
    for _ in range(10):
        run(w, 100)
        for sp in range(len(cfg.species)):
            alive = w.alive & (w.species == sp)
            if alive.any():
                best[sp] = max(best[sp], w.generation[alive].max())
    assert (best >= 1).all(), f"generación máxima por especie: {best}"


def test_separation_pulls_overlapping_bodies_apart(cfg):
    from especies.physics import separate
    w = create_world(cfg, seed=0)
    a = w.alive_idx()[:30]
    w.alive[:] = False
    w.alive[a] = True
    w.pos[a] = np.array([400.0, 400.0]) + np.random.default_rng(0).normal(0, 0.5, (30, 2))
    for _ in range(200):
        separate(w)
    d = np.linalg.norm(w.pos[a][:, None] - w.pos[a][None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1.0   # se abrieron (antes estaban todos casi en el mismo punto)


def test_never_exceeds_capacity(small_cfg):
    w = create_world(small_cfg, seed=3)
    for _ in range(1500):
        run(w, 1)
        assert w.alive.sum() <= small_cfg.sim.capacity


def test_offspring_inherit_correctly(cfg):
    w = create_world(cfg, seed=11)
    run(w, 800)
    kids = np.flatnonzero(w.alive & (w.generation > 0))
    assert len(kids) > 0
    assert (w.genes[kids] >= 1).all() and (w.genes[kids] <= 10).all()
    assert (w.parent_a[kids] >= 0).all() and (w.parent_b[kids] >= 0).all()
    assert (w.parent_a[kids] != w.parent_b[kids]).all()   # dos padres distintos
    lo, hi = cfg.behavior.instinct_min, cfg.behavior.instinct_max
    assert (w.instinct[kids] >= lo).all() and (w.instinct[kids] <= hi).all()


def test_selection_pushes_size_down_in_1a(cfg):
    """En la 1a el tamaño solo cuesta (sus ventajas llegan con las peleas en la 1b),
    así que la selección debería empujarlo hacia abajo. Si esto falla, algo no
    está seleccionando."""
    w = create_world(cfg, seed=42)
    sp = 0  # Conejos
    before = w.genes[w.alive & (w.species == sp), Gene.SIZE].mean()
    run(w, 3000)
    after = w.genes[w.alive & (w.species == sp), Gene.SIZE].mean()
    assert after < before
