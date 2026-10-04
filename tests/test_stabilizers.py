"""Mecánicas que estabilizan la coexistencia (ver docs/adr/0006)."""
from dataclasses import replace

import numpy as np

from especies.disease import crowding, sicken
from especies.physics import _pull_home, _toward_cover
from especies.state import create_world
from especies.terrain import Biome


def test_explorer_far_from_home_turns_back(make_world, cfg):
    far = cfg.movement.home_range * 3
    w = make_world(n=1, pos=[[500.0 + far, 500.0]], heading=0.0)     # mirando hacia +x
    w.home[0] = [500.0, 500.0]
    _pull_home(w, np.array([0]))
    assert np.cos(w.heading[0]) < 1.0                                  # giró hacia casa


def test_explorer_near_home_is_left_alone(make_world):
    w = make_world(n=1, pos=[[505.0, 500.0]], heading=0.3)
    w.home[0] = [500.0, 500.0]
    _pull_home(w, np.array([0]))
    assert w.heading[0] == 0.3


def test_fleeing_prey_bends_toward_cover(make_world):
    w = make_world(n=1)
    t = w.terrain
    # una celda de pradera con bosque cerca hacia algún lado
    grass = np.flatnonzero(t.biome == Biome.GRASSLAND)
    forest = set(np.flatnonzero(t.biome == Biome.FOREST).tolist())
    cell = next(c for c in grass if any((c + d) in forest for d in (1, -1, t.gw, -t.gw)))
    w.pos[0] = t.center_of(np.array([cell]))[0]
    away = np.array([[1.0, 0.0]])
    out = _toward_cover(w, np.array([0]), away)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert not np.allclose(out, away) or t.visibility[cell] <= t.visibility.min()


def test_crowded_creatures_pay_more(make_world, cfg):
    w = make_world(n=12, species=0, energy=100.0, appetite=1.0,
                   pos=[[500.0 + i, 500.0] for i in range(10)] + [[100.0, 100.0], [900.0, 900.0]])
    counts = crowding(w)
    assert counts[0] == 9 and counts[10] == 0
    sicken(w)
    assert w.energy[0] < 100.0 and w.energy[10] == 100.0


def test_grouped_founders_start_together_and_on_land(cfg):
    w = create_world(cfg, seed=4)
    for sp in range(len(cfg.species)):
        pos = w.pos[w.alive & (w.species == sp)]
        spread = np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean()
        assert spread < 4 * cfg.sim.founder_spread


def test_uniform_founders_spread_over_the_map(cfg):
    w = create_world(replace(cfg, sim=replace(cfg.sim, founder_spawn="uniform")), seed=4)
    pos = w.pos[w.alive & (w.species == 0)]
    assert np.ptp(pos[:, 0]) > cfg.world.width / 2
