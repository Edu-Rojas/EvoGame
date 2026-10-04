import numpy as np
import pytest

from especies.genes import Gene
from especies.metabolism import die, eat, spend
from especies.state import Action
from especies.terrain import Biome


def test_scarce_grass_is_shared(make_world):
    w = make_world(n=2, energy=0.0, reserve=1000.0, plant_eff=1.0, radius=3.0,
                   action=Action.EAT, target=0)
    w.pos[:2] = w.terrain.center_of(np.array([0, 0]))
    w.terrain.grass[0] = 2.0                  # alcanza para 2, quieren 4 cada una
    eat(w)
    assert w.energy[0] == pytest.approx(1.0)
    assert w.energy[1] == pytest.approx(1.0)
    assert w.terrain.grass[0] == pytest.approx(0.0)


def test_cannot_eat_from_afar(make_world):
    w = make_world(n=1, energy=0.0, reserve=1000.0, plant_eff=1.0, radius=3.0,
                   action=Action.EAT, target=0)
    w.pos[0] = w.terrain.center_of(np.array([0]))[0] + 100.0
    w.terrain.grass[0] = 30.0
    eat(w)
    assert w.energy[0] == 0.0


def test_pure_carnivore_does_not_strip_grass(make_world):
    w = make_world(n=1, energy=10.0, reserve=1000.0, plant_eff=0.0, radius=3.0,
                   action=Action.EAT, target=0)
    w.pos[0] = w.terrain.center_of(np.array([0]))[0]
    w.terrain.grass[0] = 20.0
    eat(w)
    assert w.terrain.grass[0] == 20.0 and w.energy[0] == 10.0


def test_bergmann_cold_costs_less_for_large_bodies(make_world):
    w = make_world(n=2, appetite=1.0, energy=100.0, vel=0.0)
    cold = np.flatnonzero(w.terrain.biome == Biome.TUNDRA)[0]
    w.pos[:2] = w.terrain.center_of(np.array([cold, cold]))
    w.genes[:2, Gene.SIZE] = [1.0, 10.0]      # mismo apetito: solo cuenta el extra por frío
    spend(w)
    spent = 100.0 - w.energy[:2]
    assert spent[1] < spent[0]


def test_die_removes_starved_and_old(make_world, cfg):
    w = make_world(n=3, energy=[10.0, 0.0, 10.0], age=[0.0, 0.0, cfg.body.base_lifespan])
    dead = die(w)
    assert sorted(dead.tolist()) == [1, 2]
    assert w.alive[0] and not w.alive[1] and not w.alive[2]
    assert w.deaths_total == 2
