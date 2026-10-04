import numpy as np
import pytest

from especies.genes import Gene
from especies.reproduction import reproduce
from especies.state import Action, create_world


def _full_world_with_a_couple(cfg, free_slots: int):
    """Mundo con todos los slots ocupados salvo `free_slots`, y una pareja (0, 1)
    lista y en contacto que quiere tener 4 crías (Apareamiento 10)."""
    w = create_world(cfg, seed=0)
    w.alive[:] = True
    w.alive[len(w.alive) - free_slots:] = False
    couple = np.array([0, 1])
    w.species[couple] = 0
    w.genes[couple, Gene.MATING] = 10.0
    w.radius[couple] = 4.0
    w.pos[couple] = [[100.0, 100.0], [101.0, 100.0]]
    w.energy[couple] = 100.0
    w.reserve[couple] = 200.0
    w.cooldown[couple] = 0
    w.action[couple] = Action.MATE
    w.target[couple] = [1, 0]
    ready = np.zeros(len(w.alive), dtype=bool)
    ready[couple] = True
    return w, ready


def _couple_world(make_world, n, **columns):
    defaults = dict(species=0, radius=4.0, energy=100.0, reserve=200.0, cooldown=0,
                    action=Action.MATE)
    defaults.update(columns)
    w = make_world(n=n, **defaults)
    ready = w.alive.copy()
    return w, ready


def test_each_creature_mates_once_per_tick(make_world):
    # 1 y 2 apuntan los dos a 0 y están en contacto: solo una pareja se forma
    w, ready = _couple_world(make_world, 3, target=[1, 0, 0],
                             pos=[[100.0, 100.0], [101.0, 100.0], [100.0, 101.0]])
    reproduce(w, ready)
    assert (w.cooldown[:3] > 0).sum() == 2


def test_no_mating_across_species(make_world):
    w, ready = _couple_world(make_world, 2, species=[0, 1], target=[1, 0],
                             pos=[[100.0, 100.0], [101.0, 100.0]])
    assert reproduce(w, ready) == 0


def test_no_mating_without_contact(make_world):
    w, ready = _couple_world(make_world, 2, target=[1, 0],
                             pos=[[100.0, 100.0], [180.0, 100.0]])
    assert reproduce(w, ready) == 0


def test_litter_size_by_mating_gene(cfg):
    from especies.reproduction import litter_size
    w = create_world(cfg, seed=0)
    assert {litter_size(1.0, 1.0, w) for _ in range(200)} == {1}
    assert {litter_size(10.0, 10.0, w) for _ in range(200)} == {cfg.reproduction.max_litter}
    mid = [litter_size(5.5, 5.5, w) for _ in range(4000)]
    assert np.mean(mid) == pytest.approx(2.5, abs=0.05)   # redondeo al azar


def test_offspring_inherit_correctly(cfg):
    from especies.step import run
    w = create_world(cfg, seed=11)
    run(w, 800)
    kids = np.flatnonzero(w.alive & (w.generation > 0))
    assert len(kids) > 0
    assert (w.genes[kids] >= 1).all() and (w.genes[kids] <= 10).all()
    assert (w.parent_a[kids] >= 0).all() and (w.parent_b[kids] >= 0).all()
    assert (w.parent_a[kids] != w.parent_b[kids]).all()   # dos padres distintos
    lo, hi = cfg.behavior.instinct_min, cfg.behavior.instinct_max
    assert (w.instinct[kids] >= lo).all() and (w.instinct[kids] <= hi).all()


def test_litter_size_follows_max_litter(cfg):
    from dataclasses import replace

    from especies.reproduction import litter_size
    cfg6 = replace(cfg, reproduction=replace(cfg.reproduction, max_litter=6))
    w = create_world(cfg6, seed=0)
    assert {litter_size(10.0, 10.0, w) for _ in range(100)} == {6}
    assert {litter_size(1.0, 1.0, w) for _ in range(100)} == {1}


def test_full_world_parents_pay_nothing(small_cfg):
    w, ready = _full_world_with_a_couple(small_cfg, free_slots=0)
    born = reproduce(w, ready)
    assert born == 0
    assert w.energy[0] == w.energy[1] == 100.0
    assert w.cooldown[0] == w.cooldown[1] == 0
    assert w.action[0] == Action.MATE            # siguen intentándolo


def test_partial_litter_pays_only_for_born_offspring(small_cfg):
    w, ready = _full_world_with_a_couple(small_cfg, free_slots=1)
    born = reproduce(w, ready)
    assert born == 1                             # querían 4, cabía 1
    c = small_cfg.reproduction.contribution
    assert w.energy[0] == pytest.approx(100.0 * (1 - c / 4))
    kid = len(w.alive) - 1
    # misma energía que habría recibido cada una de las 4 crías sin tope
    assert w.energy[kid] == pytest.approx(c * 200.0 / 4)
