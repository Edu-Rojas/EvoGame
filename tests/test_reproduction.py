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
