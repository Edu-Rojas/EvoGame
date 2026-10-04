import numpy as np
from scipy.spatial import cKDTree

from especies.perception import perceive
from especies.state import NO_TARGET


def _perceive(w, ready):
    slots = w.alive_idx()
    tree = cKDTree(w.pos[slots], boxsize=w.size)
    return perceive(w, slots, ready, tree, slots)


def test_mate_is_nearest_ready_same_species_and_never_itself(make_world):
    # 0 y 1: misma especie, cerca. 2: otra especie, más cerca de 0 que 1.
    w = make_world(n=3, species=[0, 0, 1], det_radius=100.0,
                   pos=[[500.0, 500.0], [520.0, 500.0], [505.0, 500.0]])
    ready = w.alive.copy()
    p = _perceive(w, ready)
    assert p.mate[0] == 1 and p.mate[1] == 0
    assert p.mate[2] == NO_TARGET            # no hay otro de su especie


def test_mate_must_be_ready_and_in_range(make_world):
    w = make_world(n=3, species=0, det_radius=50.0,
                   pos=[[500.0, 500.0], [510.0, 500.0], [700.0, 500.0]])
    ready = w.alive.copy()
    ready[1] = False                          # el único cercano no está listo
    p = _perceive(w, ready)
    assert p.mate[0] == NO_TARGET            # el 2 está fuera del radio


def test_food_picks_a_grassy_cell_in_range(make_world):
    # radio holgado: el bioma (bosque = 0,6) también recorta lo que se ve
    w = make_world(n=1, det_radius=200.0, pos=[[500.0, 500.0]])
    w.terrain.grass[:] = 0.0
    target = w.terrain.cell_of(np.array([[530.0, 500.0]]))[0]
    w.terrain.grass[target] = 25.0
    p = _perceive(w, w.alive.copy())
    assert p.food[0] == target


def test_no_food_when_everything_is_eaten(make_world):
    w = make_world(n=1, det_radius=60.0, pos=[[500.0, 500.0]])
    w.terrain.grass[:] = 0.0
    assert _perceive(w, w.alive.copy()).food[0] == NO_TARGET
