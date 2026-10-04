import numpy as np

from especies.perception import perceive
from especies.state import NO_TARGET


def _perceive(w, ready):
    return perceive(w, w.alive_idx(), ready)


def test_mate_is_found_behind_a_crowd_of_another_species(make_world):
    """Regresión: buscar entre los K vecinos de cualquier especie dejaba ciega a la
    minoría rodeada por otra especie (favorecía a la más numerosa)."""
    n_other = 12
    ring = 500.0 + 8.0 * np.stack([np.cos(np.linspace(0, 6.2, n_other)),
                                   np.sin(np.linspace(0, 6.2, n_other))], axis=1)
    pos = np.vstack([[[500.0, 500.0], [530.0, 500.0]], ring])
    w = make_world(n=2 + n_other, species=[0, 0] + [1] * n_other, det_radius=200.0, pos=pos)
    p = _perceive(w, w.alive.copy())
    assert p.mate[0] == 1


def test_unready_creatures_do_not_look_for_mates(make_world):
    w = make_world(n=2, species=0, det_radius=100.0, pos=[[500.0, 500.0], [510.0, 500.0]])
    ready = w.alive.copy()
    ready[0] = False
    p = _perceive(w, ready)
    assert p.mate[0] == NO_TARGET


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
    w = make_world(n=1, det_radius=200.0, pos=[[500.0, 500.0]], leaf_reach=0.0)
    w.terrain.grass[:] = 0.0
    target = w.terrain.cell_of(np.array([[530.0, 500.0]]))[0]
    w.terrain.grass[target] = 25.0
    p = _perceive(w, w.alive.copy())
    assert p.food[0] == target


def test_carnivore_does_not_see_grass_as_food(make_world):
    w = make_world(n=1, det_radius=200.0, pos=[[500.0, 500.0]], plant_eff=0.0, meat_eff=1.0)
    w.terrain.meat[:] = 0.0
    assert _perceive(w, w.alive.copy()).food[0] == NO_TARGET


def test_carnivore_sees_meat(make_world):
    w = make_world(n=1, det_radius=200.0, pos=[[500.0, 500.0]], plant_eff=0.0, meat_eff=1.0)
    cell = w.terrain.cell_of(np.array([[530.0, 500.0]]))[0]
    w.terrain.meat[cell] = 40.0
    assert _perceive(w, w.alive.copy()).food[0] == cell


def test_predator_sees_small_prey_but_not_big_ones(make_world):
    # 0: depredador tamaño 4; 1: presa tamaño 2 (cazable); 2: tamaño 9 (refugio por tamaño)
    w = make_world(n=3, species=[0, 1, 2], det_radius=200.0, meat_eff=[1.0, 0.0, 0.0],
                   pos=[[500.0, 500.0], [560.0, 500.0], [520.0, 500.0]])
    w.genes[:3, 0] = [4.0, 2.0, 9.0]
    p = _perceive(w, w.alive.copy())
    assert p.prey[0] == 1                      # el grande está más cerca pero no se caza


def test_prey_perceives_the_predator_as_a_threat(make_world):
    w = make_world(n=2, species=[0, 1], det_radius=200.0, meat_eff=[1.0, 0.0],
                   pos=[[500.0, 500.0], [530.0, 500.0]])
    w.genes[:2, 0] = [4.0, 2.0]
    p = _perceive(w, w.alive.copy())
    assert p.threat[1] == 0 and p.threat_level[1] > 0
    assert p.threat[0] == -1                   # la presa no come carne: no es amenaza


def test_no_food_when_everything_is_eaten(make_world):
    w = make_world(n=1, det_radius=60.0, pos=[[500.0, 500.0]], leaf_reach=1.0)
    w.terrain.grass[:] = 0.0
    w.terrain.leaves[:] = 0.0
    assert _perceive(w, w.alive.copy()).food[0] == NO_TARGET
