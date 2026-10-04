import numpy as np

from especies.decision import decide, validate_targets
from especies.perception import Perception
from especies.state import NO_TARGET, Action


def test_hungry_creature_with_food_in_sight_eats(make_world):
    w = make_world(n=1, energy=1.0, reserve=100.0, instinct=1.0)
    thinkers = np.array([0])
    p = Perception(food=np.array([42]), mate=np.array([NO_TARGET]))
    decide(w, thinkers, p, ready=np.zeros(len(w.alive), dtype=bool))
    assert w.action[0] == Action.EAT and w.target[0] == 42


def test_carnivore_ignores_grass(make_world):
    w = make_world(n=1, energy=1.0, reserve=100.0, instinct=1.0, plant_eff=0.0)
    p = Perception(food=np.array([42]), mate=np.array([NO_TARGET]))
    decide(w, np.array([0]), p, ready=np.zeros(len(w.alive), dtype=bool))
    assert w.action[0] != Action.EAT


def test_nothing_in_sight_means_explore(make_world):
    w = make_world(n=1, energy=1.0, reserve=100.0, instinct=1.0)
    p = Perception(food=np.array([NO_TARGET]), mate=np.array([NO_TARGET]))
    decide(w, np.array([0]), p, ready=np.ones(len(w.alive), dtype=bool))
    assert w.action[0] == Action.EXPLORE and w.target[0] == NO_TARGET


def test_grazed_out_target_is_dropped(make_world):
    w = make_world(n=1, action=Action.EAT, target=7)
    w.terrain.grass[7] = 0.0
    validate_targets(w, ready=np.zeros(len(w.alive), dtype=bool))
    assert w.action[0] == Action.EXPLORE and w.target[0] == NO_TARGET


def test_dead_or_unready_mate_is_dropped(make_world):
    w = make_world(n=3, action=Action.MATE, target=[1, 0, 1])
    w.alive[1] = False                        # la pareja de 0 y de 2 murió
    ready = w.alive.copy()
    validate_targets(w, ready)
    assert w.action[0] == Action.EXPLORE and w.action[2] == Action.EXPLORE
