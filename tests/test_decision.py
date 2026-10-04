import numpy as np

from especies.decision import decide, validate_targets
from especies.perception import Perception
from especies.state import NO_TARGET, Action


def _p(n=1, food=NO_TARGET, mate=NO_TARGET, prey=NO_TARGET, threat=NO_TARGET, level=0.0,
       seen=None):
    full = lambda v, dt=np.int64: np.full(n, v, dtype=dt)  # noqa: E731
    if seen is None:
        seen = 0 if prey == NO_TARGET else 4
    return Perception(food=full(food), mate=full(mate), prey=full(prey), prey_seen=full(seen),
                      threat=full(threat), threat_level=full(level, float))


def test_lone_prey_is_less_tempting_than_a_herd(make_world):
    """Holling tipo III: con una sola presa a la vista, cazar vale menos."""
    from especies.decision import utilities
    w = make_world(n=2, energy=[1.0, 50.0], reserve=100.0, instinct=1.0, meat_eff=1.0)
    ready = _none_ready(w)
    one = utilities(w, np.array([0]), _p(prey=1, seen=1), ready)[0, Action.HUNT]
    herd = utilities(w, np.array([0]), _p(prey=1, seen=4), ready)[0, Action.HUNT]
    assert 0 < one < herd


def _none_ready(w):
    return np.zeros(len(w.alive), dtype=bool)


def test_hungry_creature_with_food_in_sight_eats(make_world):
    w = make_world(n=1, energy=1.0, reserve=100.0, instinct=1.0)
    decide(w, np.array([0]), _p(food=42), _none_ready(w))
    assert w.action[0] == Action.EAT and w.target[0] == 42 and w.target_uid[0] == -1


def test_nothing_in_sight_means_explore(make_world):
    w = make_world(n=1, energy=1.0, reserve=100.0, instinct=1.0)
    decide(w, np.array([0]), _p(), np.ones(len(w.alive), dtype=bool))
    assert w.action[0] == Action.EXPLORE and w.target[0] == NO_TARGET


def test_hungry_carnivore_hunts_and_remembers_the_prey_uid(make_world):
    w = make_world(n=2, energy=[1.0, 50.0], reserve=100.0, instinct=1.0, meat_eff=[1.0, 0.0])
    decide(w, np.array([0]), _p(prey=1), _none_ready(w))
    assert w.action[0] == Action.HUNT and w.target[0] == 1
    assert w.target_uid[0] == w.uid[1]


def test_sated_predator_does_not_hunt(make_world):
    w = make_world(n=2, energy=[95.0, 50.0], reserve=100.0, instinct=1.0, meat_eff=1.0)
    decide(w, np.array([0]), _p(prey=1), _none_ready(w))
    assert w.action[0] != Action.HUNT


def test_strong_threat_makes_it_flee(make_world):
    w = make_world(n=2, energy=50.0, reserve=100.0, instinct=1.0)
    decide(w, np.array([0]), _p(food=42, threat=1, level=5.0), _none_ready(w))
    assert w.action[0] == Action.FLEE and w.target[0] == 1


def test_grazed_out_target_is_dropped(make_world):
    w = make_world(n=1, action=Action.EAT, target=7)
    w.terrain.grass[7] = w.terrain.leaves[7] = w.terrain.meat[7] = 0.0
    validate_targets(w, ready=_none_ready(w))
    assert w.action[0] == Action.EXPLORE and w.target[0] == NO_TARGET


def test_dead_or_unready_mate_is_dropped(make_world):
    w = make_world(n=3, action=Action.MATE, target=[1, 0, 1])
    w.target_uid[:3] = w.uid[[1, 0, 1]]
    w.alive[1] = False                        # la pareja de 0 y de 2 murió
    validate_targets(w, w.alive.copy())
    assert w.action[0] == Action.EXPLORE and w.action[2] == Action.EXPLORE


def test_prey_whose_slot_was_reused_is_dropped(make_world):
    w = make_world(n=2, action=Action.HUNT, target=[1, NO_TARGET], energy=10.0, reserve=100.0)
    w.target_uid[0] = w.uid[1]
    w.uid[1] = 10_000                         # murió y en su slot nació otra criatura
    validate_targets(w, _none_ready(w))
    assert w.action[0] == Action.EXPLORE


def test_chase_gives_up_after_its_limit(make_world):
    w = make_world(n=2, action=Action.HUNT, target=[1, NO_TARGET], energy=10.0, reserve=100.0)
    w.target_uid[0] = w.uid[1]
    w.hunt_ticks[0] = int(w.chase_limit[0]) + 1
    validate_targets(w, _none_ready(w))
    assert w.action[0] == Action.EXPLORE
