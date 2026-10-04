import numpy as np
import pytest

from especies.combat import fight, heal
from especies.genes import Gene
from especies.metabolism import die, eat
from especies.state import Action, DeathCause, create_world
from especies.step import run


def _attack(make_world, hunters: int, prey_size: float = 2.0, hunter_size: float = 4.0):
    """`hunters` cazadores en contacto con una presa (slot = hunters)."""
    n = hunters + 1
    pos = [[500.0, 500.0]] * n
    w = make_world(n=n, pos=pos, species=[1] * hunters + [0], energy=50.0, reserve=100.0,
                   action=[Action.HUNT] * hunters + [Action.EXPLORE],
                   target=[hunters] * hunters + [-1])
    w.genes[:n, Gene.SIZE] = [hunter_size] * hunters + [prey_size]
    w.target_uid[:hunters] = w.uid[hunters]
    w.bite_damage[:hunters] = 3.0
    w.retaliation[hunters] = 0.0
    w.health[hunters] = 100.0
    return w, hunters


def test_damage_from_several_attackers_adds_up(make_world):
    w, prey = _attack(make_world, hunters=3)
    fight(w)
    assert w.health[prey] == pytest.approx(100.0 - 3 * 3.0)


def test_predator_does_not_bite_prey_beyond_the_size_refuge(make_world):
    w, prey = _attack(make_world, hunters=1, prey_size=9.0, hunter_size=4.0)
    fight(w)
    assert w.health[prey] == 100.0


def test_docile_prey_does_not_hit_back_but_aggressive_one_does(make_world):
    w, prey = _attack(make_world, hunters=1)
    h0 = w.health[0]
    fight(w)
    assert w.health[0] == h0
    w.retaliation[prey] = 0.5
    w.bite_damage[prey] = 4.0
    fight(w)
    assert w.health[0] == pytest.approx(h0 - 2.0)


def test_kill_credits_the_last_attacker_and_leaves_meat(make_world):
    w, prey = _attack(make_world, hunters=2)
    w.health[prey] = 1.0
    cell = w.terrain.cell_of(w.pos[prey])
    w.terrain.meat[cell] = 0.0
    fight(w)
    killer_uid = w.last_hitter[prey]
    die(w)
    assert not w.alive[prey]
    killer = np.flatnonzero(w.uid == killer_uid)[0]
    assert w.kills[killer] == 1
    assert w.deaths_by_cause[0, DeathCause.PREDATION] == 1
    assert w.terrain.meat[cell] > 0


def test_meat_rots_and_disappears(cfg):
    w = create_world(cfg, seed=0)
    w.terrain.meat[:] = 0.0
    w.terrain.meat[10] = 5.0
    for _ in range(5000):
        w.terrain.regrow_step()
    assert w.terrain.meat[10] == 0.0


def test_only_large_creatures_reach_tall_leaves(make_world):
    w = make_world(n=2, energy=0.0, reserve=500.0, radius=3.0, action=Action.EAT,
                   plant_eff=1.0, meat_eff=0.0)
    forest = np.flatnonzero(w.terrain.leaves_max > 0)[0]
    w.pos[:2] = w.terrain.center_of(np.array([forest, forest]))
    w.target[:2] = forest
    w.genes[:2, Gene.SIZE] = [2.0, 9.0]
    w.leaf_reach[:2] = [0.0, 1.0]
    w.terrain.grass[forest] = 0.0
    w.terrain.leaves[forest] = 20.0
    eat(w)
    assert w.energy[0] == 0.0 and w.energy[1] > 0.0


def test_newborns_with_little_energy_are_frail(cfg):
    from especies.state import spawn
    w = create_world(cfg, seed=0)
    w.alive[:] = False
    spawn(w, species=[0, 0], genes=np.ones((2, 7)), instinct=np.ones((2, 5)),
          pos=[[1.0, 1.0], [2.0, 2.0]], energy=[1.0, 1.0], age=[0, 0], generation=[1, 1],
          parent_a=[-1, -1], parent_b=[-1, -1], health_frac=[0.3, 1.0])
    s = np.flatnonzero(w.alive)
    assert w.health[s[0]] == pytest.approx(0.3 * w.max_health[s[0]])


def test_health_regenerates_up_to_the_maximum(make_world):
    w = make_world(n=1)
    w.health[0] = w.max_health[0] * 0.5
    for _ in range(2000):
        heal(w)
    assert w.health[0] == pytest.approx(w.max_health[0])


def test_hunting_is_deterministic(cfg):
    a, b = create_world(cfg, seed=9), create_world(cfg, seed=9)
    run(a, 400)
    run(b, 400)
    assert a.deaths_by_cause[:, DeathCause.PREDATION].sum() > 0
    assert np.array_equal(a.deaths_by_cause, b.deaths_by_cause)
    assert np.array_equal(a.pos, b.pos)
