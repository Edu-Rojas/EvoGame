import numpy as np
import pytest

from especies.genes import Gene, crossover, mutate, mutation_sigmas, stochastic_round
from especies.state import derive_traits


def test_crossover_only_takes_parent_values():
    rng = np.random.default_rng(0)
    a = np.full((500, 7), 1.0)
    b = np.full((500, 7), 10.0)
    child = crossover(a, b, rng)
    assert set(np.unique(child)) == {1.0, 10.0}
    # aprox. mitad y mitad: no hay sesgo hacia un padre
    assert 0.45 < (child == 1.0).mean() < 0.55


def test_mutation_stays_in_range():
    rng = np.random.default_rng(0)
    v = mutate(np.full(10_000, 9.9), sigma=1.0, lo=1, hi=10, rng=rng)
    assert v.min() >= 1 and v.max() <= 10


def test_inactive_genes_do_not_mutate():
    rng = np.random.default_rng(0)
    sigmas = mutation_sigmas(0.3, ("agresividad", "camuflaje"))
    v = mutate(np.ones((1000, len(Gene))), sigmas, lo=1, hi=10, rng=rng)
    assert (v[:, Gene.AGGRESSION] == 1.0).all() and (v[:, Gene.CAMOUFLAGE] == 1.0).all()
    assert v[:, Gene.SIZE].std() > 0


def test_inactive_genes_stay_put_across_generations(cfg):
    from especies.state import create_world
    from especies.step import run
    w = create_world(cfg, seed=3)
    run(w, 800)
    a = w.alive & (w.generation > 0)
    assert a.any()
    assert (w.genes[a, Gene.AGGRESSION] == 1.0).all()


def test_stochastic_round_preserves_mean():
    rng = np.random.default_rng(0)
    r = stochastic_round(np.full(100_000, 2.3), rng)
    assert set(np.unique(r)) == {2, 3}
    assert r.mean() == pytest.approx(2.3, abs=0.01)


def test_kleiber_basal_cost(cfg):
    genes = np.ones((2, 7))
    genes[1, Gene.SIZE] = 10
    t = derive_traits(genes, cfg)
    assert t["appetite"][1] / t["appetite"][0] == pytest.approx(10 ** 0.75)
    assert t["radius"][1] / t["radius"][0] == pytest.approx(10 ** 0.5)
    assert t["aging_rate"][1] == pytest.approx(1 + cfg.body.aging_extra_at_max)


def test_diet_efficiency_extremes(cfg):
    genes = np.ones((2, 7))
    genes[1, Gene.DIET] = 10
    t = derive_traits(genes, cfg)
    assert t["plant_eff"][0] == pytest.approx(1.0)
    assert t["plant_eff"][1] == pytest.approx(0.0)


def test_biomes_match_configured_fractions(cfg):
    from especies.terrain import Biome, create_terrain
    t = create_terrain(cfg, np.random.default_rng(1))
    water = (t.biome == Biome.WATER).mean()
    assert water == pytest.approx(cfg.terrain.water_fraction, abs=0.01)
    assert set(np.unique(t.biome)) == set(int(b) for b in Biome)


def test_bergmann_cold_costs_less_for_large_bodies(cfg):
    from especies.metabolism import spend
    from especies.state import create_world
    from especies.terrain import Biome
    w = create_world(cfg, seed=0)
    cold = np.flatnonzero(w.terrain.biome == Biome.TUNDRA)[0]
    s = w.alive_idx()[:2]
    w.alive[:] = False
    w.alive[s] = True
    w.pos[s] = w.terrain.center_of(np.array([cold, cold]))
    w.vel[s] = 0.0
    w.genes[s, Gene.SIZE] = [1.0, 10.0]
    w.appetite[s] = 1.0             # mismo apetito: solo comparamos el extra por frío
    w.energy[s] = 100.0
    spend(w)
    spent = 100.0 - w.energy[s]
    assert spent[1] < spent[0]
