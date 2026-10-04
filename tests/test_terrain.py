import numpy as np
import pytest

from especies.terrain import Biome, create_terrain


def test_biomes_match_configured_fractions(cfg):
    t = create_terrain(cfg, np.random.default_rng(1))
    water = (t.biome == Biome.WATER).mean()
    assert water == pytest.approx(cfg.terrain.water_fraction, abs=0.01)
    assert set(np.unique(t.biome)) == {int(b) for b in Biome}


def test_grass_never_exceeds_its_maximum(cfg):
    t = create_terrain(cfg, np.random.default_rng(1))
    for _ in range(2000):
        t.regrow_step()
    assert (t.grass <= t.grass_max + 1e-9).all()


def test_cell_lookup_round_trips(cfg):
    t = create_terrain(cfg, np.random.default_rng(1))
    cells = np.arange(0, t.gw * t.gh, 97)
    assert np.array_equal(t.cell_of(t.center_of(cells)), cells)


def test_biome_keys_match_the_config(cfg):
    assert {b.key for b in Biome} == set(cfg.biomes)
