import copy
import tomllib

import pytest

from especies.config import DEFAULT_CONFIG, config_from_dict, load_config


@pytest.fixture(scope="module")
def default_raw():
    with DEFAULT_CONFIG.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture
def raw(default_raw):
    """Copia del TOML por defecto ya parseado, para romperla en cada test."""
    return copy.deepcopy(default_raw)


def test_default_config_loads():
    cfg = load_config()
    assert len(cfg.species) > 0


def test_loads_config_from_a_path(tmp_path):
    p = tmp_path / "mine.toml"
    p.write_bytes(DEFAULT_CONFIG.read_bytes())
    assert load_config(p) == load_config()


def test_rejects_unknown_section(raw):
    raw["reproducion"] = {"cooldown": 80}            # typo de [reproduction]
    with pytest.raises(ValueError, match=r"\[config\] claves desconocidas.*reproducion"):
        config_from_dict(raw)


def test_rejects_missing_section(raw):
    del raw["diet"]
    with pytest.raises(ValueError, match=r"faltan claves.*diet"):
        config_from_dict(raw)


def test_rejects_unknown_key_in_section(raw):
    raw["terrain"]["noise_scal"] = raw["terrain"].pop("noise_scale")
    with pytest.raises(ValueError, match=r"\[terrain\] claves desconocidas"):
        config_from_dict(raw)


def test_rejects_unknown_species_key(raw):
    raw["species"][0]["colour"] = [1, 2, 3]
    with pytest.raises(ValueError, match=r"species Conejos\] claves desconocidas.*colour"):
        config_from_dict(raw)


def test_rejects_missing_species_key(raw):
    del raw["species"][0]["count"]
    with pytest.raises(ValueError, match=r"faltan claves.*count"):
        config_from_dict(raw)


def test_rejects_unknown_gene(raw):
    raw["species"][0]["genes"]["tamanio"] = 2
    with pytest.raises(ValueError, match="genes desconocidos"):
        config_from_dict(raw)


def test_rejects_species_over_budget(raw):
    raw["species"][0]["genes"]["deteccion"] = 10      # Conejos: 22 puntos, presupuesto 20
    raw["species"][0]["genes"]["apareamiento"] = 10
    with pytest.raises(ValueError, match="presupuesto"):
        config_from_dict(raw)


def test_rejects_zero_think_interval(raw):
    raw["sim"]["think_interval"] = 0                   # dividiría entre cero en step()
    with pytest.raises(ValueError, match=r"\[sim\] think_interval=0"):
        config_from_dict(raw)


def test_rejects_capacity_below_founders(raw):
    raw["sim"]["capacity"] = 100                       # hay 190 fundadores
    with pytest.raises(ValueError, match="no alcanza para los 190 fundadores"):
        config_from_dict(raw)


@pytest.mark.parametrize("section, key, value", [
    ("terrain", "water_fraction", 1.5),
    ("reproduction", "contribution", -0.1),
    ("movement", "steering", 0.0),
    ("movement", "steering", 1.2),
    ("body", "senescence_start", 1.0),
    ("detection", "max_multiplier", 0.5),
    ("behavior", "instinct_min", 5.0),                 # mayor que instinct_max
])
def test_rejects_out_of_range_values(raw, section, key, value):
    raw[section][key] = value
    with pytest.raises(ValueError, match=rf"\[{section}\] {key}"):
        config_from_dict(raw)


@pytest.mark.parametrize("section, key, value", [
    ("sim", "capacity", 6000.5),
    ("sim", "think_interval", 2.5),
    ("sim", "capacity", True),
    ("reproduction", "max_litter", "4"),
    ("movement", "steering", "0.25"),
])
def test_rejects_wrong_types(raw, section, key, value):
    raw[section][key] = value
    with pytest.raises(ValueError, match=rf"\[{section}\] {key}"):
        config_from_dict(raw)


def test_rejects_fractional_species_count(raw):
    raw["species"][0]["count"] = 80.9
    with pytest.raises(ValueError, match="count=80.9"):
        config_from_dict(raw)


def test_rejects_duplicate_species_names(raw):
    raw["species"][1]["name"] = raw["species"][0]["name"]
    with pytest.raises(ValueError, match="repetidos"):
        config_from_dict(raw)


def test_rejects_world_not_multiple_of_cell_size(raw):
    raw["world"]["width"] = 1610.0                     # cell_size = 20
    with pytest.raises(ValueError, match="múltiplo"):
        config_from_dict(raw)


def test_rejects_bad_biome_color(raw):
    raw["biomes"]["bosque"]["color_lush"] = [40, 95]
    with pytest.raises(ValueError, match=r"\[biomes.bosque\] color_lush"):
        config_from_dict(raw)


def test_rejects_bad_species_count(raw):
    raw["species"][1]["count"] = 0
    with pytest.raises(ValueError, match="Especie Elefantes: count=0"):
        config_from_dict(raw)
