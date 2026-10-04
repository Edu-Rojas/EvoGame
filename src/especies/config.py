"""Carga de la configuración: TOML -> dataclasses inmutables.

Por qué así:
- Inmutable (frozen): nadie cambia un número a mitad de simulación por accidente.
- Falla con claves desconocidas: un typo en el TOML ("regrow_per_tik") revienta al
  cargar en vez de ignorarse en silencio y dejarte ajustando un número que no existe.
- Valida rangos al construir cada sección: un think_interval = 0 o una fracción de 1,5
  revientan al cargar con un mensaje claro, no a mitad de simulación.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .genes import GENE_MAX, GENE_MIN, Gene
from .terrain import Biome

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


# ---------- validación de rangos (cada dataclass la llama en __post_init__) ----------
def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _positive(obj: object, *names: str) -> None:
    for n in names:
        v = getattr(obj, n)
        _require(v > 0, f"{n}={v} debe ser > 0")


def _non_negative(obj: object, *names: str) -> None:
    for n in names:
        v = getattr(obj, n)
        _require(v >= 0, f"{n}={v} no puede ser negativo")


def _fraction(obj: object, *names: str) -> None:
    for n in names:
        v = getattr(obj, n)
        _require(0.0 <= v <= 1.0, f"{n}={v} debe estar entre 0 y 1")


def _color(name: str, c: tuple[int, ...]) -> None:
    _require(len(c) == 3 and all(0 <= v <= 255 for v in c),
             f"{name}={list(c)} debe ser [r, g, b] con valores de 0 a 255")


@dataclass(frozen=True)
class SimCfg:
    seed: int
    capacity: int
    think_interval: int

    def __post_init__(self) -> None:
        _positive(self, "capacity", "think_interval")


@dataclass(frozen=True)
class WorldCfg:
    width: float
    height: float

    def __post_init__(self) -> None:
        _positive(self, "width", "height")


@dataclass(frozen=True)
class TerrainCfg:
    cell_size: float
    noise_scale: float
    water_fraction: float
    mountain_fraction: float
    cold_fraction: float
    desert_fraction: float
    forest_fraction: float
    grass_per_cell: float
    min_to_target: float
    food_distance_penalty: float
    cold_size_exponent: float

    def __post_init__(self) -> None:
        _positive(self, "cell_size", "noise_scale")
        _fraction(self, "water_fraction", "mountain_fraction", "cold_fraction",
                  "desert_fraction", "forest_fraction")
        _non_negative(self, "grass_per_cell", "min_to_target", "food_distance_penalty",
                      "cold_size_exponent")


@dataclass(frozen=True)
class BiomeCfg:
    regrow: float
    max_grass: float
    speed: float
    visibility: float
    cost: float
    color_lush: tuple[int, int, int]
    color_bare: tuple[int, int, int]

    def __post_init__(self) -> None:
        _non_negative(self, "regrow", "max_grass", "visibility", "cost")
        _positive(self, "speed")
        _color("color_lush", self.color_lush)
        _color("color_bare", self.color_bare)


@dataclass(frozen=True)
class GenesCfg:
    budget: int
    mutation_sigma: float

    def __post_init__(self) -> None:
        _non_negative(self, "budget", "mutation_sigma")


@dataclass(frozen=True)
class BodyCfg:
    base_radius: float
    reserve_per_size: float
    appetite_base: float
    kleiber_exponent: float
    base_lifespan: int
    aging_extra_at_max: float
    senescence_start: float
    founder_energy: float
    bite: float

    def __post_init__(self) -> None:
        _positive(self, "base_radius", "reserve_per_size", "base_lifespan", "bite")
        _non_negative(self, "appetite_base", "kleiber_exponent", "aging_extra_at_max")
        _fraction(self, "founder_energy")
        # vitality() divide por (1 - senescence_start)
        _require(0.0 <= self.senescence_start < 1.0,
                 f"senescence_start={self.senescence_start} debe estar en [0, 1)")


@dataclass(frozen=True)
class MovementCfg:
    base_speed: float
    accel_speed_bonus: float
    steering: float
    cruise_fraction: float
    wander_turn: float
    move_cost: float
    veil_drag: float
    contact_distance: float
    separation_strength: float
    body_footprint: float

    def __post_init__(self) -> None:
        _positive(self, "base_speed", "body_footprint")
        _non_negative(self, "accel_speed_bonus", "wander_turn", "move_cost",
                      "contact_distance", "separation_strength")
        _fraction(self, "cruise_fraction", "veil_drag")
        # steering = 0 congela la velocidad: nadie se movería nunca
        _require(0.0 < self.steering <= 1.0, f"steering={self.steering} debe estar en (0, 1]")


@dataclass(frozen=True)
class DetectionCfg:
    base_radius: float
    max_multiplier: float
    appetite_extra_at_max: float
    k_neighbors: int

    def __post_init__(self) -> None:
        _positive(self, "base_radius", "k_neighbors")
        _non_negative(self, "appetite_extra_at_max")
        _require(self.max_multiplier >= 1,
                 f"max_multiplier={self.max_multiplier} debe ser >= 1")


@dataclass(frozen=True)
class DietCfg:
    exponent: float

    def __post_init__(self) -> None:
        _positive(self, "exponent")


@dataclass(frozen=True)
class ReproductionCfg:
    maturity_fraction: float
    ready_energy_fraction: float
    contribution: float
    max_litter: int
    cooldown: int
    spawn_spread: float

    def __post_init__(self) -> None:
        _fraction(self, "maturity_fraction", "ready_energy_fraction", "contribution")
        _positive(self, "max_litter")
        _non_negative(self, "cooldown", "spawn_spread")


@dataclass(frozen=True)
class BehaviorCfg:
    decision_noise: float
    explore_base: float
    eat_base: float
    instinct_sigma: float
    instinct_min: float
    instinct_max: float

    def __post_init__(self) -> None:
        _non_negative(self, "decision_noise", "explore_base", "eat_base", "instinct_sigma")
        _positive(self, "instinct_min")
        _require(self.instinct_min <= self.instinct_max,
                 f"instinct_min={self.instinct_min} es mayor que instinct_max={self.instinct_max}")


@dataclass(frozen=True)
class StatsCfg:
    generation_bonus: int


@dataclass(frozen=True)
class SpeciesCfg:
    name: str
    color: tuple[int, int, int]
    count: int
    genes: tuple[float, ...]  # ordenados según Gene

    def __post_init__(self) -> None:
        _positive(self, "count")
        _color("color", self.color)


@dataclass(frozen=True)
class Config:
    sim: SimCfg
    world: WorldCfg
    terrain: TerrainCfg
    biomes: Mapping[str, BiomeCfg]
    genes: GenesCfg
    body: BodyCfg
    movement: MovementCfg
    detection: DetectionCfg
    diet: DietCfg
    reproduction: ReproductionCfg
    behavior: BehaviorCfg
    stats: StatsCfg
    species: tuple[SpeciesCfg, ...]


SECTIONS: dict[str, type] = {
    "sim": SimCfg, "world": WorldCfg, "terrain": TerrainCfg, "genes": GenesCfg,
    "body": BodyCfg, "movement": MovementCfg, "detection": DetectionCfg,
    "diet": DietCfg, "reproduction": ReproductionCfg, "behavior": BehaviorCfg,
    "stats": StatsCfg,
}
SPECIES_KEYS = {"name", "color", "count", "genes"}


def _check_keys(data: Mapping[str, Any], expected: set[str], section: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"[{section}] claves desconocidas: {sorted(unknown)}")
    if missing:
        raise ValueError(f"[{section}] faltan claves: {sorted(missing)}")


def _build(cls: type, data: dict[str, Any], section: str):
    """Construye una dataclass exigiendo exactamente las claves esperadas y rangos válidos."""
    _check_keys(data, {f.name for f in fields(cls)}, section)
    try:
        return cls(**data)
    except ValueError as e:
        raise ValueError(f"[{section}] {e}") from None


def _build_species(raw: dict[str, Any], budget: int) -> SpeciesCfg:
    name = raw.get("name", "?")
    _check_keys(raw, SPECIES_KEYS, f"species {name}")
    genes_raw = raw["genes"]
    unknown = set(genes_raw) - {g.key for g in Gene}
    if unknown:
        raise ValueError(f"Especie {name}: genes desconocidos {sorted(unknown)}")
    values = tuple(float(genes_raw.get(g.key, GENE_MIN)) for g in Gene)
    for g, v in zip(Gene, values):
        if not GENE_MIN <= v <= GENE_MAX:
            raise ValueError(f"Especie {name}: {g.key}={v} fuera de 1..10")
    spent = sum(v - GENE_MIN for v in values)
    if spent > budget + 1e-9:
        raise ValueError(f"Especie {name}: gasta {spent:g} puntos y el presupuesto es {budget}")
    try:
        return SpeciesCfg(name=name, color=tuple(raw["color"]), count=int(raw["count"]),
                          genes=values)
    except ValueError as e:
        raise ValueError(f"Especie {name}: {e}") from None


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f))


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Valida y construye la Config a partir del TOML ya parseado."""
    _check_keys(raw, set(SECTIONS) | {"biomes", "species"}, "config")
    built = {name: _build(cls, raw[name], name) for name, cls in SECTIONS.items()}
    budget = built["genes"].budget
    species = tuple(_build_species(s, budget) for s in raw["species"])
    if not species:
        raise ValueError("La config no define ninguna [[species]]")
    cfg = Config(**built, biomes=_build_biomes(raw["biomes"]), species=species)
    _check_cross_sections(cfg)
    return cfg


def _check_cross_sections(cfg: Config) -> None:
    """Reglas que involucran más de una sección."""
    founders = sum(s.count for s in cfg.species)
    if founders > cfg.sim.capacity:
        raise ValueError(f"[sim] capacity={cfg.sim.capacity} no alcanza para los "
                         f"{founders} fundadores de las especies")
    for side in ("width", "height"):
        v = getattr(cfg.world, side)
        if v % cfg.terrain.cell_size != 0:
            raise ValueError(f"[world] {side}={v} debe ser múltiplo de "
                             f"[terrain] cell_size={cfg.terrain.cell_size}")


def _build_biomes(raw: dict[str, Any]) -> Mapping[str, BiomeCfg]:
    expected = {b.key for b in Biome}
    if set(raw) != expected:
        raise ValueError(f"[biomes] deben ser exactamente {sorted(expected)}, hay {sorted(raw)}")
    built = {}
    for name, data in raw.items():
        data = {k: tuple(v) if k.startswith("color_") else v for k, v in data.items()}
        built[name] = _build(BiomeCfg, data, f"biomes.{name}")
    return MappingProxyType(built)  # dict de solo lectura
