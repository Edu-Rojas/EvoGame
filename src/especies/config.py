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
from collections.abc import Mapping
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .genes import GENE_MAX, GENE_MIN, Gene
from .terrain import Biome

# Viaja dentro del paquete: funciona igual instalado con pip que desde el repo
DEFAULT_CONFIG = files("especies") / "data" / "default.toml"


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
    _require(len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c),
             f"{name}={list(c)} debe ser [r, g, b] con valores de 0 a 255")


FOUNDER_SPAWNS = ("grouped", "uniform")


@dataclass(frozen=True)
class SimCfg:
    seed: int
    capacity: int
    think_interval: int
    founder_spawn: str        # "grouped": cada especie en su zona; "uniform": por todo el mapa
    founder_spread: float     # radio aproximado de la zona de cada especie (modo agrupado)

    def __post_init__(self) -> None:
        _positive(self, "capacity", "think_interval", "founder_spread")
        _require(self.founder_spawn in FOUNDER_SPAWNS,
                 f"founder_spawn={self.founder_spawn!r} debe ser uno de {list(FOUNDER_SPAWNS)}")


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
    temperature_scale: float
    grass_per_cell: float
    min_to_target: float
    min_to_stay: float
    food_distance_penalty: float
    cold_size_exponent: float

    def __post_init__(self) -> None:
        _positive(self, "cell_size", "noise_scale", "temperature_scale")
        _fraction(self, "water_fraction", "mountain_fraction", "cold_fraction",
                  "desert_fraction", "forest_fraction")
        _non_negative(self, "grass_per_cell", "min_to_target", "min_to_stay",
                      "food_distance_penalty", "cold_size_exponent")


@dataclass(frozen=True)
class BiomeCfg:
    regrow: float
    max_grass: float
    max_leaves: float
    leaf_regrow: float
    speed: float
    visibility: float
    cost: float
    color_lush: tuple[int, int, int]
    color_bare: tuple[int, int, int]

    def __post_init__(self) -> None:
        _non_negative(self, "regrow", "max_grass", "max_leaves", "leaf_regrow", "visibility", "cost")
        _positive(self, "speed")
        _color("color_lush", self.color_lush)
        _color("color_bare", self.color_bare)


@dataclass(frozen=True)
class GenesCfg:
    budget: int
    mutation_sigma: float
    inactive: tuple[str, ...]   # genes que todavía no tienen efecto: no mutan

    def __post_init__(self) -> None:
        _non_negative(self, "budget", "mutation_sigma")
        object.__setattr__(self, "inactive", tuple(self.inactive))   # inmutable aunque venga lista
        unknown = set(self.inactive) - {g.key for g in Gene}
        _require(not unknown, f"inactive tiene genes desconocidos: {sorted(unknown)}")


@dataclass(frozen=True)
class BodyCfg:
    base_radius: float
    reserve_per_size: float
    appetite_base: float
    kleiber_exponent: float
    base_lifespan: int
    aging_extra_at_max: float
    aging_curve: float
    senescence_start: float
    old_age_speed: float
    founder_energy: float
    founder_max_age: float
    bite: float

    def __post_init__(self) -> None:
        _positive(self, "base_radius", "reserve_per_size", "base_lifespan", "bite",
                  "aging_curve")
        _non_negative(self, "appetite_base", "kleiber_exponent", "aging_extra_at_max")
        _fraction(self, "founder_energy", "founder_max_age", "old_age_speed")
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
    home_range: float
    home_pull: float
    cover_seek: float

    def __post_init__(self) -> None:
        _positive(self, "base_speed", "body_footprint", "home_range")
        _fraction(self, "home_pull")
        _non_negative(self, "cover_seek")
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
    appetite_curve: float

    def __post_init__(self) -> None:
        _positive(self, "base_radius", "appetite_curve")
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
class LeavesCfg:
    per_cell: float
    reach_min_size: float
    reach_full_size: float

    def __post_init__(self) -> None:
        _non_negative(self, "per_cell", "reach_min_size")
        _require(self.reach_full_size > self.reach_min_size,
                 "reach_full_size debe ser mayor que reach_min_size")


@dataclass(frozen=True)
class CombatCfg:
    health_per_size: float
    health_regen: float
    bite_base: float
    bite_size_exponent: float
    weapons_herbivore: float
    weapons_carnivore: float
    bite_energy_cost: float
    retaliation: float
    max_prey_ratio: float
    predator_meat_eff: float
    idle_threat: float
    aggression_factor_min: float
    aggression_factor_max: float
    chase_ticks: int
    chase_min_mult: float
    chase_max_mult: float
    satiety_fraction: float
    prey_search_half: float
    newborn_health_energy: float
    newborn_health_min: float

    def __post_init__(self) -> None:
        _positive(self, "health_per_size", "bite_base", "max_prey_ratio", "chase_ticks",
                  "aggression_factor_min", "newborn_health_energy")
        _non_negative(self, "bite_size_exponent", "weapons_herbivore", "weapons_carnivore",
                      "bite_energy_cost", "retaliation", "chase_min_mult", "prey_search_half")
        _fraction(self, "health_regen", "predator_meat_eff", "satiety_fraction",
                  "newborn_health_min", "idle_threat")
        _require(self.aggression_factor_max >= self.aggression_factor_min,
                 "aggression_factor_max debe ser >= aggression_factor_min")
        _require(self.chase_max_mult >= self.chase_min_mult,
                 "chase_max_mult debe ser >= chase_min_mult")


@dataclass(frozen=True)
class MeatCfg:
    per_size: float
    energy_fraction: float
    rot: float
    min_amount: float

    def __post_init__(self) -> None:
        _non_negative(self, "per_size", "min_amount")
        _fraction(self, "energy_fraction", "rot")


@dataclass(frozen=True)
class DiseaseCfg:
    radius: float
    free_neighbors: int
    cost_per_neighbor: float

    def __post_init__(self) -> None:
        _positive(self, "radius")
        _non_negative(self, "free_neighbors", "cost_per_neighbor")


@dataclass(frozen=True)
class BehaviorCfg:
    decision_noise: float
    explore_base: float
    eat_base: float
    mate_base: float
    hunt_base: float
    flee_base: float
    instinct_sigma: float
    instinct_min: float
    instinct_max: float

    def __post_init__(self) -> None:
        _non_negative(self, "decision_noise", "explore_base", "eat_base", "mate_base",
                      "hunt_base", "flee_base", "instinct_sigma")
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
    leaves: LeavesCfg
    combat: CombatCfg
    meat: MeatCfg
    disease: DiseaseCfg
    behavior: BehaviorCfg
    stats: StatsCfg
    species: tuple[SpeciesCfg, ...]


SECTIONS: dict[str, type] = {
    "sim": SimCfg, "world": WorldCfg, "terrain": TerrainCfg, "genes": GenesCfg,
    "body": BodyCfg, "movement": MovementCfg, "detection": DetectionCfg,
    "diet": DietCfg, "reproduction": ReproductionCfg, "leaves": LeavesCfg,
    "combat": CombatCfg, "meat": MeatCfg, "disease": DiseaseCfg, "behavior": BehaviorCfg,
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


def _check_types(cls: type, data: Mapping[str, Any], section: str) -> None:
    """Enteros donde van enteros y números donde van números (TOML distingue 3 de 3.0;
    un 6000.5 en `capacity` no debe pasar ni truncarse en silencio)."""
    for f in fields(cls):
        v = data[f.name]
        if f.type == "int" and (not isinstance(v, int) or isinstance(v, bool)):
            raise ValueError(f"[{section}] {f.name}={v!r} debe ser un número entero")
        if f.type == "float" and (not isinstance(v, int | float) or isinstance(v, bool)):
            raise ValueError(f"[{section}] {f.name}={v!r} debe ser un número")
        if f.type == "str" and not isinstance(v, str):
            raise ValueError(f"[{section}] {f.name}={v!r} debe ser un texto")


def _build(cls: type, data: dict[str, Any], section: str):
    """Construye una dataclass exigiendo exactamente las claves esperadas, sus tipos y
    rangos válidos."""
    _check_keys(data, {f.name for f in fields(cls)}, section)
    _check_types(cls, data, section)
    try:
        return cls(**data)
    except ValueError as e:
        raise ValueError(f"[{section}] {e}") from None


def _build_species(raw: dict[str, Any], budget: int) -> SpeciesCfg:
    name = raw.get("name", "?")
    _check_keys(raw, SPECIES_KEYS, f"species {name}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Especie {name!r}: el nombre debe ser un texto no vacío")
    if not isinstance(raw["count"], int) or isinstance(raw["count"], bool):
        raise ValueError(f"Especie {name}: count={raw['count']!r} debe ser un número entero")
    genes_raw = raw["genes"]
    unknown = set(genes_raw) - {g.key for g in Gene}
    if unknown:
        raise ValueError(f"Especie {name}: genes desconocidos {sorted(unknown)}")
    bad = [k for k, v in genes_raw.items() if not isinstance(v, int | float) or isinstance(v, bool)]
    if bad:
        raise ValueError(f"Especie {name}: los genes {sorted(bad)} deben ser números")
    values = tuple(float(genes_raw.get(g.key, GENE_MIN)) for g in Gene)
    for g, v in zip(Gene, values, strict=True):
        if not GENE_MIN <= v <= GENE_MAX:
            raise ValueError(f"Especie {name}: {g.key}={v} fuera de 1..10")
    spent = sum(v - GENE_MIN for v in values)
    if spent > budget + 1e-9:
        raise ValueError(f"Especie {name}: gasta {spent:g} puntos y el presupuesto es {budget}")
    try:
        return SpeciesCfg(name=name, color=tuple(raw["color"]), count=raw["count"], genes=values)
    except ValueError as e:
        raise ValueError(f"Especie {name}: {e}") from None


def load_config(path: str | Path | None = None) -> Config:
    """Carga un TOML de config; sin `path`, la config por defecto del paquete."""
    source = Path(path) if path else DEFAULT_CONFIG
    with source.open("rb") as f:
        return config_from_dict(tomllib.load(f))


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Valida y construye la Config a partir del TOML ya parseado."""
    _check_keys(raw, set(SECTIONS) | {"biomes", "species"}, "config")
    built = {name: _build(cls, raw[name], name) for name, cls in SECTIONS.items()}
    budget = built["genes"].budget
    species = tuple(_build_species(s, budget) for s in raw["species"])
    if not species:
        raise ValueError("La config no define ninguna [[species]]")
    names = [s.name for s in species]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Nombres de especie repetidos: {dupes}")
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
