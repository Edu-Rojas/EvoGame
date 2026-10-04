"""Carga de la configuración: TOML -> dataclasses inmutables.

Por qué así:
- Inmutable (frozen): nadie cambia un número a mitad de simulación por accidente.
- Falla con claves desconocidas: un typo en el TOML ("regrow_per_tik") revienta al
  cargar en vez de ignorarse en silencio y dejarte ajustando un número que no existe.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .genes import Gene
from .terrain import Biome

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class SimCfg:
    seed: int
    capacity: int
    think_interval: int


@dataclass(frozen=True)
class WorldCfg:
    width: float
    height: float


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


@dataclass(frozen=True)
class BiomeCfg:
    regrow: float
    max_grass: float
    speed: float
    visibility: float
    cost: float
    color_lush: tuple[int, int, int]
    color_bare: tuple[int, int, int]


@dataclass(frozen=True)
class GenesCfg:
    budget: int
    mutation_sigma: float


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


@dataclass(frozen=True)
class DetectionCfg:
    base_radius: float
    max_multiplier: float
    appetite_extra_at_max: float
    k_neighbors: int


@dataclass(frozen=True)
class DietCfg:
    exponent: float


@dataclass(frozen=True)
class ReproductionCfg:
    maturity_fraction: float
    ready_energy_fraction: float
    contribution: float
    max_litter: int
    cooldown: int
    spawn_spread: float


@dataclass(frozen=True)
class BehaviorCfg:
    decision_noise: float
    explore_base: float
    eat_base: float
    instinct_sigma: float
    instinct_min: float
    instinct_max: float


@dataclass(frozen=True)
class StatsCfg:
    generation_bonus: int


@dataclass(frozen=True)
class SpeciesCfg:
    name: str
    color: tuple[int, int, int]
    count: int
    genes: tuple[float, ...]  # ordenados según Gene


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


def _build(cls: type, data: dict[str, Any], section: str):
    """Construye una dataclass exigiendo exactamente las claves esperadas."""
    expected = {f.name for f in fields(cls)}
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"[{section}] claves desconocidas: {sorted(unknown)}")
    if missing:
        raise ValueError(f"[{section}] faltan claves: {sorted(missing)}")
    return cls(**data)


def _build_species(raw: dict[str, Any], budget: int) -> SpeciesCfg:
    name = raw["name"]
    genes_raw = raw["genes"]
    unknown = set(genes_raw) - {g.key for g in Gene}
    if unknown:
        raise ValueError(f"Especie {name}: genes desconocidos {sorted(unknown)}")
    values = tuple(float(genes_raw.get(g.key, 1.0)) for g in Gene)
    for g, v in zip(Gene, values):
        if not 1.0 <= v <= 10.0:
            raise ValueError(f"Especie {name}: {g.key}={v} fuera de 1..10")
    spent = sum(v - 1.0 for v in values)
    if spent > budget + 1e-9:
        raise ValueError(f"Especie {name}: gasta {spent:g} puntos y el presupuesto es {budget}")
    return SpeciesCfg(name=name, color=tuple(raw["color"]), count=int(raw["count"]), genes=values)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sections = {
        "sim": SimCfg, "world": WorldCfg, "terrain": TerrainCfg, "genes": GenesCfg,
        "body": BodyCfg, "movement": MovementCfg, "detection": DetectionCfg,
        "diet": DietCfg, "reproduction": ReproductionCfg, "behavior": BehaviorCfg,
        "stats": StatsCfg,
    }
    built = {name: _build(cls, raw[name], name) for name, cls in sections.items()}
    budget = built["genes"].budget
    species = tuple(_build_species(s, budget) for s in raw.get("species", []))
    if not species:
        raise ValueError("La config no define ninguna [[species]]")
    return Config(**built, biomes=_build_biomes(raw.get("biomes", {})), species=species)


def _build_biomes(raw: dict[str, Any]) -> Mapping[str, BiomeCfg]:
    expected = {b.key for b in Biome}
    if set(raw) != expected:
        raise ValueError(f"[biomes] deben ser exactamente {sorted(expected)}, hay {sorted(raw)}")
    built = {}
    for name, data in raw.items():
        data = dict(data)
        data["color_lush"] = tuple(data["color_lush"])
        data["color_bare"] = tuple(data["color_bare"])
        built[name] = _build(BiomeCfg, data, f"biomes.{name}")
    return MappingProxyType(built)  # dict de solo lectura
