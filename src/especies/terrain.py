"""Terreno: biomas en una grilla y pasto que crece en cada celda.

Por qué una grilla y no plantas sueltas:
- El pasto "se ve": una pradera comida se pone pardita y vuelve a verdear. Ves las
  olas de pastoreo sin dibujar nada extra.
- Buscar comida es mirar celdas cercanas (índices de array), sin KD-tree de plantas.
- Cada bioma es una fila de parámetros (rebrote, velocidad, visibilidad, gasto), así
  que agregar un bioma es config, no código nuevo.

Generación: ruido suavizado (gaussiano con bordes envueltos, porque el mundo es un
toro) normalizado por cuantiles. Así cada umbral de la config significa directamente
"qué fracción del mapa": water_fraction = 0.07 es aprox. 7% de agua.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:   # config importa este módulo: solo para los tipos, sin ciclo en ejecución
    from .config import Config, TerrainCfg


class Biome(IntEnum):
    GRASSLAND = 0
    FOREST = 1
    DESERT = 2
    MOUNTAIN = 3
    TUNDRA = 4
    WATER = 5

    @property
    def key(self) -> str:
        """Nombre de la sección [biomes.<key>] en el TOML (en español, como lo ve el jugador)."""
        return _BIOME_KEYS[self]


_BIOME_KEYS = {
    Biome.GRASSLAND: "pradera",
    Biome.FOREST: "bosque",
    Biome.DESERT: "desierto",
    Biome.MOUNTAIN: "montana",
    Biome.TUNDRA: "helada",
    Biome.WATER: "agua",
}


@dataclass
class Terrain:
    cell_size: float
    gw: int                 # celdas a lo ancho
    gh: int                 # celdas a lo alto
    biome: np.ndarray       # [gh*gw] int8, índice plano = fila*gw + columna
    grass: np.ndarray       # [gh*gw] energía de pasto disponible
    grass_max: np.ndarray   # [gh*gw]
    regrow: np.ndarray      # [gh*gw] rebrote por tick
    speed: np.ndarray       # [gh*gw] multiplicador de velocidad
    visibility: np.ndarray  # [gh*gw] multiplicador del radio de detección
    cost: np.ndarray        # [gh*gw] multiplicador de gasto de energía
    leaves: np.ndarray      # [gh*gw] hojas altas: solo las alcanzan los grandes
    leaves_max: np.ndarray  # [gh*gw]
    leaf_regrow: np.ndarray # [gh*gw]
    meat: np.ndarray        # [gh*gw] carne de cadáveres (se pudre)
    meat_rot: float         # fracción de la carne que se pudre por tick
    meat_min: float         # debajo de esto la carne desaparece
    food_offsets: np.ndarray      # [O, 2] desplazamientos (dx, dy) en celdas
    food_offset_dist: np.ndarray  # [O] distancia de cada desplazamiento

    def cell_of(self, pos: np.ndarray) -> np.ndarray:
        cx = (pos[..., 0] // self.cell_size).astype(np.int64) % self.gw
        cy = (pos[..., 1] // self.cell_size).astype(np.int64) % self.gh
        return cy * self.gw + cx

    def center_of(self, cell: np.ndarray) -> np.ndarray:
        cy, cx = np.divmod(cell, self.gw)
        return np.stack([(cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size], axis=-1)

    def regrow_step(self) -> None:
        """Rebrota el pasto y las hojas; la carne se pudre y desaparece al quedar poca."""
        np.minimum(self.grass + self.regrow, self.grass_max, out=self.grass)
        np.minimum(self.leaves + self.leaf_regrow, self.leaves_max, out=self.leaves)
        self.meat *= 1.0 - self.meat_rot
        self.meat[self.meat < self.meat_min] = 0.0


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Ruido suave en [0, 1] con distribución uniforme (por cuantiles)."""
    noise = gaussian_filter(rng.random(shape), sigma=scale, mode="wrap")
    ranks = noise.ravel().argsort().argsort()
    return (ranks / (ranks.size - 1)).reshape(shape)


def generate_biomes(rng: np.random.Generator, gh: int, gw: int, t: TerrainCfg) -> np.ndarray:
    """Clasifica cada celda según elevación, humedad y temperatura."""
    elev = _smooth_field(rng, (gh, gw), t.noise_scale)
    moist = _smooth_field(rng, (gh, gw), t.noise_scale)
    temp = _smooth_field(rng, (gh, gw), t.noise_scale * t.temperature_scale)  # clima más suave

    b = np.full((gh, gw), Biome.GRASSLAND, dtype=np.int8)
    b[moist > 1 - t.forest_fraction] = Biome.FOREST
    b[moist < t.desert_fraction] = Biome.DESERT
    b[temp < t.cold_fraction] = Biome.TUNDRA
    b[elev > 1 - t.mountain_fraction] = Biome.MOUNTAIN
    b[elev < t.water_fraction] = Biome.WATER
    return b.ravel()


def create_terrain(cfg: Config, rng: np.random.Generator) -> Terrain:
    t = cfg.terrain
    gw = int(round(cfg.world.width / t.cell_size))
    gh = int(round(cfg.world.height / t.cell_size))
    biome = generate_biomes(rng, gh, gw, t)

    def per_cell(attr: str) -> np.ndarray:
        table = np.array([getattr(cfg.biomes[b.key], attr) for b in Biome], dtype=float)
        return table[biome]

    grass_max = per_cell("max_grass") * t.grass_per_cell
    leaves_max = per_cell("max_leaves") * cfg.leaves.per_cell
    visibility = per_cell("visibility")
    d = cfg.detection
    offsets, offset_dist = food_offsets(t.cell_size,
                                        d.base_radius * d.max_multiplier * float(visibility.max()))
    return Terrain(
        cell_size=t.cell_size, gw=gw, gh=gh, biome=biome,
        grass=grass_max * rng.random(biome.size),
        grass_max=grass_max,
        regrow=per_cell("regrow"),
        speed=per_cell("speed"),
        visibility=visibility,
        cost=per_cell("cost"),
        leaves=leaves_max.copy(),
        leaves_max=leaves_max,
        leaf_regrow=per_cell("leaf_regrow"),
        meat=np.zeros(biome.size),
        meat_rot=cfg.meat.rot,
        meat_min=cfg.meat.min_amount,
        food_offsets=offsets,
        food_offset_dist=offset_dist,
    )


def food_offsets(cell_size: float, r_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Desplazamientos (en celdas) dentro del radio de búsqueda más grande posible.

    Se calculan una vez; al buscar comida cada criatura los suma a su celda y luego
    filtra por su propio radio. Es el mismo truco del KD-tree: buscar con el máximo
    y filtrar con el individual.
    """
    n = int(np.ceil(r_max / cell_size))
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    dist = np.hypot(dx, dy) * cell_size
    keep = dist <= r_max
    return np.stack([dx[keep], dy[keep]], axis=1), dist[keep]
