"""Estado del mundo: todas las criaturas como columnas de arrays (structure of arrays).

Por qué arrays y no un objeto por criatura:
- numpy opera sobre columnas enteras en C. "Gastar energía" para 5.000 bichos es
  UNA resta vectorizada, no 5.000 llamadas a un método de Python.
- Capacidad fija + máscara `alive`: no se crean ni destruyen arrays en cada
  nacimiento o muerte (eso sería lento). Un slot muerto se reutiliza.
- Cada criatura tiene además un `uid` único que nunca se reutiliza, para el linaje:
  el slot es "dónde vive en memoria", el uid es "quién es".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .genes import Gene, N_GENES, norm
from .geometry import wrap
from .terrain import Terrain, create_terrain


class Action:
    """Acciones de la v1a. Se usan como índices de columna (instintos, utilidades)."""
    EXPLORE = 0
    EAT = 1
    MATE = 2
    N = 3
    LABELS = ("explorar", "comer", "aparearse")


NO_TARGET = -1


@dataclass
class World:
    cfg: Config
    rng: np.random.Generator
    seed: int
    tick: int = 0
    next_uid: int = 0
    # --- criaturas (capacidad fija) ---
    alive: np.ndarray = field(init=False)
    uid: np.ndarray = field(init=False)
    parent_a: np.ndarray = field(init=False)
    parent_b: np.ndarray = field(init=False)
    species: np.ndarray = field(init=False)
    generation: np.ndarray = field(init=False)
    genes: np.ndarray = field(init=False)
    instinct: np.ndarray = field(init=False)
    pos: np.ndarray = field(init=False)
    vel: np.ndarray = field(init=False)
    heading: np.ndarray = field(init=False)
    energy: np.ndarray = field(init=False)
    age: np.ndarray = field(init=False)
    cooldown: np.ndarray = field(init=False)
    action: np.ndarray = field(init=False)
    target: np.ndarray = field(init=False)
    # --- rasgos derivados de los genes (se calculan una vez al nacer) ---
    radius: np.ndarray = field(init=False)
    reserve: np.ndarray = field(init=False)
    vmax: np.ndarray = field(init=False)
    det_radius: np.ndarray = field(init=False)
    appetite: np.ndarray = field(init=False)
    aging_rate: np.ndarray = field(init=False)
    plant_eff: np.ndarray = field(init=False)
    # --- terreno: biomas + pasto por celda ---
    terrain: Terrain = field(init=False)
    # --- contadores de eventos (para métricas) ---
    births_total: int = 0
    deaths_total: int = 0

    @property
    def size(self) -> np.ndarray:
        return np.array([self.cfg.world.width, self.cfg.world.height])

    def alive_idx(self) -> np.ndarray:
        return np.flatnonzero(self.alive)


def derive_traits(genes: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """Genome -> Traits: convierte genes (1..10) en números físicos.

    Es la única función que sabe "qué hace cada gen" con el cuerpo, así que
    retocar el balance de un gen es tocar aquí y en la config, no por todo el código.
    """
    b, m, d = cfg.body, cfg.movement, cfg.detection
    size = genes[:, Gene.SIZE]
    x_acc = norm(genes[:, Gene.ACCELERATION])
    x_det = norm(genes[:, Gene.DETECTION])
    x_diet = norm(genes[:, Gene.DIET])
    x_mate = norm(genes[:, Gene.MATING])
    x_size = norm(size)

    kleiber = size ** b.kleiber_exponent
    return {
        # radio: el área (2D) escala con la masa, así que el radio va con la raíz
        "radius": b.base_radius * np.sqrt(size),
        "reserve": b.reserve_per_size * size,
        # plumas largas frenan un poco (un adorno vistoso tiene costo)
        "vmax": m.base_speed * (1 + m.accel_speed_bonus * x_acc) * (1 - m.veil_drag * x_mate),
        "det_radius": d.base_radius * (1 + (d.max_multiplier - 1) * x_det),
        # Kleiber + costo de los ojos (proporcional al área que cubre: radio^2)
        "appetite": b.appetite_base * kleiber * (1 + d.appetite_extra_at_max * x_det ** 2),
        # los grandes envejecen más rápido: x1 / x1,18 / x1,6 para tamaño 1 / 5 / 10
        "aging_rate": 1 + b.aging_extra_at_max * x_size ** 1.5,
        "plant_eff": (1 - x_diet) ** cfg.diet.exponent,
    }


def create_world(cfg: Config, seed: int | None = None) -> World:
    """Mundo vacío + terreno + fundadores de cada especie."""
    if seed is None:
        seed = cfg.sim.seed
    if seed < 0:
        seed = int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)
    w = World(cfg=cfg, rng=rng, seed=seed)
    n = cfg.sim.capacity

    w.alive = np.zeros(n, dtype=bool)
    w.uid = np.full(n, -1, dtype=np.int64)
    w.parent_a = np.full(n, -1, dtype=np.int64)
    w.parent_b = np.full(n, -1, dtype=np.int64)
    w.species = np.zeros(n, dtype=np.int16)
    w.generation = np.zeros(n, dtype=np.int32)
    w.genes = np.ones((n, N_GENES))
    w.instinct = np.ones((n, Action.N))
    w.pos = np.zeros((n, 2))
    w.vel = np.zeros((n, 2))
    w.heading = np.zeros(n)
    w.energy = np.zeros(n)
    w.age = np.zeros(n)
    w.cooldown = np.zeros(n, dtype=np.int32)
    w.action = np.zeros(n, dtype=np.int8)
    w.target = np.full(n, NO_TARGET, dtype=np.int32)
    for name in ("radius", "reserve", "vmax", "det_radius", "appetite", "aging_rate", "plant_eff"):
        setattr(w, name, np.zeros(n))

    w.terrain = create_terrain(cfg, rng)

    for sp_id, sp in enumerate(cfg.species):
        genes = np.tile(np.array(sp.genes), (sp.count, 1))
        # pequeña variación inicial: sin variación no hay nada que seleccionar
        genes = np.clip(genes + rng.normal(0, cfg.genes.mutation_sigma, genes.shape), 1, 10)
        spawn(
            w,
            species=np.full(sp.count, sp_id),
            genes=genes,
            instinct=np.ones((sp.count, Action.N)),
            pos=rng.random((sp.count, 2)) * w.size,
            energy=None,  # se calcula con la reserva
            age=rng.random(sp.count) * cfg.body.base_lifespan * 0.3,  # edades mezcladas
            generation=np.zeros(sp.count, dtype=np.int32),
            parent_a=np.full(sp.count, -1),
            parent_b=np.full(sp.count, -1),
        )
    return w


def spawn(w: World, *, species, genes, instinct, pos, energy, age, generation,
          parent_a, parent_b) -> int:
    """Inserta un lote de criaturas en slots libres. Devuelve cuántas cupieron.

    Si no hay slots libres (tope duro de rendimiento), las que sobran no nacen.
    """
    free = np.flatnonzero(~w.alive)
    k = min(len(free), len(species))
    if k == 0:
        return 0
    s = free[:k]
    genes = np.asarray(genes)[:k]
    traits = derive_traits(genes, w.cfg)

    w.alive[s] = True
    w.uid[s] = np.arange(w.next_uid, w.next_uid + k)
    w.next_uid += k
    w.parent_a[s] = np.asarray(parent_a)[:k]
    w.parent_b[s] = np.asarray(parent_b)[:k]
    w.species[s] = np.asarray(species)[:k]
    w.generation[s] = np.asarray(generation)[:k]
    w.genes[s] = genes
    w.instinct[s] = np.asarray(instinct)[:k]
    w.pos[s] = wrap(np.asarray(pos)[:k], w.size)
    w.vel[s] = 0.0
    w.heading[s] = w.rng.random(k) * 2 * np.pi
    w.age[s] = np.asarray(age)[:k]
    w.cooldown[s] = 0
    w.action[s] = Action.EXPLORE
    w.target[s] = NO_TARGET
    for name, values in traits.items():
        getattr(w, name)[s] = values
    if energy is None:
        w.energy[s] = w.reserve[s] * w.cfg.body.founder_energy
    else:
        w.energy[s] = np.minimum(np.asarray(energy)[:k], w.reserve[s])
    return k
