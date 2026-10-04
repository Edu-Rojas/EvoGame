"""Percepción: qué ve cada criatura que piensa este tick.

Comida: mira las celdas alrededor y valora cada una por la energía que ELLA sacaría:
pasto y hojas según su eficiencia con plantas (las hojas, además, según su alcance),
carne según su eficiencia con carne.

Otras criaturas: KD-trees (SciPy, en C, mundo toroidal con boxsize) por especie. Así
nadie "pierde de vista" algo por estar rodeado de otra especie: buscar entre los K
vecinos de cualquier especie favorecía a la especie más numerosa, y el motor no puede
elegir quién gana. El bucle es por especie (pocas), nunca por criatura.
- Pareja: la lista más cercana de su especie (solo buscan las que están listas).
- Presa: la más cercana de otra especie que puede cazar (tamaño <= el suyo x
  max_prey_ratio). Solo buscan los que comen carne.
- Amenaza: la de otra especie que come carne, la puede cazar y tiene más "poder
  visible" relativo, pesado por cercanía. El poder sale de rasgos visibles (tamaño y
  dieta), nunca de stats ocultas.

Detalle: `distance_upper_bound` de SciPy es UN número para todas las consultas,
así que buscamos con el radio máximo y después filtramos con el radio de cada una.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .genes import Gene
from .state import NO_TARGET, Action, World

# Candidatos por especie que se revisan al buscar presa o amenaza (los K más cercanos
# de ESA especie; numérico, no balance)
NEIGHBORS_PER_SPECIES = 4


@dataclass
class Perception:
    food: np.ndarray          # celda con comida elegida o NO_TARGET
    mate: np.ndarray          # slot de pareja elegida o NO_TARGET
    prey: np.ndarray          # slot de presa elegida o NO_TARGET
    prey_seen: np.ndarray     # cuántas presas cazables ve (imagen de búsqueda, Holling III)
    threat: np.ndarray        # slot de la amenaza principal o NO_TARGET
    threat_level: np.ndarray  # 0..1: 0 sin amenaza, 1 = mucho más fuerte y encima


def _first_valid(ok: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Por fila, el primer candidato válido (los vecinos vienen ordenados por distancia)."""
    has = ok.any(axis=1)
    first = ok.argmax(axis=1)
    picked = candidates[np.arange(len(candidates)), first]
    return np.where(has, picked, NO_TARGET)


def perceive(w: World, thinkers: np.ndarray, ready: np.ndarray) -> Perception:
    pos = w.pos[thinkers]
    # El bioma donde está el que mira cambia cuánto ve (en el bosque se ve menos)
    vis = w.terrain.visibility[w.terrain.cell_of(pos)]
    r = w.det_radius[thinkers] * vis

    food = _best_food_cell(w, thinkers, pos, r)
    mate = _nearest_ready_mate(w, thinkers, pos, r, ready)
    prey, seen, threat, level = _prey_and_threats(w, thinkers, pos, r)
    return Perception(food=food, mate=mate, prey=prey, prey_seen=seen, threat=threat,
                      threat_level=level)


def power(w: World, slots: np.ndarray) -> np.ndarray:
    """Poder visible: lo que se nota a simple vista de cuánto pega y cuánto aguanta."""
    return w.bite_damage[slots] * w.max_health[slots]


def can_hunt(w: World, hunter: np.ndarray, prey: np.ndarray) -> np.ndarray:
    """Refugio por tamaño: solo se caza lo que no es mucho más grande que uno."""
    ratio = w.cfg.combat.max_prey_ratio
    return w.genes[prey, Gene.SIZE] <= w.genes[hunter, Gene.SIZE] * ratio


def _nearest_ready_mate(w: World, thinkers: np.ndarray, pos: np.ndarray, r: np.ndarray,
                        ready: np.ndarray) -> np.ndarray:
    """La pareja lista más cercana de la misma especie dentro del radio de cada una."""
    mate = np.full(len(thinkers), NO_TARGET, dtype=np.int64)
    alive = w.alive_idx()
    pool_all = alive[ready[alive]]
    looking = ready[thinkers]
    for sp in np.unique(w.species[thinkers[looking]]):
        pool = pool_all[w.species[pool_all] == sp]
        if len(pool) < 2:
            continue
        who = np.flatnonzero(looking & (w.species[thinkers] == sp))
        tree = cKDTree(w.pos[pool], boxsize=w.size)
        # k=2: la más cercana suele ser ella misma; la siguiente es la pareja
        d, j = tree.query(pos[who], k=2, distance_upper_bound=float(r[who].max()))
        found = j < len(pool)
        nb = pool[np.where(found, j, 0)]
        ok = found & (nb != thinkers[who][:, None]) & (d <= r[who][:, None])
        mate[who] = _first_valid(ok, nb)
    return mate


def _prey_and_threats(w: World, thinkers: np.ndarray, pos: np.ndarray, r: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = w.cfg.combat
    n = len(thinkers)
    prey = np.full(n, NO_TARGET, dtype=np.int64)
    prey_dist = np.full(n, np.inf)
    seen = np.zeros(n, dtype=np.int64)
    threat = np.full(n, NO_TARGET, dtype=np.int64)
    level = np.zeros(n)
    hunter = w.meat_eff[thinkers] >= c.predator_meat_eff
    my_power = power(w, thinkers) * w.aggression_factor[thinkers]
    alive = w.alive_idx()
    # cobertura: el bioma donde está el OTRO también cuenta (en el bosque se esconde,
    # en el desierto queda expuesto). Es el refugio de las presas.
    t = w.terrain
    cover = t.visibility[t.cell_of(w.pos)]
    reach = float(r.max() * t.visibility.max())
    for sp in np.unique(w.species[alive]):
        pool = alive[w.species[alive] == sp]
        who = np.flatnonzero(w.species[thinkers] != sp)
        if len(who) == 0:
            continue
        k = min(NEIGHBORS_PER_SPECIES, len(pool))
        tree = cKDTree(w.pos[pool], boxsize=w.size)
        d, j = tree.query(pos[who], k=k, distance_upper_bound=reach)
        d, j = d.reshape(len(who), k), j.reshape(len(who), k)      # k=1 devuelve 1D
        found = j < len(pool)
        nb = pool[np.where(found, j, 0)]
        me = thinkers[who][:, None]
        seen_at = r[who][:, None] * cover[nb]                       # hasta dónde lo ve
        near = found & (d <= seen_at)

        # presa: la más cercana que puede cazar
        ok = near & hunter[who][:, None] & can_hunt(w, me, nb)
        seen[who] += ok.sum(axis=1)
        has = ok.any(axis=1)
        first = ok.argmax(axis=1)
        dist = np.where(has, d[np.arange(len(who)), first], np.inf)
        better = dist < prey_dist[who]
        prey[who[better]] = nb[np.arange(len(who)), first][better]
        prey_dist[who[better]] = dist[better]

        # amenaza: come carne y me puede cazar. El nivel va de 0 a 1: 0 si no es más
        # fuerte que yo (contando mi agresividad), tiende a 1 si me supera de lejos;
        # pesa más mientras más cerca está, y un depredador que no está cazando (algo
        # visible: no viene corriendo) asusta menos
        danger = near & (w.meat_eff[nb] >= c.predator_meat_eff) & can_hunt(w, nb, me)
        ratio = power(w, nb) / my_power[who][:, None]
        strength = np.clip(1.0 - 1.0 / np.maximum(ratio, 1e-9), 0.0, 1.0)
        closeness = np.clip(1.0 - d / seen_at, 0.0, 1.0)
        # lo que se ve: viene por mí (máximo), persigue a otro (la mitad) o no caza
        chasing = w.action[nb] == Action.HUNT
        after_me = chasing & (w.target[nb] == me)
        calm = np.where(after_me, 1.0, np.where(chasing, 0.5, c.idle_threat))
        lvl = np.where(danger, strength * closeness * calm, 0.0)
        best = lvl.argmax(axis=1)
        top = lvl[np.arange(len(who)), best]
        stronger = top > level[who]
        threat[who[stronger]] = nb[np.arange(len(who)), best][stronger]
        level[who[stronger]] = top[stronger]
    return prey, seen, threat, level


def food_value(w: World, slots: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Energía que cada criatura sacaría de cada celda (cells: [n, ...] de índices)."""
    t = w.terrain
    shape = (len(slots),) + (1,) * (cells.ndim - 1)
    plant = w.plant_eff[slots].reshape(shape)
    reach = w.leaf_reach[slots].reshape(shape)
    meat = w.meat_eff[slots].reshape(shape)
    return t.grass[cells] * plant + t.leaves[cells] * plant * reach + t.meat[cells] * meat


def travel_cost(w: World, slots: np.ndarray) -> np.ndarray:
    """Energía que le cuesta a cada criatura recorrer una unidad de distancia a toda
    velocidad: (apetito + costo de moverse) / velocidad."""
    m, b = w.cfg.movement, w.cfg.body
    size = w.genes[slots, Gene.SIZE]
    move = m.move_cost * size ** b.kleiber_exponent * (w.vmax[slots] / m.base_speed) ** 2
    return (w.appetite[slots] + move) / np.maximum(w.vmax[slots], 1e-9)


def _best_food_cell(w: World, slots: np.ndarray, pos: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Para cada criatura, la celda que más conviene: la energía que sacaría (sin pasar
    de lo que le cabe) menos la energía del viaje.

    Con un costo de viaje real (y no un número chico fijo), un animal no abandona la
    celda donde está comiendo por otra apenas mejor que queda lejos.
    Matriz [criaturas, desplazamientos]: cada fila son las celdas alrededor de una
    criatura. Todo vectorizado; el costo es criaturas x celdas del radio máximo.
    """
    t = w.terrain
    cfg = w.cfg.terrain
    cx = (pos[:, 0] // t.cell_size).astype(np.int64)
    cy = (pos[:, 1] // t.cell_size).astype(np.int64)
    ox, oy = t.food_offsets[:, 0], t.food_offsets[:, 1]
    cells = ((cy[:, None] + oy) % t.gh) * t.gw + (cx[:, None] + ox) % t.gw
    value = food_value(w, slots, cells)
    room = np.maximum(w.reserve[slots] - w.energy[slots], 0.0)[:, None]
    dist = t.food_offset_dist[None, :]
    ok = (dist <= radius[:, None]) & (value >= cfg.min_to_target)
    cost = cfg.food_distance_penalty * travel_cost(w, slots)[:, None] * dist
    score = np.where(ok, np.minimum(value, room) - cost, -np.inf)
    best = score.argmax(axis=1)
    picked = cells[np.arange(len(cells)), best]
    return np.where(ok.any(axis=1), picked, NO_TARGET)
