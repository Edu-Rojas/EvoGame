"""Percepción: qué ve cada criatura que piensa este tick.

Comida: mira las celdas de pasto a su alrededor.
Pareja: un KD-tree (SciPy, en C, mundo toroidal con boxsize) por especie, solo con
las criaturas listas para aparearse. Así nadie "pierde de vista" a su pareja por
estar rodeado de otra especie: buscar entre los K vecinos de cualquier especie
favorecía a la especie más numerosa, y el motor no puede elegir quién gana.

Detalle: `distance_upper_bound` de SciPy es UN número para todas las consultas,
así que buscamos con el radio máximo y después filtramos con el radio de cada una.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .state import NO_TARGET, World


@dataclass
class Perception:
    food: np.ndarray   # celda de pasto elegida o NO_TARGET
    mate: np.ndarray   # slot de pareja elegida o NO_TARGET


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

    # --- comida: la mejor celda de pasto dentro del radio (más pasto, menos lejos) ---
    food = _best_grass_cell(w, pos, r)
    return Perception(food=food, mate=_nearest_ready_mate(w, thinkers, pos, r, ready))


def _nearest_ready_mate(w: World, thinkers: np.ndarray, pos: np.ndarray, r: np.ndarray,
                        ready: np.ndarray) -> np.ndarray:
    """La pareja lista más cercana de la misma especie dentro del radio de cada una.

    Solo buscan las que están listas. El bucle es por especie (pocas), no por criatura.
    """
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


def _best_grass_cell(w: World, pos: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Para cada criatura, la celda que más conviene: pasto - penalización por distancia.

    Matriz [criaturas, desplazamientos]: cada fila son las celdas alrededor de una
    criatura. Todo vectorizado; el costo es criaturas x celdas del radio máximo.
    """
    t = w.terrain
    cfg = w.cfg.terrain
    cx = (pos[:, 0] // t.cell_size).astype(np.int64)
    cy = (pos[:, 1] // t.cell_size).astype(np.int64)
    ox, oy = t.food_offsets[:, 0], t.food_offsets[:, 1]
    cells = ((cy[:, None] + oy) % t.gh) * t.gw + (cx[:, None] + ox) % t.gw
    grass = t.grass[cells]
    dist = t.food_offset_dist[None, :]
    ok = (dist <= radius[:, None]) & (grass >= cfg.min_to_target)
    score = np.where(ok, grass - cfg.food_distance_penalty * dist, -np.inf)
    best = score.argmax(axis=1)
    picked = cells[np.arange(len(cells)), best]
    return np.where(ok.any(axis=1), picked, NO_TARGET)
