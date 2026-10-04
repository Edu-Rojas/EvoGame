"""Percepción: qué ve cada criatura que piensa este tick.

Comida: mira las celdas de pasto a su alrededor. Otras criaturas: KD-tree (SciPy,
en C) con mundo toroidal (boxsize); cada una mira solo sus K vecinos más cercanos,
así el costo por criatura queda acotado aunque haya miles.

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


def perceive(w: World, thinkers: np.ndarray, ready: np.ndarray, tree: cKDTree,
             tree_slots: np.ndarray) -> Perception:
    cfg = w.cfg
    k = cfg.detection.k_neighbors   # la config garantiza k >= 1
    pos = w.pos[thinkers]
    # El bioma donde está el que mira cambia cuánto ve (en el bosque se ve menos)
    vis = w.terrain.visibility[w.terrain.cell_of(pos)]
    r = (w.det_radius[thinkers] * vis)[:, None]
    r_max = float(r.max())

    # --- comida: la mejor celda de pasto dentro del radio (más pasto, menos lejos) ---
    food = _best_grass_cell(w, pos, r[:, 0])

    # --- pareja: el adulto listo más cercano de la misma especie ---
    d, j = tree.query(pos, k=k + 1, distance_upper_bound=r_max)  # +1 porque se ve a sí misma
    ok = j < len(tree_slots)
    nb = tree_slots[np.where(ok, j, 0)]
    ok &= nb != thinkers[:, None]
    ok &= d <= r
    ok &= w.species[nb] == w.species[thinkers][:, None]
    ok &= ready[nb]
    mate = _first_valid(ok, nb)

    return Perception(food=food, mate=mate)


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
