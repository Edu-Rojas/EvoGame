"""Movimiento, deterioro por vejez y separación de cuerpos."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import torus_delta, wrap
from .state import Action, NO_TARGET, World


def vitality(w: World, idx: np.ndarray) -> np.ndarray:
    """1 en plenitud; baja linealmente hasta 0 en el último tramo de la vida."""
    b = w.cfg.body
    life_frac = w.age[idx] / b.base_lifespan
    decline = (life_frac - b.senescence_start) / (1 - b.senescence_start)
    return 1.0 - np.clip(decline, 0.0, 1.0)


def move(w: World) -> None:
    """Calcula la velocidad deseada según la acción y gira suave hacia ella."""
    m = w.cfg.movement
    a = w.alive_idx()
    if len(a) == 0:
        return
    act = w.action[a]
    tgt = w.target[a]
    vmax = w.vmax[a] * (0.5 + 0.5 * vitality(w, a))  # los viejos van más lento
    vmax = vmax * w.terrain.speed[w.terrain.cell_of(w.pos[a])]  # bosque, agua, montaña frenan

    # Explorar: paseo aleatorio suave (el rumbo cambia un poco cada tick)
    explore = act == Action.EXPLORE
    w.heading[a[explore]] += w.rng.normal(0.0, m.wander_turn, explore.sum())
    direction = np.stack([np.cos(w.heading[a]), np.sin(w.heading[a])], axis=1)
    speed = np.where(explore, m.cruise_fraction * vmax, vmax)

    # Ir hacia un objetivo (celda de pasto o pareja), frenando al llegar
    has_tgt = (tgt != NO_TARGET) & ~explore
    if has_tgt.any():
        i = np.flatnonzero(has_tgt)
        t = tgt[i]
        is_food = act[i] == Action.EAT
        target_pos = np.empty((len(i), 2))
        target_pos[is_food] = w.terrain.center_of(t[is_food])  # objetivo = celda de pasto
        target_pos[~is_food] = w.pos[t[~is_food]]          # objetivo = otra criatura
        delta = torus_delta(w.pos[a[i]], target_pos, w.size)
        dist = np.linalg.norm(delta, axis=1)
        safe = np.maximum(dist, 1e-9)[:, None]
        direction[i] = delta / safe
        speed[i] = np.minimum(vmax[i], dist * 0.5)

    desired = direction * speed[:, None]
    w.vel[a] += m.steering * (desired - w.vel[a])
    w.pos[a] = wrap(w.pos[a] + w.vel[a], w.size)

    moving = np.linalg.norm(w.vel[a], axis=1) > 1e-3
    targeted = has_tgt & moving
    w.heading[a[targeted]] = np.arctan2(w.vel[a[targeted], 1], w.vel[a[targeted], 0])


def separate(w: World) -> None:
    """Dos cuerpos no ocupan el mismo lugar: si se pisan, se empujan.

    Con el KD-tree cada criatura mira sus 7 vecinos más cercanos y se aleja de los
    que la tocan, en proporción a cuánto se superponen. Es un empuje suave (no una
    física exacta de choques), suficiente para que no se amontonen en la comida.
    """
    m = w.cfg.movement
    a = w.alive_idx()
    if len(a) < 2 or m.separation_strength <= 0:
        return
    pos = w.pos[a]
    rad = w.radius[a] * m.body_footprint   # el cuerpo es más largo que la cabeza
    tree = cKDTree(pos, boxsize=w.size)
    d, j = tree.query(pos, k=8, distance_upper_bound=2 * rad.max())
    valid = (j < len(a)) & (j != np.arange(len(a))[:, None])
    jc = np.where(valid, j, 0)
    min_d = rad[:, None] + rad[jc]
    overlap = np.where(valid & (d < min_d), min_d - d, 0.0)
    if not overlap.any():
        return
    away = torus_delta(pos[jc], pos[:, None, :], w.size)          # del vecino hacia mí
    dist = np.linalg.norm(away, axis=2, keepdims=True)
    away = np.where(dist > 1e-9, away / np.maximum(dist, 1e-9), np.array([1.0, 0.0]))
    push = (away * overlap[..., None]).sum(axis=1) * 0.5 * m.separation_strength
    w.pos[a] = wrap(pos + push, w.size)
