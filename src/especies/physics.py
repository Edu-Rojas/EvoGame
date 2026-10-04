"""Movimiento, deterioro por vejez y separación de cuerpos."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import torus_delta, wrap
from .state import NO_TARGET, Action, World

# Al acercarse al objetivo la velocidad deseada es esta fracción de la distancia que
# falta: frena suave en vez de pasarse (numérica, no balance)
ARRIVAL_SLOWDOWN = 0.5
# Vecinos que revisa cada cuerpo al separarse (incluida ella misma)
SEPARATION_NEIGHBORS = 8


def vitality(w: World, idx: np.ndarray) -> np.ndarray:
    """1 en plenitud; baja linealmente hasta 0 en el último tramo de la vida."""
    b = w.cfg.body
    life_frac = w.age[idx] / b.base_lifespan
    decline = (life_frac - b.senescence_start) / (1 - b.senescence_start)
    return 1.0 - np.clip(decline, 0.0, 1.0)


def _pull_home(w: World, explorers: np.ndarray) -> None:
    """Filopatría: al explorar lejos de donde nació, el rumbo se tuerce hacia allá.

    Crea poblaciones locales (estructura espacial), lo que por sí solo favorece la
    coexistencia. Depende solo de la distancia, nunca de la especie.
    """
    m = w.cfg.movement
    if len(explorers) == 0 or m.home_pull <= 0:
        return
    to_home = torus_delta(w.pos[explorers], w.home[explorers], w.size)
    dist = np.linalg.norm(to_home, axis=1)
    pull = m.home_pull * np.clip((dist - m.home_range) / m.home_range, 0.0, 1.0)
    far = pull > 0
    if not far.any():
        return
    e = explorers[far]
    h = np.stack([np.cos(w.heading[e]), np.sin(w.heading[e])], axis=1)
    home_dir = to_home[far] / dist[far, None]
    blended = h * (1 - pull[far, None]) + home_dir * pull[far, None]
    # si la casa quedaba justo detrás, las dos direcciones se anulan: va directo a casa
    cancel = np.linalg.norm(blended, axis=1) < 1e-6
    blended[cancel] = home_dir[cancel]
    w.heading[e] = np.arctan2(blended[:, 1], blended[:, 0])


# Celdas que mira una presa al buscar dónde esconderse (radio en celdas; geometría)
COVER_SEARCH_CELLS = 3
_COVER_OFFSETS = np.array([(dx, dy) for dx in range(-COVER_SEARCH_CELLS, COVER_SEARCH_CELLS + 1)
                           for dy in range(-COVER_SEARCH_CELLS, COVER_SEARCH_CELLS + 1)
                           if 0 < dx * dx + dy * dy <= COVER_SEARCH_CELLS ** 2])


def _toward_cover(w: World, fleeing: np.ndarray, away: np.ndarray) -> np.ndarray:
    """Dirección de huida torcida hacia la celda cercana con más cobertura (menos
    visibilidad) que no quede hacia la amenaza. Si ya está en la mejor, sigue derecho."""
    t = w.terrain
    pull = w.cfg.movement.cover_seek
    if pull <= 0:
        return away
    cx = (w.pos[fleeing, 0] // t.cell_size).astype(np.int64)
    cy = (w.pos[fleeing, 1] // t.cell_size).astype(np.int64)
    cells = ((cy[:, None] + _COVER_OFFSETS[:, 1]) % t.gh) * t.gw + (cx[:, None] + _COVER_OFFSETS[:, 0]) % t.gw
    dirs = _COVER_OFFSETS / np.linalg.norm(_COVER_OFFSETS, axis=1, keepdims=True)
    ahead = away @ dirs.T > -0.2                      # no correr hacia el depredador
    vis = np.where(ahead, t.visibility[cells], np.inf)
    best = vis.argmin(axis=1)
    here = t.visibility[t.cell_of(w.pos[fleeing])]
    better = vis[np.arange(len(fleeing)), best] < here
    blended = away + pull * dirs[best] * better[:, None]
    return blended / np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-9)


def move(w: World) -> None:
    """Calcula la velocidad deseada según la acción y gira suave hacia ella."""
    m = w.cfg.movement
    a = w.alive_idx()
    if len(a) == 0:
        return
    act = w.action[a]
    tgt = w.target[a]
    slow = w.cfg.body.old_age_speed                      # los viejos van más lento
    vmax = w.vmax[a] * (slow + (1 - slow) * vitality(w, a))
    vmax = vmax * w.terrain.speed[w.terrain.cell_of(w.pos[a])]  # bosque, agua, montaña frenan

    # Explorar: paseo aleatorio suave (el rumbo cambia un poco cada tick)
    explore = act == Action.EXPLORE
    w.heading[a[explore]] += w.rng.normal(0.0, m.wander_turn, explore.sum())
    _pull_home(w, a[explore])
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
        speed[i] = np.minimum(vmax[i], dist * ARRIVAL_SLOWDOWN)
        # huir: en sentido contrario a la amenaza y a toda velocidad, sin frenar, y
        # torciendo hacia la cobertura más cercana (el bosque esconde: refugio)
        flee = act[i] == Action.FLEE
        direction[i[flee]] *= -1.0
        speed[i[flee]] = vmax[i[flee]]
        if flee.any():
            direction[i[flee]] = _toward_cover(w, a[i[flee]], direction[i[flee]])

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
    d, j = tree.query(pos, k=SEPARATION_NEIGHBORS, distance_upper_bound=2 * rad.max())
    valid = (j < len(a)) & (j != np.arange(len(a))[:, None])
    jc = np.where(valid, j, 0)
    min_d = rad[:, None] + rad[jc]
    overlap = np.where(valid & (d < min_d), min_d - d, 0.0)
    if not overlap.any():
        return
    away = torus_delta(pos[jc], pos[:, None, :], w.size)          # del vecino hacia mí
    dist = np.linalg.norm(away, axis=2, keepdims=True)
    # Dos cuerpos en el mismo punto no tienen dirección: se empujan en sentidos opuestos
    # según el índice. Con la misma dirección para ambos, viajarían juntos sin separarse.
    sign = np.where(np.arange(len(a))[:, None] < jc, 1.0, -1.0)
    fallback = np.stack([sign, np.zeros_like(sign)], axis=-1)
    away = np.where(dist > 1e-9, away / np.maximum(dist, 1e-9), fallback)
    push = (away * overlap[..., None]).sum(axis=1) * 0.5 * m.separation_strength
    w.pos[a] = wrap(pos + push, w.size)
