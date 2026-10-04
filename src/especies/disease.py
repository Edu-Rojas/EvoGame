"""Enfermedad por densidad: vivir apiñado con los de tu especie cuesta energía (contagio).

Es la mecánica estabilizadora más "artificial" de la lista de diseño, y por eso la
última: solo entra si las otras (refugio por tamaño, saciedad, hojas altas, cobertura,
filopatría) no alcanzan. En ecología se conoce como "kill the winner": castiga a la
especie más abundante en cada lugar, sea presa o depredador, y frena tanto la
explosión de conejos como la de lobos.

Depende solo de cuántos vecinos de la MISMA especie hay cerca, nunca de qué especie es.
El costo es proporcional al apetito de cada uno, así no castiga más a los grandes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .state import World


def crowding(w: World) -> np.ndarray:
    """Por criatura viva (en el orden de alive_idx): vecinos de su especie en el radio."""
    d = w.cfg.disease
    a = w.alive_idx()
    counts = np.zeros(len(a), dtype=np.int64)
    for sp in np.unique(w.species[a]):
        mine = np.flatnonzero(w.species[a] == sp)
        tree = cKDTree(w.pos[a[mine]], boxsize=w.size)
        counts[mine] = tree.query_ball_point(w.pos[a[mine]], r=d.radius, return_length=True) - 1
    return counts


def sicken(w: World) -> None:
    d = w.cfg.disease
    if d.cost_per_neighbor <= 0:
        return
    a = w.alive_idx()
    if len(a) == 0:
        return
    excess = np.maximum(crowding(w) - d.free_neighbors, 0)
    w.energy[a] -= w.appetite[a] * d.cost_per_neighbor * excess
