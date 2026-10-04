"""Geometría del mundo toroidal: lo que sale por un borde entra por el opuesto.

Módulo sin dependencias del resto del paquete, para que cualquiera (state, physics,
el visor) lo pueda importar sin ciclos.
"""
from __future__ import annotations

import numpy as np


def wrap(pos: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Mete posiciones dentro de [0, size). Lo que sale por un borde entra por el otro.

    El np.where cubre un caso feo de punto flotante: np.mod(-1e-17, 1600) puede dar
    exactamente 1600.0, y el KD-tree con boxsize rechaza puntos == boxsize.
    """
    p = np.mod(pos, size)
    return np.where(p >= size, 0.0, p)


def torus_delta(frm: np.ndarray, to: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Vector más corto de `frm` a `to` en un toro (puede cruzar el borde)."""
    d = to - frm
    return d - size * np.round(d / size)
