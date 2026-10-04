"""Utilidades visuales pequeñas (colores y crecimiento). Sin pygame.

La geometría de los cuerpos vive en animation.py (columna, patas con IK y plumas).
"""
from __future__ import annotations

import numpy as np


def growth(age: np.ndarray, maturity_age: float) -> np.ndarray:
    """Las crías nacen a la mitad del tamaño y crecen hasta la madurez."""
    return 0.5 + 0.5 * np.minimum(age / maturity_age, 1.0)


def lighten(c, k=0.45) -> tuple[int, int, int]:
    return tuple(int(v + (255 - v) * k) for v in c)


def dim(c, k) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, v * k))) for v in c)
