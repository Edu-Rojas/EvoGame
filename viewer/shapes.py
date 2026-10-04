"""Utilidades visuales pequeñas. Sin pygame.

La geometría de los cuerpos vive en animation.py y los colores en palette.py.
"""
from __future__ import annotations

import numpy as np


def growth(age: np.ndarray, maturity_age: float) -> np.ndarray:
    """Las crías nacen a la mitad del tamaño y crecen hasta la madurez.

    Con madurez 0 (la config lo permite) nacen ya de tamaño adulto.
    """
    if maturity_age <= 0:
        return np.ones_like(age, dtype=float)
    return 0.5 + 0.5 * np.minimum(age / maturity_age, 1.0)
