"""Genes: índices, normalización y las dos operaciones de la herencia.

Todo aquí son funciones puras sobre arrays: fáciles de testear y de vectorizar.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

GENE_MIN, GENE_MAX = 1.0, 10.0


class Gene(IntEnum):
    """Columna de cada gen en la matriz genes[n_criaturas, N_GENES].

    IntEnum para poder escribir genes[:, Gene.SIZE] en vez de genes[:, 0].
    """
    SIZE = 0
    ACCELERATION = 1
    AGGRESSION = 2
    DIET = 3
    CAMOUFLAGE = 4
    DETECTION = 5
    MATING = 6

    @property
    def key(self) -> str:
        """Nombre del gen en el TOML de las especies (lo escriben los jugadores, en español)."""
        return _GENE_KEYS[self]


_GENE_KEYS = {
    Gene.SIZE: "tamano",
    Gene.ACCELERATION: "aceleracion",
    Gene.AGGRESSION: "agresividad",
    Gene.DIET: "dieta",
    Gene.CAMOUFLAGE: "camuflaje",
    Gene.DETECTION: "deteccion",
    Gene.MATING: "apareamiento",
}

N_GENES = len(Gene)


def norm(values: np.ndarray) -> np.ndarray:
    """Gen 1..10 -> 0..1. Casi todas las fórmulas trabajan con este x."""
    return (values - GENE_MIN) / (GENE_MAX - GENE_MIN)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cruce uniforme: cada gen viene de uno de los dos padres, al azar.

    NO se promedia: promediar borra la variación en pocas generaciones
    (la "herencia por mezcla" que le criticaron a Darwin). Funciona igual
    con un par de vectores o con matrices de varios hijos.
    """
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b)


def mutate(values: np.ndarray, sigma: float, lo: float, hi: float,
           rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano pequeño, recortado al rango válido."""
    return np.clip(values + rng.normal(0.0, sigma, values.shape), lo, hi)


def stochastic_round(x: np.ndarray | float, rng: np.random.Generator) -> np.ndarray:
    """Redondeo al azar: 2,3 da 2 con 70% de prob. y 3 con 30%.

    El promedio es exactamente x, así que cada décima de un gen importa un poco
    y la selección tiene una pendiente que seguir (con round() normal, una
    mutación de ±0,3 muchas veces no cambia nada y el gen se estanca).
    """
    x = np.asarray(x, dtype=float)
    floor = np.floor(x)
    return (floor + (rng.random(x.shape) < (x - floor))).astype(int)
