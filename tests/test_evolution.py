"""Tests de integración largos: la evolución hace lo que el diseño dice.

Marcados `slow`: `pytest -m "not slow"` los salta para iterar rápido.
"""
import numpy as np
import pytest

from especies.genes import Gene
from especies.state import create_world
from especies.step import run


@pytest.mark.slow
def test_every_species_reproduces(cfg):
    """Regresión: la separación de cuerpos llegó a impedir que se aparearan los grandes."""
    w = create_world(cfg, seed=42)
    best = np.zeros(len(cfg.species), dtype=int)
    for _ in range(10):
        run(w, 100)
        for sp in range(len(cfg.species)):
            alive = w.alive & (w.species == sp)
            if alive.any():
                best[sp] = max(best[sp], w.generation[alive].max())
    assert (best >= 1).all(), f"generación máxima por especie: {best}"


@pytest.mark.slow
def test_selection_pushes_size_down_in_1a(cfg):
    """En la 1a el tamaño solo cuesta (sus ventajas llegan con las peleas en la 1b),
    así que la selección debería empujarlo hacia abajo. Si esto falla, algo no
    está seleccionando."""
    w = create_world(cfg, seed=42)
    sp = 0  # Conejos
    before = w.genes[w.alive & (w.species == sp), Gene.SIZE].mean()
    run(w, 3000)
    after = w.genes[w.alive & (w.species == sp), Gene.SIZE].mean()
    assert after < before


@pytest.mark.slow
def test_inactive_genes_stay_put_across_generations(cfg):
    w = create_world(cfg, seed=3)
    run(w, 800)
    kids = w.alive & (w.generation > 0)
    assert kids.any()
    assert (w.genes[kids, Gene.AGGRESSION] == 1.0).all()
