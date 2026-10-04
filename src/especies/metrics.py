"""Métricas para el headless, las gráficas y (más adelante) la base de datos."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genes import Gene
from .state import Action, World


@dataclass(frozen=True)
class SpeciesSummary:
    name: str
    count: int
    mean_genes: dict[str, float]
    max_generation: int
    mean_energy_frac: float


def summarize(w: World) -> list[SpeciesSummary]:
    out = []
    for sp_id, sp in enumerate(w.cfg.species):
        idx = np.flatnonzero(w.alive & (w.species == sp_id))
        if len(idx) == 0:
            out.append(SpeciesSummary(sp.name, 0, {g.key: float("nan") for g in Gene}, 0, 0.0))
            continue
        means = w.genes[idx].mean(axis=0)
        out.append(SpeciesSummary(
            name=sp.name,
            count=len(idx),
            mean_genes={g.key: float(means[g]) for g in Gene},
            max_generation=int(w.generation[idx].max()),
            mean_energy_frac=float((w.energy[idx] / w.reserve[idx]).mean()),
        ))
    return out


def describe_creature(w: World, slot: int) -> dict:
    """Ficha de una criatura (para el clic en el visor y, después, la API)."""
    return {
        "uid": int(w.uid[slot]),
        "species": w.cfg.species[w.species[slot]].name,
        "generation": int(w.generation[slot]),
        "parents": (int(w.parent_a[slot]), int(w.parent_b[slot])),
        "genes": {g.key: round(float(w.genes[slot, g]), 2) for g in Gene},
        "instincts": [round(float(v), 2) for v in w.instinct[slot]],
        "energy": f"{w.energy[slot]:.0f}/{w.reserve[slot]:.0f}",
        "age": int(w.age[slot]),
        "action": Action.LABELS[w.action[slot]],
    }
