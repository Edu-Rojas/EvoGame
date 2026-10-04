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


@dataclass(frozen=True)
class CreatureInfo:
    """Ficha de una criatura con datos crudos: el formato lo pone cada cliente
    (el visor redondea y traduce; la API la serializa con dataclasses.asdict)."""
    uid: int
    species: str
    generation: int
    parents: tuple[int, int]        # uid de los padres (-1 = fundador)
    genes: dict[str, float]         # clave del TOML -> valor 1..10
    instincts: dict[str, float]     # acción -> multiplicador heredable
    energy: float
    reserve: float
    health: float
    max_health: float
    kills: int
    age: float
    action: str                     # nombre de la acción en minúsculas (Action)


def describe_creature(w: World, slot: int) -> CreatureInfo:
    """Ficha de una criatura (para el clic en el visor y, después, la API)."""
    return CreatureInfo(
        uid=int(w.uid[slot]),
        species=w.cfg.species[w.species[slot]].name,
        generation=int(w.generation[slot]),
        parents=(int(w.parent_a[slot]), int(w.parent_b[slot])),
        genes={g.key: float(w.genes[slot, g]) for g in Gene},
        instincts={a.name.lower(): float(w.instinct[slot, a]) for a in Action},
        energy=float(w.energy[slot]),
        reserve=float(w.reserve[slot]),
        health=float(w.health[slot]),
        max_health=float(w.max_health[slot]),
        kills=int(w.kills[slot]),
        age=float(w.age[slot]),
        action=Action(int(w.action[slot])).name.lower(),
    )
