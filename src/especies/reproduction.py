"""Reproducción sexual entre hermafroditas: cruce uniforme + mutación.

Aquí SÍ hay un bucle de Python, y es a propósito: recorre parejas que se aparean
este tick (unas pocas), no criaturas (miles). Regla práctica: vectorizar lo que es
"por criatura", y un bucle está bien para eventos raros.
"""
from __future__ import annotations

import numpy as np

from .genes import GENE_MAX, GENE_MIN, Gene, crossover, mutate, stochastic_round
from .geometry import torus_delta
from .state import Action, NO_TARGET, World, spawn


def ready_mask(w: World) -> np.ndarray:
    """Adulto, con energía suficiente y sin enfriamiento."""
    r = w.cfg.reproduction
    maturity = r.maturity_fraction * w.cfg.body.base_lifespan
    return (w.alive
            & (w.age >= maturity)
            & (w.energy >= r.ready_energy_fraction * w.reserve)
            & (w.cooldown == 0))


def litter_size(gene_a: float, gene_b: float, w: World) -> int:
    """Promedio del gen Apareamiento de los padres -> 1..max_litter crías.

    Ojo: esto NO es herencia (los genes de la cría no se promedian). Es solo la
    decisión de cuántas crías tiene esta camada.
    """
    avg = (gene_a + gene_b) / 2
    expected = 1 + (avg - 1) / 3            # gen 1 -> 1 cría, gen 10 -> 4
    n = int(stochastic_round(expected, w.rng))
    return int(np.clip(n, 1, w.cfg.reproduction.max_litter))


def reproduce(w: World, ready: np.ndarray) -> int:
    """Aparea a las parejas que están en contacto. Devuelve cuántas crías nacieron."""
    cfg = w.cfg
    r, m = cfg.reproduction, cfg.movement
    a = w.alive_idx()
    cand = a[(w.action[a] == Action.MATE) & (w.target[a] != NO_TARGET)]
    if len(cand) == 0:
        return 0
    mate = w.target[cand]
    delta = torus_delta(w.pos[cand], w.pos[mate], w.size)
    dist = np.linalg.norm(delta, axis=1)
    ok = (w.alive[mate] & ready[cand] & ready[mate]
          & (w.species[mate] == w.species[cand])
          # misma "huella" que usa la separación de cuerpos; si no, la separación
          # los mantendría siempre un poquito más lejos de lo necesario para aparearse
          & (dist <= (w.radius[cand] + w.radius[mate]) * m.body_footprint + m.contact_distance))

    # Con la capacidad llena, aparearse no puede ser pura pérdida: si no, el tope de
    # rendimiento castigaría a las especies que más se reproducen (y decidiría quién gana).
    free = int((~w.alive).sum())
    used = np.zeros(len(w.alive), dtype=bool)
    batches: list[dict[str, np.ndarray]] = []
    for i, j, dl in zip(cand[ok], mate[ok], delta[ok]):
        if free == 0:
            break                         # las parejas que faltan no pagan; reintentan luego
        if used[i] or used[j]:
            continue                      # cada uno se aparea una vez por tick
        used[i] = used[j] = True

        planned = litter_size(w.genes[i, Gene.MATING], w.genes[j, Gene.MATING], w)
        n = min(planned, free)
        free -= n
        # se paga solo por las crías que nacen: cada una recibe lo mismo que sin tope
        paid = r.contribution * n / planned
        litter_energy = paid * (w.energy[i] + w.energy[j])
        w.energy[i] *= 1 - paid
        w.energy[j] *= 1 - paid
        w.cooldown[i] = w.cooldown[j] = r.cooldown
        for p in (i, j):
            w.action[p] = Action.EXPLORE
            w.target[p] = NO_TARGET

        ga = np.tile(w.genes[i], (n, 1))
        gb = np.tile(w.genes[j], (n, 1))
        genes = mutate(crossover(ga, gb, w.rng), cfg.genes.mutation_sigma,
                       GENE_MIN, GENE_MAX, w.rng)
        b = cfg.behavior
        ia = np.tile(w.instinct[i], (n, 1))
        ib = np.tile(w.instinct[j], (n, 1))
        instinct = mutate(crossover(ia, ib, w.rng), b.instinct_sigma,
                          b.instinct_min, b.instinct_max, w.rng)

        midpoint = w.pos[i] + dl / 2
        batches.append({
            "species": np.full(n, w.species[i]),
            "genes": genes,
            "instinct": instinct,
            "pos": midpoint + w.rng.normal(0, r.spawn_spread, (n, 2)),
            "energy": np.full(n, litter_energy / n),
            "age": np.zeros(n),
            "generation": np.full(n, max(w.generation[i], w.generation[j]) + 1),
            "parent_a": np.full(n, w.uid[i]),
            "parent_b": np.full(n, w.uid[j]),
        })

    if not batches:
        return 0
    merged = {k: np.concatenate([bt[k] for bt in batches]) for k in batches[0]}
    born = spawn(w, **merged)
    w.births_total += born
    return born
