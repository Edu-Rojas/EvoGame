"""Energía y vida: comer (pasto, hojas o carne), gastar, envejecer, morir y dejar carne."""
from __future__ import annotations

import numpy as np

from .genes import Gene
from .geometry import torus_delta
from .state import NO_TARGET, Action, DeathCause, World
from .terrain import Biome

# Cuán cerca del centro de la celda hay que estar para pastar, en celdas (geometría:
# 0,35 queda dentro de la celda incluso en diagonal, sin exigir llegar al centro exacto)
EAT_REACH_CELLS = 0.35


def eat(w: World) -> None:
    """Las criaturas que llegaron a su celda objetivo comen de la capa que más energía
    les rinde: pasto, hojas altas (si las alcanzan) o carne.

    Si varias comen la misma capa de la misma celda y no alcanza, se reparte en
    proporción (vectorizado con bincount, sin bucles por criatura).
    """
    m, t = w.cfg.movement, w.terrain
    a = w.alive_idx()
    eaters = a[(w.action[a] == Action.EAT) & (w.target[a] != NO_TARGET)]
    if len(eaters) == 0:
        return
    cell = w.target[eaters]
    dist = np.linalg.norm(torus_delta(w.pos[eaters], t.center_of(cell), w.size), axis=1)
    arrived = dist <= w.radius[eaters] + m.contact_distance + t.cell_size * EAT_REACH_CELLS
    eaters, cell = eaters[arrived], cell[arrived]
    if len(eaters) == 0:
        return

    layers = (t.grass, t.leaves, t.meat)
    eff = np.stack([w.plant_eff[eaters], w.plant_eff[eaters] * w.leaf_reach[eaters],
                    w.meat_eff[eaters]], axis=1)
    stock = np.stack([layer[cell] > 0 for layer in layers], axis=1)
    rate = eff * stock                               # cuánto rinde cada capa disponible
    best = rate.argmax(axis=1)
    for k, layer in enumerate(layers):
        on = (best == k) & (rate[:, k] > 0)
        if on.any():
            _graze(w, layer, eaters[on], cell[on], eff[on, k])


def _graze(w: World, layer: np.ndarray, eaters: np.ndarray, cell: np.ndarray,
           eff: np.ndarray) -> None:
    """Come de una capa del terreno con reparto justo cuando no alcanza."""
    room = np.maximum(w.reserve[eaters] - w.energy[eaters], 0.0)
    # come lo que le cabe contando lo que de verdad aprovecha (eff > 0 aquí)
    want = np.minimum(w.cfg.body.bite, room / eff)
    n_cells = len(layer)
    demand = np.bincount(cell, weights=want, minlength=n_cells)
    ratio = np.divide(layer, demand, out=np.ones(n_cells), where=demand > 0)
    got = want * np.minimum(ratio, 1.0)[cell]
    layer -= np.minimum(demand, layer)
    w.energy[eaters] = np.minimum(w.reserve[eaters], w.energy[eaters] + got * eff)


def spend(w: World) -> None:
    """Gasto por tick: (apetito + moverse) x costo del bioma donde está parado.

    El frío de la helada aplica la regla de Bergmann: los animales grandes pierden
    menos calor (menos superficie por volumen), así que el extra por frío se divide
    por tamaño^(1/3). Es la primera ventaja real del Tamaño en la 1a.
    """
    b, m, t = w.cfg.body, w.cfg.movement, w.terrain
    a = w.alive_idx()
    size = w.genes[a, Gene.SIZE]
    speed = np.linalg.norm(w.vel[a], axis=1)
    kleiber = size ** b.kleiber_exponent
    move = m.move_cost * kleiber * (speed / m.base_speed) ** 2

    cell = t.cell_of(w.pos[a])
    extra = t.cost[cell] - 1.0
    cold = t.biome[cell] == Biome.TUNDRA
    extra = np.where(cold, extra * size ** -w.cfg.terrain.cold_size_exponent, extra)

    w.energy[a] -= (w.appetite[a] + move) * (1.0 + extra)
    w.age[a] += w.aging_rate[a]
    w.cooldown[a] = np.maximum(w.cooldown[a] - 1, 0)


def die(w: World) -> np.ndarray:
    """Muere quien se queda sin energía o llega al final de su vida. Devuelve los slots.

    Causas: cazado (vida en 0; el crédito es del último que la mordió), hambre y vejez.
    Todo cadáver deja carne en su celda: por su tamaño más parte de la energía que
    tenía (los muertos de hambre o vejez también: carroña). Cuenta la causa por
    especie y anota el tick en que una especie se extingue.
    """
    hunted = w.alive & (w.health <= 0)
    starved = w.alive & ~hunted & (w.energy <= 0)
    old = w.alive & ~hunted & ~starved & (w.age >= w.cfg.body.base_lifespan)
    idx = np.flatnonzero(hunted | starved | old)
    if len(idx) == 0:
        return idx
    for cause, mask in ((DeathCause.PREDATION, hunted), (DeathCause.STARVATION, starved),
                        (DeathCause.OLD_AGE, old)):
        np.add.at(w.deaths_by_cause[:, cause], w.species[mask], 1)
    _credit_kills(w, w.last_hitter[hunted])
    _drop_meat(w, idx)
    w.alive[idx] = False
    w.target[idx] = NO_TARGET
    w.target_uid[idx] = -1
    w.deaths_total += len(idx)
    _mark_extinctions(w, np.unique(w.species[idx]))
    return idx


def _credit_kills(w: World, killer_uids: np.ndarray) -> None:
    killer_uids = killer_uids[killer_uids >= 0]
    if len(killer_uids) == 0:
        return
    a = w.alive_idx()
    order = np.argsort(w.uid[a])
    sorted_uids = w.uid[a][order]
    pos = np.searchsorted(sorted_uids, killer_uids)
    found = (pos < len(sorted_uids)) & (sorted_uids[np.minimum(pos, len(sorted_uids) - 1)] == killer_uids)
    np.add.at(w.kills, a[order[pos[found]]], 1)


def _drop_meat(w: World, dead: np.ndarray) -> None:
    mc = w.cfg.meat
    meat = mc.per_size * w.genes[dead, Gene.SIZE] + mc.energy_fraction * np.maximum(w.energy[dead], 0.0)
    np.add.at(w.terrain.meat, w.terrain.cell_of(w.pos[dead]), meat)


def _mark_extinctions(w: World, species: np.ndarray) -> None:
    for sp in species:
        if w.extinct_at[sp] < 0 and not (w.alive & (w.species == sp)).any():
            w.extinct_at[sp] = w.tick


def regrow(w: World) -> None:
    """Rebrote de pasto y hojas, y la carne que se pudre."""
    w.terrain.regrow_step()
