"""Energía y vida: comer pasto, gastar, envejecer, morir, y el rebrote del pasto."""
from __future__ import annotations

import numpy as np

from .genes import Gene
from .geometry import torus_delta
from .state import NO_TARGET, Action, World
from .terrain import Biome

# Cuán cerca del centro de la celda hay que estar para pastar, en celdas (geometría:
# 0,35 queda dentro de la celda incluso en diagonal, sin exigir llegar al centro exacto)
EAT_REACH_CELLS = 0.35


def eat(w: World) -> None:
    """Las criaturas que llegaron a su celda objetivo pastan.

    Si varias pastan la misma celda y no alcanza, se reparte en proporción
    (vectorizado con bincount, sin bucles por criatura).
    """
    b, m, t = w.cfg.body, w.cfg.movement, w.terrain
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

    eff = w.plant_eff[eaters]
    room = np.maximum(w.reserve[eaters] - w.energy[eaters], 0.0)
    # pasto que le cabe, contando lo que de verdad aprovecha; quien no saca nada de
    # las plantas (carnívoro puro) no arranca pasto para nada
    want = np.where(eff > 0, np.minimum(b.bite, room / np.maximum(eff, 1e-12)), 0.0)

    n_cells = len(t.grass)
    demand = np.bincount(cell, weights=want, minlength=n_cells)
    ratio = np.divide(t.grass, demand, out=np.ones(n_cells), where=demand > 0)
    got = want * np.minimum(ratio, 1.0)[cell]

    t.grass -= np.minimum(demand, t.grass)
    w.energy[eaters] = np.minimum(w.reserve[eaters], w.energy[eaters] + got * w.plant_eff[eaters])


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
    """Muere quien se queda sin energía o llega al final de su vida. Devuelve los slots."""
    dead = w.alive & ((w.energy <= 0) | (w.age >= w.cfg.body.base_lifespan))
    idx = np.flatnonzero(dead)
    w.alive[idx] = False
    w.target[idx] = NO_TARGET
    w.deaths_total += len(idx)
    return idx


def regrow_grass(w: World) -> None:
    w.terrain.regrow_step()
