"""Peleas: el cazador en contacto muerde; la presa devuelve el golpe según su agresividad.

Todo vectorizado. Varios cazadores pueden morder a la misma presa en el mismo tick:
el daño se acumula con np.add.at (con indexado normal, `x[i] += d` con índices
repetidos solo sumaría una vez).
"""
from __future__ import annotations

import numpy as np

from .geometry import torus_delta
from .perception import can_hunt
from .state import NO_TARGET, Action, World


def fight(w: World) -> None:
    c, m = w.cfg.combat, w.cfg.movement
    a = w.alive_idx()
    hunting = a[(w.action[a] == Action.HUNT) & (w.target[a] != NO_TARGET)]
    w.hunt_ticks[hunting] += 1                       # cuánto lleva persiguiendo
    if len(hunting) == 0:
        return
    prey = w.target[hunting]
    # objetivo vivo, el mismo que eligió (uid) y dentro del refugio por tamaño
    valid = w.alive[prey] & (w.uid[prey] == w.target_uid[hunting]) & can_hunt(w, hunting, prey)
    hunting, prey = hunting[valid], prey[valid]
    dist = np.linalg.norm(torus_delta(w.pos[hunting], w.pos[prey], w.size), axis=1)
    reach = (w.radius[hunting] + w.radius[prey]) * m.body_footprint + m.contact_distance
    biting = dist <= reach
    hunter, prey = hunting[biting], prey[biting]
    if len(hunter) == 0:
        return

    damage = np.zeros(len(w.alive))
    np.add.at(damage, prey, w.bite_damage[hunter])
    # la presa devuelve el golpe: un herbívoro dócil casi no pega
    np.add.at(damage, hunter, w.bite_damage[prey] * w.retaliation[prey])
    w.energy[hunter] -= c.bite_energy_cost
    w.health -= damage
    # crédito de la muerte: quién mordió último (con índices repetidos gana el último)
    w.last_hitter[prey] = w.uid[hunter]
    hit_back = w.retaliation[prey] > 0
    w.last_hitter[hunter[hit_back]] = w.uid[prey[hit_back]]


def heal(w: World) -> None:
    """Regeneración lenta de la vida."""
    a = w.alive_idx()
    regen = w.cfg.combat.health_regen * w.max_health[a]
    w.health[a] = np.minimum(w.max_health[a], w.health[a] + regen)
