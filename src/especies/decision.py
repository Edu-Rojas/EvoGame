"""Decisión por utilidad: instinto base (escrito aquí) x instinto heredable (genes ocultos).

Cada acción recibe un puntaje; gana la más alta, con un poco de ruido para que dos
bichos idénticos no hagan siempre exactamente lo mismo. Esta es la pieza que en la
v2 se puede cambiar por una red neuronal sin tocar el resto: entra lo percibido y el
estado, salen utilidades.
"""
from __future__ import annotations

import numpy as np

from .perception import Perception
from .state import Action, NO_TARGET, World


def utilities(w: World, thinkers: np.ndarray, p: Perception, ready: np.ndarray) -> np.ndarray:
    """Matriz [n_thinkers, Action.N] con el puntaje de cada acción."""
    b = w.cfg.behavior
    hunger = 1.0 - np.clip(w.energy[thinkers] / w.reserve[thinkers], 0.0, 1.0)

    u = np.zeros((len(thinkers), Action.N))
    u[:, Action.EXPLORE] = b.explore_base
    u[:, Action.EAT] = (p.food != NO_TARGET) * (b.eat_base + hunger)
    u[:, Action.MATE] = (p.mate != NO_TARGET) * ready[thinkers] * 1.0
    return u * w.instinct[thinkers]


def decide(w: World, thinkers: np.ndarray, p: Perception, ready: np.ndarray) -> None:
    u = utilities(w, thinkers, p, ready)
    u += w.rng.normal(0.0, w.cfg.behavior.decision_noise, u.shape)
    act = u.argmax(axis=1).astype(np.int8)

    target = np.full(len(thinkers), NO_TARGET, dtype=np.int32)
    target[act == Action.EAT] = p.food[act == Action.EAT]
    target[act == Action.MATE] = p.mate[act == Action.MATE]
    # el ruido podría elegir "comer" sin comida a la vista: eso es explorar
    act[(act != Action.EXPLORE) & (target == NO_TARGET)] = Action.EXPLORE

    w.action[thinkers] = act
    w.target[thinkers] = target


def validate_targets(w: World, ready: np.ndarray) -> None:
    """Entre una pensada y otra el objetivo puede desaparecer: pasto comido, pareja
    muerta o que ya no está lista. En ese caso la criatura vuelve a explorar."""
    a = w.alive_idx()
    act, tgt = w.action[a], w.target[a]

    eating = (act == Action.EAT) & (tgt != NO_TARGET)
    gone = np.zeros(len(a), dtype=bool)
    gone[eating] = w.terrain.grass[tgt[eating]] < 0.5   # se acabó el pasto

    mating = (act == Action.MATE) & (tgt != NO_TARGET)
    t = tgt[mating]
    gone[mating] = ~w.alive[t] | ~ready[t] | ~ready[a[mating]]

    lost = a[gone]
    w.action[lost] = Action.EXPLORE
    w.target[lost] = NO_TARGET
