"""Decisión por utilidad: instinto base (escrito aquí) x instinto heredable (genes ocultos).

Cada acción recibe un puntaje; gana la más alta, con un poco de ruido para que dos
bichos idénticos no hagan siempre exactamente lo mismo. Esta es la pieza que en la
v2 se puede cambiar por una red neuronal sin tocar el resto: entra lo percibido y el
estado, salen utilidades.
"""
from __future__ import annotations

import numpy as np

from .perception import Perception, food_value
from .state import CREATURE_TARGETS, N_ACTIONS, NO_TARGET, Action, World


def utilities(w: World, thinkers: np.ndarray, p: Perception, ready: np.ndarray) -> np.ndarray:
    """Matriz [n_thinkers, N_ACTIONS] con el puntaje de cada acción."""
    b, c = w.cfg.behavior, w.cfg.combat
    energy_frac = np.clip(w.energy[thinkers] / w.reserve[thinkers], 0.0, 1.0)
    hunger = 1.0 - energy_frac
    # saciedad (Holling tipo II): con la panza llena no caza, aunque vea presas
    sated = energy_frac > c.satiety_fraction

    u = np.zeros((len(thinkers), N_ACTIONS))
    u[:, Action.EXPLORE] = b.explore_base
    u[:, Action.EAT] = (p.food != NO_TARGET) * (b.eat_base + hunger)
    u[:, Action.MATE] = (p.mate != NO_TARGET) * ready[thinkers] * b.mate_base
    # imagen de búsqueda (Holling tipo III): con presas escasas a la vista casi no vale la
    # pena salir a cazar; la presa rara encuentra refugio en su rareza y se recupera
    denom = p.prey_seen + c.prey_search_half
    search_image = np.divide(p.prey_seen, denom, out=np.ones(len(thinkers)), where=denom > 0)
    u[:, Action.HUNT] = (((p.prey != NO_TARGET) & ~sated) * b.hunt_base * hunger
                         * w.meat_eff[thinkers] * search_image)
    u[:, Action.FLEE] = b.flee_base * p.threat_level
    return u * w.instinct[thinkers]


def decide(w: World, thinkers: np.ndarray, p: Perception, ready: np.ndarray) -> None:
    u = utilities(w, thinkers, p, ready)
    u += w.rng.normal(0.0, w.cfg.behavior.decision_noise, u.shape)
    act = u.argmax(axis=1).astype(np.int8)

    target = np.full(len(thinkers), NO_TARGET, dtype=np.int64)
    for action, chosen in ((Action.EAT, p.food), (Action.MATE, p.mate),
                           (Action.HUNT, p.prey), (Action.FLEE, p.threat)):
        target[act == action] = chosen[act == action]
    # el ruido podría elegir una acción sin objetivo a la vista: eso es explorar
    act[(act != Action.EXPLORE) & (target == NO_TARGET)] = Action.EXPLORE

    # la persecución se cuenta desde que cambia la presa
    new_chase = (act != Action.HUNT) | (target != w.target[thinkers])
    w.hunt_ticks[thinkers[new_chase]] = 0
    w.action[thinkers] = act
    w.target[thinkers] = target
    # uid solo para objetivos que son criaturas (el de comer es una celda, no un slot)
    is_creature = np.isin(act, CREATURE_TARGETS)
    uid = np.full(len(thinkers), -1, dtype=np.int64)
    uid[is_creature] = w.uid[target[is_creature]]
    w.target_uid[thinkers] = uid


def validate_targets(w: World, ready: np.ndarray) -> None:
    """Entre una pensada y otra el objetivo puede desaparecer: comida agotada, pareja
    o presa muerta (o su slot reutilizado por otra criatura), pareja que ya no está
    lista, persecución demasiado larga o panza llena. Entonces vuelve a explorar."""
    a = w.alive_idx()
    act, tgt = w.action[a], w.target[a]
    gone = np.zeros(len(a), dtype=bool)

    eating = (act == Action.EAT) & (tgt != NO_TARGET)
    gone[eating] = food_value(w, a[eating], tgt[eating]) < w.cfg.terrain.min_to_stay

    # objetivos que son criaturas: muertas o con el slot reutilizado se detectan por uid
    creature = np.isin(act, CREATURE_TARGETS) & (tgt != NO_TARGET)
    t = tgt[creature]
    gone[creature] = ~w.alive[t] | (w.uid[t] != w.target_uid[a[creature]])

    mating = (act == Action.MATE) & (tgt != NO_TARGET)
    gone[mating] |= ~ready[tgt[mating]] | ~ready[a[mating]]

    hunting = (act == Action.HUNT) & (tgt != NO_TARGET)
    h = a[hunting]
    sated = w.energy[h] > w.cfg.combat.satiety_fraction * w.reserve[h]
    gone[hunting] |= (w.hunt_ticks[h] >= w.chase_limit[h]) | sated

    lost = a[gone]
    w.action[lost] = Action.EXPLORE
    w.target[lost] = NO_TARGET
    w.target_uid[lost] = -1
