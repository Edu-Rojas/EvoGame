"""Un tick de simulación: el orden en que corren los sistemas.

Cada sistema es una función que lee y escribe columnas del World. Mutan los arrays
en el sitio (por rendimiento), pero con la misma disciplina que el estilo funcional:
no guardan estado propio, todo lo que necesitan entra por el World, y toda la
aleatoriedad sale de w.rng. Por eso misma semilla = misma simulación.
"""
from __future__ import annotations

from .combat import fight, heal
from .decision import decide, validate_targets
from .disease import sicken
from .metabolism import die, eat, regrow, spend
from .perception import perceive
from .physics import move, separate
from .reproduction import ready_mask, reproduce
from .state import World


def step(w: World) -> None:
    ready = ready_mask(w)
    validate_targets(w, ready)

    # Solo piensa una fracción cada tick, repartida por slot: (tick + slot) % intervalo.
    # Cada criatura re-decide cada `think_interval` ticks y entre medio sigue su plan.
    slots = w.alive_idx()
    thinkers = slots[(w.tick + slots) % w.cfg.sim.think_interval == 0]
    if len(thinkers) > 0:
        p = perceive(w, thinkers, ready)
        decide(w, thinkers, p, ready)

    move(w)
    separate(w)
    fight(w)
    eat(w)
    reproduce(w, ready)
    spend(w)
    sicken(w)
    heal(w)
    regrow(w)
    die(w)
    w.tick += 1


def run(w: World, ticks: int) -> None:
    for _ in range(ticks):
        step(w)
