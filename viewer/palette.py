"""Paleta del visor: tinta, papel y cómo se estilizan los colores de la config.

La idea visual: un mundo pastel y apagado, como dibujado en papel, y criaturas con
contorno de tinta, cuerpo suave y la cabeza de un color vivo que identifica a la
especie. El mundo no compite con las criaturas: ellas son lo que se mira.

Sin pygame: son funciones de color sobre arrays, reutilizables en otro visor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Color = tuple[int, int, int]

INK: Color = (38, 30, 46)          # negro violáceo, más vivo que el negro puro
PAPER: Color = (244, 236, 222)     # blanco cálido de cuaderno
SHADOW_ALPHA = 70                  # opacidad de la sombra de las criaturas


def mix(a: np.ndarray | Color, b: np.ndarray | Color, t: float | np.ndarray) -> np.ndarray:
    """Interpola entre dos colores (t=0 -> a, t=1 -> b). Acepta arrays de colores."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    t_ = np.asarray(t, float)
    if t_.ndim:
        t_ = t_[..., None]
    return a_ + (b_ - a_) * t_


def desaturate(c: np.ndarray | Color, amount: float | np.ndarray) -> np.ndarray:
    """Acerca el color a su gris (luminancia). amount=1 -> gris total."""
    c_ = np.asarray(c, float)
    gray = (c_ @ np.array([0.299, 0.587, 0.114]))[..., None]
    return mix(c_, np.broadcast_to(gray, c_.shape), amount)


def to_rgb(c: np.ndarray) -> Color:
    r, g, b = (min(255, max(0, int(round(float(v))))) for v in c)
    return r, g, b


def to_rgb_list(c: np.ndarray) -> list[Color]:
    """Muchos colores [n, 3] -> lista de tuplas (rápido, para pasarle a pygame)."""
    return [tuple(x) for x in np.clip(np.rint(c), 0, 255).astype(int).tolist()]


def stylize_terrain(colors: np.ndarray) -> np.ndarray:
    """Colores de bioma de la config -> versión pastel apagada del visor."""
    return mix(desaturate(colors, 0.15), PAPER, 0.2)


@dataclass(frozen=True)
class SpeciesColors:
    body: Color        # cuerpo, pastel apagado
    light: Color       # brillo del lomo (volumen)
    shade: Color       # manchas y detalles
    accent: Color      # cabeza: el color que identifica a la especie
    accent_light: Color


def species_colors(rgb: Color) -> SpeciesColors:
    base = np.asarray(rgb, float)
    body = mix(desaturate(base, 0.35), PAPER, 0.28)
    accent = mix(base, INK, 0.05)
    return SpeciesColors(
        body=to_rgb(body),
        light=to_rgb(mix(body, PAPER, 0.45)),
        shade=to_rgb(mix(body, INK, 0.3)),
        accent=to_rgb(accent),
        accent_light=to_rgb(mix(accent, PAPER, 0.5)),
    )


def hungry_body(body: Color, energy_frac: np.ndarray) -> np.ndarray:
    """Con hambre el cuerpo se apaga hacia el gris: se nota quién está mal sin texto.

    Devuelve un color por criatura: [n, 3].
    """
    hunger = 1.0 - np.clip(energy_frac, 0.0, 1.0)
    body_n = np.broadcast_to(np.asarray(body, float), (len(hunger), 3))
    return mix(desaturate(body_n, 0.8 * hunger), INK, 0.15 * hunger)
