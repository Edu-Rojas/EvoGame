"""Dibujo con pygame sobre un lienzo de baja resolución (pixel art).

Todo el mundo se dibuja en un lienzo `pixel` veces más chico que la ventana y luego
se escala con vecino más cercano: cada píxel del lienzo es un bloque nítido en
pantalla. Así la animación procedural queda como pixel art de verdad (como Rain World
o Hyper Light Drifter) y de paso dibujar sale más barato.

- TerrainRenderer: biomas en pastel apagado; entre celdas, dithering ordenado (Bayer)
  en vez de mezcla suave, y "garabatos" a mano (matas, arbustos, piedras, olas) que
  aparecen al acercarse. Las matas desaparecen cuando el pasto se come: el pastoreo
  se ve.
- CreatureRenderer: tres niveles de detalle según el tamaño en pantalla. De cerca,
  silueta con contorno de tinta, lomo con brillo, manchas, cabeza de color vivo,
  ojos de tinta, patas con garras y branquias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from especies.terrain import Biome

from . import palette as pal
from .animation import Rig

# Dithering ordenado 4x4 (umbrales en [0, 1))
BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
LIGHT_DIR = np.array([-0.55, -0.83])          # la luz viene de arriba a la izquierda
SHADOW_OFFSET = np.array([0.35, 0.5])         # sombra hacia abajo a la derecha (x radio)

# Tamaño mínimo (en píxeles del lienzo) para cada nivel de detalle
DECOR_MIN_CELL_PX = 12
SMALL_MIN_PX = 1.2
DETAIL_MIN_PX = 3.0
FANG_MIN_DIET = 0.35       # dieta normalizada desde la que se dibujan colmillos
MEAT_FULL = 60.0           # carne en una celda que la tiñe al máximo
MEAT_TINT_MAX = 0.75


@dataclass
class View:
    """Cámara traducida al lienzo: centro de la cámara (mundo), zoom y centro del lienzo."""
    center: np.ndarray
    zoom: float                # píxeles del lienzo por unidad de mundo
    canvas_center: np.ndarray
    canvas_size: tuple[int, int]


# ---------------------------------------------------------------- terreno
class Decor:
    TUFT, FLOWER, BUSH, PEBBLE, ROCK, SNOW, WAVE = range(7)


# Garabatos por bioma: (tipo, probabilidad por ranura)
BIOME_DECOR = {
    Biome.GRASSLAND: [(Decor.TUFT, 0.9), (Decor.TUFT, 0.6), (Decor.FLOWER, 0.15)],
    Biome.FOREST: [(Decor.BUSH, 0.55), (Decor.TUFT, 0.7), (Decor.TUFT, 0.4)],
    Biome.DESERT: [(Decor.PEBBLE, 0.35), (Decor.PEBBLE, 0.15), (Decor.TUFT, 0.1)],
    Biome.MOUNTAIN: [(Decor.ROCK, 0.55), (Decor.PEBBLE, 0.4), (Decor.PEBBLE, 0.2)],
    Biome.TUNDRA: [(Decor.SNOW, 0.5), (Decor.SNOW, 0.3), (Decor.TUFT, 0.1)],
    Biome.WATER: [(Decor.WAVE, 0.55), (Decor.WAVE, 0.2), (Decor.WAVE, 0.0)],
}
FLOWER_COLORS = [(246, 190, 200), (250, 226, 150), (236, 236, 250), (200, 190, 245)]


class TerrainRenderer:
    def __init__(self, world, rng: np.random.Generator):
        t, cfg = world.terrain, world.cfg
        self.t = t
        lush = np.array([cfg.biomes[b.key].color_lush for b in Biome], float)
        bare = np.array([cfg.biomes[b.key].color_bare for b in Biome], float)
        self.biome_lush = pal.stylize_terrain(lush)
        texture = rng.normal(0, 3.5, (len(t.biome), 1))     # cada celda un poco distinta
        self.lush = self.biome_lush[t.biome] + texture
        self.bare = pal.stylize_terrain(bare)[t.biome] + texture
        self._prepare_decor(rng)
        self._cache_key: tuple | None = None
        self._cell_idx = np.zeros((0, 0), dtype=np.int64)
        self._grain = np.zeros((0, 0, 1))

    def _prepare_decor(self, rng: np.random.Generator) -> None:
        n = len(self.t.biome)
        slots = 3
        kind = np.full((n, slots), -1, dtype=np.int8)
        for b, spec in BIOME_DECOR.items():
            cells = np.flatnonzero(self.t.biome == b)
            for k, (d, p) in enumerate(spec):
                on = rng.random(len(cells)) < p
                kind[cells[on], k] = d
        self.decor_kind = kind
        self.decor_off = rng.uniform(0.15, 0.85, (n, slots, 2))   # dentro de la celda
        self.decor_size = rng.uniform(0.7, 1.3, (n, slots))
        self.decor_seed = rng.random((n, slots))
        self.decor_color = rng.integers(0, len(FLOWER_COLORS), (n, slots))
        # colores de los garabatos por celda, precalculados (son fijos)
        self.decor_dark = pal.to_rgb_list(pal.mix(self.lush, pal.INK, 0.4))
        self.decor_mid = pal.to_rgb_list(pal.mix(self.lush, pal.INK, 0.2))
        self.decor_light = pal.to_rgb_list(pal.mix(self.lush, pal.PAPER, 0.35))

    def cell_colors(self) -> np.ndarray:
        t = self.t
        frac = np.divide(t.grass, t.grass_max, out=np.ones_like(t.grass), where=t.grass_max > 0)
        rgb = self.bare + (self.lush - self.bare) * frac[:, None]
        # carne: donde hubo una muerte el suelo se mancha, y se borra al pudrirse
        meat = np.clip(t.meat / MEAT_FULL, 0.0, 1.0) * MEAT_TINT_MAX
        rgb = pal.mix(rgb, pal.MEAT, meat)
        return np.clip(rgb, 0, 255)

    def draw(self, canvas: pygame.Surface, v: View, time: float) -> None:
        # Qué celda pinta cada píxel (con su dithering) y el grano solo cambian cuando
        # se mueve la cámara: se cachean, y cada frame solo cambia el color del pasto
        key = (float(v.center[0]), float(v.center[1]), v.zoom, v.canvas_size)
        if key != self._cache_key:
            self._cache_key = key
            self._cell_idx, self._grain = self._pixel_map(v)
        img = self.cell_colors()[self._cell_idx] + self._grain
        pygame.surfarray.blit_array(canvas, np.clip(img, 0, 255).astype(np.uint8))

        if self.t.cell_size * v.zoom >= DECOR_MIN_CELL_PX:
            self._draw_decor(canvas, v, time)

    def _pixel_map(self, v: View) -> tuple[np.ndarray, np.ndarray]:
        """Para cada píxel del lienzo: índice de celda [cw, ch] y grano [cw, ch, 1]."""
        t = self.t
        cw, ch = v.canvas_size

        # coordenadas de mundo del centro de cada píxel del lienzo
        xs = v.center[0] + (np.arange(cw) + 0.5 - v.canvas_center[0]) / v.zoom
        ys = v.center[1] + (np.arange(ch) + 0.5 - v.canvas_center[1]) / v.zoom
        gx, gy = xs / t.cell_size - 0.5, ys / t.cell_size - 0.5
        ix0, iy0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix0, gy - iy0
        # el patrón va anclado al mundo (no a la pantalla) para que no "nade" al mover
        px = np.floor(xs * v.zoom).astype(np.int64)
        py = np.floor(ys * v.zoom).astype(np.int64)
        bx = BAYER4[py[:, None] % 4, px[None, :] % 4]                 # [ch, cw]
        by = BAYER4.T[py[:, None] % 4, px[None, :] % 4]
        cx = (ix0[None, :] + (fx[None, :] > bx)) % t.gw
        cy = (iy0[:, None] + (fy[:, None] > by)) % t.gh
        # grano de papel: ruido fijo por píxel de mundo
        h = (px[None, :] * 73856093) ^ (py[:, None] * 19349663)
        grain = ((h >> 7) & 7).astype(float) - 3.5
        # surfarray usa [x, y]: se trasponen una vez aquí y no en cada frame
        return (cy * t.gw + cx).T.copy(), grain.T[..., None].copy()

    def _visible_cells(self, v: View) -> tuple[np.ndarray, np.ndarray]:
        t = self.t
        half = np.array(v.canvas_size) / 2 / v.zoom
        x0 = int(np.floor((v.center[0] - half[0]) / t.cell_size)) - 1
        x1 = int(np.ceil((v.center[0] + half[0]) / t.cell_size))
        y0 = int(np.floor((v.center[1] - half[1]) / t.cell_size)) - 1
        y1 = int(np.ceil((v.center[1] + half[1]) / t.cell_size))
        ix, iy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        return ix.ravel(), iy.ravel()

    def _draw_decor(self, canvas: pygame.Surface, v: View, time: float) -> None:
        t = self.t
        ix, iy = self._visible_cells(v)
        cell = (iy % t.gh) * t.gw + (ix % t.gw)
        kind = self.decor_kind[cell]                                   # [m, slots]
        frac = np.divide(t.grass[cell], t.grass_max[cell],
                         out=np.zeros(len(cell)), where=t.grass_max[cell] > 0)
        z, cs = v.zoom, t.cell_size
        # posición en el lienzo (celdas sin envolver: así cruzar el borde no salta)
        origin = (np.stack([ix, iy], axis=1) * cs - v.center) * z + v.canvas_center
        on = kind >= 0
        # las matas y flores solo están si queda pasto, y los arbustos si quedan hojas
        # altas: se ve qué se está comiendo cada uno
        seed_k = self.decor_seed[cell]
        grassy = (kind == Decor.TUFT) | (kind == Decor.FLOWER)
        on &= ~grassy | (frac[:, None] >= 0.35 + 0.4 * seed_k)
        leaf_frac = np.divide(t.leaves[cell], t.leaves_max[cell],
                              out=np.ones(len(cell)), where=t.leaves_max[cell] > 0)
        on &= (kind != Decor.BUSH) | (leaf_frac[:, None] >= 0.15 + 0.6 * seed_k)
        ii, kk = np.nonzero(on)
        pos = (origin[ii] + self.decor_off[cell[ii], kk] * cs * z).tolist()
        size = (self.decor_size[cell[ii], kk] * cs * z).tolist()
        seed = self.decor_seed[cell[ii], kk].tolist()
        for j, (i, k) in enumerate(zip(ii.tolist(), kk.tolist(), strict=True)):
            c = int(cell[i])
            self._draw_one(canvas, int(kind[i, k]), pos[j], size[j], c, seed[j],
                           int(self.decor_color[c, k]), time)

    def _draw_one(self, canvas, d, p, s, cell, seed, color_idx, time) -> None:
        x, y = p
        dark_c, light_c = self.decor_dark[cell], self.decor_light[cell]
        if d == Decor.TUFT:                       # mata vista desde arriba: asterisco
            r = max(1.5, s * 0.09)
            for a in (0.3 + seed, 1.35 + seed, 2.4 + seed):
                dx, dy = np.cos(a) * r, np.sin(a) * r
                pygame.draw.line(canvas, dark_c, (x - dx, y - dy), (x + dx, y + dy))
        elif d == Decor.FLOWER:
            r = max(1.0, s * 0.04)
            pygame.draw.circle(canvas, FLOWER_COLORS[color_idx], (x, y), r)
        elif d == Decor.BUSH:                     # copa redonda con contorno y brillo
            r = max(2.0, s * 0.3)
            pygame.draw.circle(canvas, pal.INK, (x, y + 1), r + 1)
            pygame.draw.circle(canvas, dark_c, (x, y), r)
            pygame.draw.circle(canvas, self.decor_mid[cell], (x - r * 0.3, y - r * 0.35), r * 0.45)
        elif d == Decor.PEBBLE:
            r = max(1.0, s * 0.07)
            pygame.draw.ellipse(canvas, dark_c, (x - r * 1.3, y - r, r * 2.6, r * 2))
        elif d == Decor.ROCK:                     # roca facetada
            r = max(2.0, s * 0.18)
            ang = np.linspace(0, 2 * np.pi, 6, endpoint=False) + seed * 6
            rad = r * (0.75 + 0.25 * np.sin(ang * 3 + seed * 10))
            pts = np.stack([x + np.cos(ang) * rad, y + np.sin(ang) * rad * 0.8], axis=1).tolist()
            pygame.draw.polygon(canvas, dark_c, pts)
            pygame.draw.polygon(canvas, light_c, [pts[3], pts[4], (x, y)])
            pygame.draw.polygon(canvas, pal.INK, pts, 1)
        elif d == Decor.SNOW:
            r = max(1.0, s * 0.05)
            pygame.draw.line(canvas, light_c, (x - r, y), (x + r, y))
            pygame.draw.line(canvas, light_c, (x, y - r), (x, y + r))
        elif d == Decor.WAVE:                     # ola que va y viene
            r = max(2.0, s * 0.18)
            shift = np.sin(time * 1.2 + seed * 6.28) * r * 0.5
            u = np.linspace(-1, 1, 5)
            pts = np.stack([x + shift + u * r, y + np.sin(u * 3.1) * r * 0.25], axis=1).tolist()
            pygame.draw.lines(canvas, light_c, False, pts)


# ---------------------------------------------------------------- criaturas
class CreatureRenderer:
    def __init__(self, species_rgb: list[pal.Color]):
        self.colors = [pal.species_colors(c) for c in species_rgb]

    def draw(self, canvas: pygame.Surface, shadow: pygame.Surface, v: View, rig: Rig,
             slots: np.ndarray, species: np.ndarray, base: np.ndarray, radius: np.ndarray,
             energy_frac: np.ndarray, x_acc: np.ndarray, x_det: np.ndarray,
             x_mate: np.ndarray, x_agg: np.ndarray, x_diet: np.ndarray, time: float) -> None:
        """base: posición de cada cabeza relativa al centro de la cámara (mundo)."""
        z, cc = v.zoom, v.canvas_center
        px_r = radius * z
        head_canvas = base * z + cc

        tiny = px_r < SMALL_MIN_PX
        if tiny.any():
            self._draw_tiny(canvas, head_canvas[tiny], species[tiny])
        small = ~tiny & (px_r < DETAIL_MIN_PX)
        for j in np.flatnonzero(small):
            c = self.colors[species[j]]
            p, r = head_canvas[j], px_r[j]
            pygame.draw.circle(canvas, pal.INK, p, r + 1)
            pygame.draw.circle(canvas, c.body, p, r)
            pygame.draw.circle(canvas, c.accent, p, max(1.0, r * 0.6))

        d = np.flatnonzero(~tiny & ~small)
        if len(d) == 0:
            return
        s = slots[d]
        head_cont = rig.spine[s, 0]
        bd = base[d]

        def to_canvas(pts: np.ndarray) -> np.ndarray:
            shape = (len(s),) + (1,) * (pts.ndim - 2) + (2,)
            return (bd.reshape(shape) + pts - head_cont.reshape(shape)) * z + cc

        rad = radius[d]
        r_px = px_r[d]
        pad = np.maximum(1.0, r_px * 0.12) / z                     # contorno: >= 1 píxel
        body_ink_a = to_canvas(rig.outline(s, rad, pad))
        shadow_off = SHADOW_OFFSET[None, None, :] * r_px[:, None, None]
        # pygame recorre listas de Python mucho más rápido que arrays de numpy
        body = to_canvas(rig.outline(s, rad)).tolist()
        body_ink = body_ink_a.tolist()
        shadows = (body_ink_a + shadow_off).tolist()
        head_a = to_canvas(rig.head(s, rad))
        head = head_a.tolist()
        head_ink = to_canvas(rig.head(s, rad, pad)).tolist()
        light = to_canvas(rig.back_highlight(s, rad, LIGHT_DIR)).tolist()
        hip, knee, foot, toes = rig.legs(s, rad, x_acc[d])
        legs = np.stack([to_canvas(hip), to_canvas(knee), to_canvas(foot)], axis=2).tolist()
        toes = np.concatenate([to_canvas(foot)[:, :, None], to_canvas(toes)], axis=2).tolist()
        gills = to_canvas(rig.gills(s, rad, x_mate[d], time)).tolist()
        spot_pos, spot_r = rig.spots(s, rad)
        spot_pos = to_canvas(spot_pos).tolist()
        spot_r = np.maximum(1.0, spot_r * z).tolist()
        eye_l, eye_r, eye_rad = rig.eyes(s, rad, x_det[d])
        eyes = np.stack([to_canvas(eye_l[:, None])[:, 0], to_canvas(eye_r[:, None])[:, 0]],
                        axis=1).tolist()
        eye_px = np.maximum(1.0, eye_rad * z).tolist()
        eye_polys = to_canvas(rig.eye_shapes(s, rad, x_det[d], x_agg[d])).tolist()
        ears = to_canvas(rig.ears(s, rad, x_agg[d])).tolist()
        fangs = to_canvas(rig.fangs(s, rad, x_diet[d])).tolist()
        carnivore = (x_diet[d] > FANG_MIN_DIET).tolist()
        head_glow = (head_a.mean(axis=1) + LIGHT_DIR * r_px[:, None] * 0.35).tolist()
        bodies = pal.to_rgb_list(self._bodies(species[d], energy_frac[d]))
        mates = x_mate[d].tolist()
        sp_list = species[d].tolist()
        r_list = r_px.tolist()

        shadow_color = (*pal.INK, pal.SHADOW_ALPHA)
        for poly in shadows:
            pygame.draw.polygon(shadow, shadow_color, poly)

        ink = pal.INK
        for i, r in enumerate(r_list):
            c = self.colors[sp_list[i]]
            # patas y garras (debajo del cuerpo)
            leg_w = max(1, int(round(r * 0.3)))
            for k in range(4):
                pygame.draw.lines(canvas, ink, False, legs[i][k], leg_w)
                if r >= 7:
                    f0, *claws = toes[i][k]
                    for toe in claws:
                        pygame.draw.line(canvas, ink, f0, toe)
            # branquias pastel (salen de debajo de la cabeza): de color, para no
            # confundirse con las patas de tinta
            if mates[i] > 0.05:
                gw = max(1, int(round(r * 0.14)))
                for strand in gills[i]:
                    pygame.draw.lines(canvas, c.accent_light, False, strand, gw)
                    if r >= 5:
                        pygame.draw.circle(canvas, c.accent, strand[-1], max(1.0, r * 0.12))
            # orejas (debajo de la cabeza): la forma la pone Agresividad
            if r >= 4:
                for ear in ears[i]:
                    pygame.draw.polygon(canvas, c.shade, ear)
                    pygame.draw.polygon(canvas, ink, ear, 1)
            # silueta de tinta, cuerpo, volumen y manchas
            pygame.draw.polygon(canvas, ink, body_ink[i])
            pygame.draw.polygon(canvas, ink, head_ink[i])
            pygame.draw.polygon(canvas, bodies[i], body[i])
            if r >= 5:
                pygame.draw.polygon(canvas, c.light, light[i])
                for p, sr in zip(spot_pos[i], spot_r[i], strict=True):
                    pygame.draw.circle(canvas, c.shade, p, sr)
            # cabeza de color vivo y ojos de tinta
            pygame.draw.polygon(canvas, c.accent, head[i])
            if r >= 6:
                pygame.draw.circle(canvas, c.accent_light, head_glow[i], max(1.0, r * 0.2))
            # colmillos de carnívoro (Dieta)
            if carnivore[i] and r >= 5:
                for fang in fangs[i]:
                    pygame.draw.polygon(canvas, pal.PAPER, fang)
                    pygame.draw.polygon(canvas, ink, fang, 1)
            # ojos: redondos si es dócil, rasgados si es agresivo
            er = eye_px[i]
            for e, poly in zip(eyes[i], eye_polys[i], strict=True):
                if r >= 6:
                    pygame.draw.polygon(canvas, ink, poly)
                else:
                    pygame.draw.circle(canvas, ink, e, er)
                if r >= 8:
                    pygame.draw.circle(canvas, pal.PAPER, (e[0] - er * 0.3, e[1] - er * 0.4),
                                       max(1.0, er * 0.3))

    def _bodies(self, species: np.ndarray, energy_frac: np.ndarray) -> np.ndarray:
        out = np.empty((len(species), 3))
        for sp in np.unique(species):
            m = species == sp
            out[m] = pal.hungry_body(self.colors[sp].body, energy_frac[m])
        return out

    def _draw_tiny(self, canvas: pygame.Surface, pos: np.ndarray, species: np.ndarray) -> None:
        """Lejos: cada criatura es un píxel del color de su cabeza con un píxel de sombra."""
        w, h = canvas.get_size()
        cols = np.floor(pos[:, 0]).astype(int)
        rows = np.floor(pos[:, 1]).astype(int)
        ok = (cols >= 0) & (cols < w - 1) & (rows >= 0) & (rows < h - 1)
        cols, rows, species = cols[ok], rows[ok], species[ok]
        accent = np.array([c.accent for c in self.colors])[species]
        px = pygame.surfarray.pixels3d(canvas)
        px[cols + 1, rows + 1] = pal.INK
        px[cols, rows] = accent
        del px                                      # libera el bloqueo de la superficie
