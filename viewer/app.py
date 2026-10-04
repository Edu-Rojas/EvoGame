"""Visor de desarrollo con pygame. Solo LEE el estado de la simulación.

Controles:
    rueda del mouse     zoom (hacia donde apunta el cursor)
    clic derecho + arrastrar, o WASD   mover la cámara
    clic izquierdo      seleccionar criatura (ficha + radio de detección)
    F                   la cámara sigue a la criatura seleccionada
    P                   tamaño del píxel (2, 3 o 4)
    ESPACIO             pausa / continúa
    ↑ / ↓               más o menos ticks por segundo
    ESC                 salir

Uso:
    python -m viewer
    python -m viewer --seed 7 --tps 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame

from especies.config import load_config
from especies.genes import Gene, norm
from especies.geometry import torus_delta, wrap
from especies.metrics import describe_creature
from especies.state import World, create_world
from especies.step import step
from especies.terrain import Biome

from . import palette as pal
from .animation import Rig
from .render import CreatureRenderer, TerrainRenderer, View
from .shapes import growth

MAX_SCREEN = np.array([1400.0, 860.0])
MAX_ZOOM = 12.0
PIXEL_SIZES = (2, 3, 4)
MAX_TICKS_PER_FRAME = 50
# Etiquetas de la ficha de criatura en el HUD (la simulación devuelve claves neutras)
CREATURE_LABELS = {
    "uid": "uid", "species": "especie", "generation": "generacion", "parents": "padres",
    "genes": "genes", "instincts": "instintos", "energy": "energia", "age": "edad",
    "action": "accion",
}


class Camera:
    """Cámara sobre un mundo toroidal: centro (en coordenadas de mundo) + zoom."""

    def __init__(self, world_size: np.ndarray):
        self.size = world_size
        self.min_zoom = float(min(MAX_SCREEN / world_size))
        self.screen = np.floor(world_size * self.min_zoom)
        self.zoom = self.min_zoom
        self.center = world_size / 2

    def rel(self, p: np.ndarray) -> np.ndarray:
        """Posición relativa al centro de cámara, por el camino corto del toro."""
        return torus_delta(self.center, p, self.size)

    def to_world(self, s) -> np.ndarray:
        return wrap(self.center + (np.asarray(s, float) - self.screen / 2) / self.zoom, self.size)

    def zoom_at(self, s, factor: float) -> None:
        anchor = self.to_world(s)
        self.zoom = float(np.clip(self.zoom * factor, self.min_zoom, MAX_ZOOM))
        self.center = wrap(anchor - (np.asarray(s, float) - self.screen / 2) / self.zoom, self.size)

    def pan(self, ds) -> None:
        self.center = wrap(self.center - np.asarray(ds, float) / self.zoom, self.size)

    @property
    def half_extent(self) -> np.ndarray:
        return self.screen / 2 / self.zoom


class Viewer:
    def __init__(self, world: World, tps: float):
        self.w = world
        self.tps = tps
        self.paused = False
        self.follow = False
        self.selected_uid = -1
        self.acc = 0.0
        self.prev_pos = world.pos.copy()
        self.prev_uid = world.uid.copy()
        self.time = 0.0
        cfg = world.cfg
        self.cam = Camera(world.size)
        self.screen = pygame.display.set_mode(tuple(int(v) for v in self.cam.screen))
        pygame.display.set_caption("Guerra de especies")
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.font_bold = pygame.font.SysFont("consolas,menlo,monospace", 15, bold=True)
        self.clock = pygame.time.Clock()
        self.rig = Rig(cfg.sim.capacity, world.size)
        self.maturity = cfg.reproduction.maturity_fraction * cfg.body.base_lifespan
        # rng propio del visor: si usara el de la simulación, abrir el visor cambiaría
        # la simulación (y rompería la semilla)
        self.terrain_view = TerrainRenderer(world, np.random.default_rng(0))
        self.creature_view = CreatureRenderer([sp.color for sp in cfg.species])
        self.set_pixel(3)

    def set_pixel(self, size: int) -> None:
        self.pixel = size
        canvas = tuple(int(np.ceil(v / size)) for v in self.cam.screen)
        self.canvas = pygame.Surface(canvas)
        self.shadow = pygame.Surface(canvas, pygame.SRCALPHA)

    def view(self) -> View:
        return View(center=self.cam.center, zoom=self.cam.zoom / self.pixel,
                    canvas_center=self.cam.screen / 2 / self.pixel,
                    canvas_size=self.canvas.get_size())

    # ---------- entrada ----------
    def handle(self, ev) -> bool:
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.paused = not self.paused
                if not self.paused:
                    # en pausa se dibuja el tick actual; al seguir, se corre uno ya y
                    # se interpola desde ahí (si no, el mundo saltaría un tick atrás)
                    self.acc = 1.0
            elif ev.key == pygame.K_UP:
                self.tps = min(self.tps * 1.5, 600)
            elif ev.key == pygame.K_DOWN:
                self.tps = max(self.tps / 1.5, 1)
            elif ev.key == pygame.K_f:
                self.follow = not self.follow
            elif ev.key == pygame.K_p:
                i = PIXEL_SIZES.index(self.pixel)
                self.set_pixel(PIXEL_SIZES[(i + 1) % len(PIXEL_SIZES)])
        elif ev.type == pygame.MOUSEWHEEL:
            self.cam.zoom_at(pygame.mouse.get_pos(), 1.15 ** ev.y)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.select(self.cam.to_world(ev.pos))
        elif ev.type == pygame.MOUSEMOTION and ev.buttons[2]:
            self.follow = False
            self.cam.pan(ev.rel)
        return True

    def handle_keys(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        speed = 600 * dt
        move = np.array([keys[pygame.K_a] - keys[pygame.K_d], keys[pygame.K_w] - keys[pygame.K_s]], float)
        if move.any():
            self.follow = False
            self.cam.pan(move * speed)

    def select(self, world_pos: np.ndarray) -> None:
        a = self.w.alive_idx()
        if len(a) == 0:
            return
        d = np.linalg.norm(torus_delta(self.w.pos[a], world_pos, self.w.size), axis=1)
        i = int(np.argmin(d))
        self.selected_uid = int(self.w.uid[a[i]]) if d[i] * self.cam.zoom < 25 else -1

    # ---------- simulación ----------
    def advance(self, dt: float) -> None:
        """Corre ticks a `tps` por segundo. Guarda la posición anterior para interpolar:
        la simulación avanza a saltos (20 ticks/s) pero se dibuja a 60 fps."""
        if self.paused:
            return
        self.acc += dt * self.tps
        n = min(int(self.acc), MAX_TICKS_PER_FRAME)  # tope por frame para no congelar la ventana
        self.acc -= n
        # Si la simulación no da abasto, se descarta el atraso: si no, al bajar la
        # velocidad seguiría corriendo 50 ticks por frame hasta vaciarlo
        self.acc = min(self.acc, 1.0)
        for _ in range(n):
            self.prev_pos = self.w.pos.copy()
            self.prev_uid = self.w.uid.copy()
            step(self.w)

    def interpolated_heads(self, slots: np.ndarray) -> np.ndarray:
        alpha = np.full(len(slots), 1.0 if self.paused else min(self.acc, 1.0))
        # un slot reutilizado por una cría guarda la posición del muerto anterior:
        # la cría se dibuja directo donde nació
        alpha[self.prev_uid[slots] != self.w.uid[slots]] = 1.0
        prev, cur = self.prev_pos[slots], self.w.pos[slots]
        return wrap(prev + torus_delta(prev, cur, self.w.size) * alpha[:, None], self.w.size)

    # ---------- criaturas ----------
    def draw_creatures(self, dt: float) -> None:
        w, cam = self.w, self.cam
        a = w.alive_idx()
        if len(a) == 0:
            return
        g = w.genes[a]
        x_acc = norm(g[:, Gene.ACCELERATION])
        radius = w.radius[a] * growth(w.age[a], self.maturity)
        heads = self.interpolated_heads(a)

        self.rig.sync(a, w.uid[a], heads, w.heading[a], radius, x_acc)
        self.rig.update(a, heads, radius, x_acc, 0.0 if self.paused else dt)

        # Cámara: base = dónde cae la cabeza respecto al centro; el resto del rig se
        # dibuja relativo a la cabeza (así nada salta al cruzar el borde del toro)
        base = cam.rel(heads)
        reach = radius * 6
        visible = np.all(np.abs(base) < cam.half_extent + reach[:, None], axis=1)
        if not visible.any():
            return
        a, base, radius, g = a[visible], base[visible], radius[visible], g[visible]
        self.creature_view.draw(
            self.canvas, self.shadow, self.view(), self.rig, a, w.species[a], base, radius,
            energy_frac=w.energy[a] / w.reserve[a],
            x_acc=norm(g[:, Gene.ACCELERATION]), x_det=norm(g[:, Gene.DETECTION]),
            x_mate=norm(g[:, Gene.MATING]), time=self.time)

    # ---------- dibujo general ----------
    def selected_slot(self) -> int | None:
        if self.selected_uid < 0:
            return None
        hit = np.flatnonzero(self.w.alive & (self.w.uid == self.selected_uid))
        if len(hit) == 0:
            self.selected_uid = -1   # murió
            self.follow = False
            return None
        return int(hit[0])

    def draw(self, dt: float) -> None:
        sel = self.selected_slot()
        if self.follow and sel is not None:
            self.cam.center = self.interpolated_heads(np.array([sel]))[0]
        v = self.view()
        self.terrain_view.draw(self.canvas, v, self.time)
        self.shadow.fill((0, 0, 0, 0))
        self.draw_creatures(dt)
        self.canvas.blit(self.shadow, (0, 0))
        if sel is not None:
            self._draw_detection_ring(sel, v)
        size = (self.canvas.get_width() * self.pixel, self.canvas.get_height() * self.pixel)
        self.screen.blit(pygame.transform.scale(self.canvas, size), (0, 0))
        self.draw_hud(sel)
        pygame.display.flip()

    def _draw_detection_ring(self, sel: int, v: View) -> None:
        w = self.w
        vis = w.terrain.visibility[w.terrain.cell_of(w.pos[sel])]
        head = self.interpolated_heads(np.array([sel]))[0]
        p = self.cam.rel(head) * v.zoom + v.canvas_center
        r = w.det_radius[sel] * vis * v.zoom
        # anillo punteado a mano: arcos cortos alternados
        n = max(12, int(r / 3))
        for k in range(0, n, 2):
            a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
            pygame.draw.arc(self.canvas, pal.PAPER, (p[0] - r, p[1] - r, 2 * r, 2 * r), a0, a1)

    # ---------- HUD: fichas de papel con borde de tinta ----------
    def _panel(self, lines: list[tuple[str, pal.Color]], pos: tuple[int, int],
               chips: list[pal.Color | None] | None = None) -> None:
        pad, lh = 10, 18
        width = max(self.font.size(t)[0] for t, _ in lines) + 2 * pad + (18 if chips else 0)
        height = len(lines) * lh + 2 * pad
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((*pal.PAPER, 235))
        self.screen.blit(panel, pos)
        pygame.draw.rect(self.screen, pal.INK, (*pos, width, height), 2)
        y = pos[1] + pad
        for j, (text, color) in enumerate(lines):
            x = pos[0] + pad
            chip = chips[j] if chips else None
            if chip is not None:
                pygame.draw.rect(self.screen, chip, (x, y + 3, 11, 11))
                pygame.draw.rect(self.screen, pal.INK, (x, y + 3, 11, 11), 1)
                x += 18
            font = self.font_bold if j == 0 else self.font
            self.screen.blit(font.render(text, True, color), (x, y))
            y += lh

    def draw_hud(self, sel: int | None) -> None:
        w = self.w
        mouse_cell = w.terrain.cell_of(self.cam.to_world(pygame.mouse.get_pos()))
        biome = Biome(int(w.terrain.biome[mouse_cell])).key
        status = ("   [PAUSA]" if self.paused else "") + ("   [SIGUIENDO]" if self.follow else "")
        lines: list[tuple[str, pal.Color]] = [(f"tick {w.tick}{status}", pal.INK)]
        chips: list[pal.Color | None] = [None]
        for sp_id, sp in enumerate(w.cfg.species):
            n = int((w.alive & (w.species == sp_id)).sum())
            lines.append((f"{sp.name:<10} {n:>5}", pal.INK))
            chips.append(self.creature_view.colors[sp_id].accent)
        self._panel(lines, (10, 10), chips)

        info = (f"{self.tps:.0f} ticks/s · {self.clock.get_fps():.0f} fps · "
                f"zoom x{self.cam.zoom / self.cam.min_zoom:.1f} · píxel {self.pixel} · {biome}")
        self._panel([(info, pal.INK)], (10, int(self.cam.screen[1]) - 48))

        if sel is not None:
            card = [(f"criatura #{w.uid[sel]}", pal.INK)]
            for k, val in describe_creature(w, sel).items():
                if k != "uid":
                    card.append((f"{CREATURE_LABELS[k]}: {val}", pal.INK))
            self._panel(card, (int(self.cam.screen[0]) - 470, 10))

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000
            self.time += dt
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.handle_keys(dt)
            self.advance(dt)
            self.draw(dt)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m viewer")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tps", type=float, default=20.0, help="ticks de simulación por segundo")
    args = ap.parse_args()

    pygame.init()
    world = create_world(load_config(args.config), seed=args.seed)
    print(f"semilla: {world.seed}")
    Viewer(world, args.tps).run()
    pygame.quit()


if __name__ == "__main__":
    main()
