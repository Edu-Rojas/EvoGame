"""Visor de desarrollo con pygame. Solo LEE el estado de la simulación.

Controles:
    rueda del mouse     zoom (hacia donde apunta el cursor)
    clic derecho + arrastrar, o WASD   mover la cámara
    clic izquierdo      seleccionar criatura (ficha + radio de detección)
    F                   la cámara sigue a la criatura seleccionada
    ESPACIO             pausa / continúa
    ↑ / ↓               más o menos ticks por segundo
    ESC                 salir

Uso:
    python viewer/pygame_viewer.py
    python viewer/pygame_viewer.py --seed 7 --tps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from especies.config import load_config  # noqa: E402
from especies.genes import Gene, norm  # noqa: E402
from especies.metrics import describe_creature  # noqa: E402
from especies.geometry import torus_delta, wrap  # noqa: E402
from especies.state import create_world  # noqa: E402
from especies.step import step  # noqa: E402
from especies.terrain import Biome  # noqa: E402

from animation import Rig  # noqa: E402
from shapes import dim, growth, lighten  # noqa: E402

TXT = (240, 244, 248)
SHADOW = (20, 22, 26)
MAX_SCREEN = np.array([1400.0, 860.0])
MAX_ZOOM = 12.0
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
        self.screen = world_size * self.min_zoom
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
    def __init__(self, world, tps: float):
        self.w = world
        self.tps = tps
        self.paused = False
        self.follow = False
        self.selected_uid = -1
        self.acc = 0.0
        self.prev_pos = world.pos.copy()
        self.time = 0.0
        cfg = world.cfg
        self.cam = Camera(world.size)
        self.screen = pygame.display.set_mode(tuple(int(v) for v in self.cam.screen))
        pygame.display.set_caption("Guerra de especies · etapa 1a")
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.clock = pygame.time.Clock()
        self.rig = Rig(cfg.sim.capacity, world.size)
        self.maturity = cfg.reproduction.maturity_fraction * cfg.body.base_lifespan
        self._prepare_terrain_colors()

    # ---------- terreno ----------
    def _prepare_terrain_colors(self) -> None:
        t, cfg = self.w.terrain, self.w.cfg
        lush = np.array([cfg.biomes[b.key].color_lush for b in Biome], float)[t.biome]
        bare = np.array([cfg.biomes[b.key].color_bare for b in Biome], float)[t.biome]
        # variación fija por celda para que no se vea plano. rng propio: si usara el de
        # la simulación, abrir el visor cambiaría la simulación (y rompería la semilla)
        texture = np.random.default_rng(0).normal(0, 5, (len(t.biome), 1))
        self.lush, self.bare = lush + texture, bare + texture

    def _terrain_rgb(self) -> np.ndarray:
        t = self.w.terrain
        frac = np.divide(t.grass, t.grass_max, out=np.ones_like(t.grass), where=t.grass_max > 0)
        rgb = self.bare + (self.lush - self.bare) * frac[:, None]
        return np.clip(rgb, 0, 255).astype(np.uint8).reshape(t.gh, t.gw, 3)

    def draw_terrain(self) -> None:
        t, cam = self.w.terrain, self.cam
        rgb = self._terrain_rgb()
        cs, z = t.cell_size, cam.zoom
        world_px = self.w.size * z
        if world_px[0] <= 2.2 * cam.screen[0]:
            # Lejos: una imagen chica del mapa escalada y repetida (el mundo es un toro)
            surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (int(world_px[0]) + 1, int(world_px[1]) + 1))
            origin = cam.screen / 2 - cam.center * z
            ox, oy = origin % world_px
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.screen.blit(surf, (ox + dx * world_px[0], oy + dy * world_px[1]))
            return
        # Cerca: solo las celdas visibles, como rectángulos
        half = cam.half_extent
        x0, x1 = int(np.floor((cam.center[0] - half[0]) / cs)), int(np.ceil((cam.center[0] + half[0]) / cs))
        y0, y1 = int(np.floor((cam.center[1] - half[1]) / cs)), int(np.ceil((cam.center[1] + half[1]) / cs))
        side = int(np.ceil(cs * z)) + 1
        for iy in range(y0, y1 + 1):
            sy = (iy * cs - cam.center[1]) * z + cam.screen[1] / 2
            row = rgb[iy % t.gh]
            for ix in range(x0, x1 + 1):
                sx = (ix * cs - cam.center[0]) * z + cam.screen[0] / 2
                pygame.draw.rect(self.screen, tuple(int(c) for c in row[ix % t.gw]), (sx, sy, side, side))

    # ---------- entrada ----------
    def handle(self, ev) -> bool:
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif ev.key == pygame.K_UP:
                self.tps = min(self.tps * 1.5, 600)
            elif ev.key == pygame.K_DOWN:
                self.tps = max(self.tps / 1.5, 1)
            elif ev.key == pygame.K_f:
                self.follow = not self.follow
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
        n = min(int(self.acc), 50)  # tope por frame para no congelar la ventana
        self.acc -= n
        # Si la simulación no da abasto, se descarta el atraso: si no, al bajar la
        # velocidad seguiría corriendo 50 ticks por frame hasta vaciarlo
        self.acc = min(self.acc, 1.0)
        for _ in range(n):
            self.prev_pos = self.w.pos.copy()
            step(self.w)

    def interpolated_heads(self, slots: np.ndarray) -> np.ndarray:
        alpha = 0.0 if self.paused else min(self.acc, 1.0)
        prev, cur = self.prev_pos[slots], self.w.pos[slots]
        return wrap(prev + torus_delta(prev, cur, self.w.size) * alpha, self.w.size)

    # ---------- criaturas ----------
    def draw_creatures(self, dt: float) -> None:
        w, cam = self.w, self.cam
        a = w.alive_idx()
        if len(a) == 0:
            return
        g = w.genes[a]
        x_acc, x_det, x_mate = norm(g[:, Gene.ACCELERATION]), norm(g[:, Gene.DETECTION]), norm(g[:, Gene.MATING])
        radius = w.radius[a] * growth(w.age[a], self.maturity)
        heads = self.interpolated_heads(a)

        self.rig.sync(a, w.uid[a], heads, w.heading[a], radius, x_acc)
        self.rig.update(a, heads, radius, x_acc, 0.0 if self.paused else dt)

        # Cámara: base = dónde cae la cabeza en pantalla; el resto del rig se dibuja
        # relativo a la cabeza (así nada salta al cruzar el borde del toro)
        base = cam.rel(heads)
        reach = radius * 6
        visible = np.all(np.abs(base) < cam.half_extent + reach[:, None], axis=1)
        if not visible.any():
            return
        a, base, radius = a[visible], base[visible], radius[visible]
        x_acc, x_det, x_mate = x_acc[visible], x_det[visible], x_mate[visible]
        head_cont = self.rig.spine[a, 0]
        z, center = cam.zoom, cam.screen / 2

        def to_screen(pts: np.ndarray) -> np.ndarray:
            shape = (len(a),) + (1,) * (pts.ndim - 2) + (2,)
            return (base.reshape(shape) + pts - head_cont.reshape(shape)) * z + center

        energy_k = 0.5 + 0.5 * np.minimum(w.energy[a] / w.reserve[a], 1.0)
        px_radius = radius * z
        detailed = px_radius >= 2.2

        # Lejos: un punto de color por criatura (rápido)
        for i in np.flatnonzero(~detailed):
            color = w.cfg.species[w.species[a[i]]].color
            pygame.draw.circle(self.screen, color, base[i] * z + center, max(1.5, px_radius[i]))

        d = np.flatnonzero(detailed)
        if len(d) == 0:
            return
        outline = to_screen(self.rig.outline(a, radius))
        hip, knee, foot = (to_screen(p) for p in self.rig.legs(a, radius, x_acc))
        plumes = to_screen(self.rig.plumes(a, radius, x_mate, self.time))
        eye_l, eye_r, eye_r_world = self.rig.eyes(a, radius, x_det)
        eye_l, eye_r = to_screen(eye_l[:, None])[:, 0], to_screen(eye_r[:, None])[:, 0]

        for i in d:
            color = w.cfg.species[w.species[a[i]]].color
            body = dim(color, energy_k[i])
            dark = dim(color, 0.35)
            limbs = px_radius[i] >= 4.0   # patas y plumas solo si se alcanzan a ver
            if limbs and x_mate[i] > 0.05:
                for p in plumes[i]:
                    pygame.draw.lines(self.screen, lighten(color, 0.3), False, p, max(1, int(px_radius[i] * 0.35)))
            leg_w = max(1, int(px_radius[i] * 0.3))
            for k in range(4 if limbs else 0):
                pygame.draw.lines(self.screen, dark, False, (hip[i, k], knee[i, k], foot[i, k]), leg_w)
            pygame.draw.polygon(self.screen, body, outline[i])
            pygame.draw.polygon(self.screen, SHADOW, outline[i], 1)
            er = max(1.0, eye_r_world[i] * z)
            for e in (eye_l[i], eye_r[i]):
                pygame.draw.circle(self.screen, (245, 245, 240), e, er)
                pygame.draw.circle(self.screen, SHADOW, e, er * 0.45)

    # ---------- dibujo general ----------
    def selected_slot(self):
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
        self.draw_terrain()
        self.draw_creatures(dt)
        if sel is not None:
            w = self.w
            vis = w.terrain.visibility[w.terrain.cell_of(w.pos[sel])]
            p = self.cam.rel(self.interpolated_heads(np.array([sel]))[0]) * self.cam.zoom + self.cam.screen / 2
            pygame.draw.circle(self.screen, TXT, p, w.det_radius[sel] * vis * self.cam.zoom, 1)
        self.draw_hud(sel)
        pygame.display.flip()

    def draw_hud(self, sel) -> None:
        w = self.w
        mouse_cell = w.terrain.cell_of(self.cam.to_world(pygame.mouse.get_pos()))
        biome = Biome(int(w.terrain.biome[mouse_cell])).key
        lines = [
            (f"tick {w.tick}   {self.tps:.0f} ticks/s   {self.clock.get_fps():.0f} fps   "
             f"zoom x{self.cam.zoom / self.cam.min_zoom:.1f}   bajo el cursor: {biome}"
             + ("   [PAUSA]" if self.paused else "") + ("   [SIGUIENDO]" if self.follow else ""), TXT),
        ]
        for sp_id, sp in enumerate(w.cfg.species):
            n = int((w.alive & (w.species == sp_id)).sum())
            lines.append((f"{sp.name:<10} {n:>5}", sp.color))
        if sel is not None:
            lines.append(("", TXT))
            for k, v in describe_creature(w, sel).items():
                lines.append((f"{CREATURE_LABELS[k]}: {v}", TXT))
        y = 8
        for text, color in lines:
            self.screen.blit(self.font.render(text, True, SHADOW), (11, y + 1))
            self.screen.blit(self.font.render(text, True, color), (10, y))
            y += 18
        help_ = "rueda: zoom · clic der./WASD: mover · clic: elegir · F: seguir · espacio: pausa · ↑↓: velocidad"
        self.screen.blit(self.font.render(help_, True, TXT), (10, self.cam.screen[1] - 22))

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
    ap = argparse.ArgumentParser()
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
