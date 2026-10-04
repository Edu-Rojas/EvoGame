"""Animación procedural: columna que sigue a la cabeza + patas con IK + branquias.

Esto es SOLO visual: vive en el visor, no en la simulación. La simulación solo sabe
"esta criatura está en (x, y) mirando hacia θ"; el cuerpo, las patas y las branquias
salen de aquí. Todo está vectorizado: cada paso se hace para todas las criaturas a
la vez (las filas son slots del World). Sin pygame: devuelve geometría en coordenadas
de mundo y cualquier renderer (pygame hoy, PixiJS después) la dibuja.

Las tres ideas que hacen que se vea orgánico:
1. Seguir al líder: cada segmento de la columna se queda a distancia fija del
   anterior. La cabeza manda y el cuerpo "serpentea" detrás sin animarlo a mano.
2. Pies plantados: un pie NO se mueve hasta que el cuerpo se aleja demasiado de
   donde debería estar. Entonces da un paso rápido hacia adelante. Eso da el peso.
3. Patas en diagonal: delantera izquierda + trasera derecha pisan juntas, y no
   pueden dar el paso mientras la otra diagonal está en el aire (como un lagarto).
La rodilla sale de la cinemática inversa de dos huesos (ley del coseno).

Coordenadas: el rig vive "desenvuelto" alrededor de la cabeza (sin saltos de borde
del toro), y se re-centra cuando la cabeza cruza un borde.
"""
from __future__ import annotations

import numpy as np

from especies.geometry import torus_delta

# ---------- plan del cuerpo (en múltiplos del radio de la criatura) ----------
#              cabeza cuello pecho vientre cadera cola  punta
SEG_RADII = np.array([0.70, 0.42, 0.80, 0.84, 0.58, 0.24, 0.12])
SEG_LEN = np.array([0.0, 0.75, 0.90, 0.95, 0.95, 1.20, 1.35])   # largo del eslabón k-1 -> k
MAX_BEND = np.radians([0, 0, 30, 30, 35, 45, 55])              # la cola latiguea más
N_SEG = len(SEG_RADII)
LEG_SEG = np.array([2, 2, 4, 4])                     # patas en el pecho y la cadera
LEG_SIDE = np.array([1.0, -1.0, 1.0, -1.0])          # izquierda / derecha
LEG_GROUP = np.array([0, 1, 1, 0])                   # diagonales: (DI, TD) y (DD, TI)
TOE_ANGLES = np.radians([-35.0, 0.0, 35.0])
HEAD_SCALE = 1.1                                     # la cabeza sobresale del cuello
# Silueta de la cabeza en coordenadas locales (x hacia adelante, y hacia la izquierda):
# cráneo ancho atrás y hocico largo y redondeado adelante, como una lagartija
HEAD_SHAPE = np.array([
    (1.45, 0.00), (1.30, 0.32), (0.85, 0.58), (0.15, 0.80), (-0.45, 0.78),
    (-0.80, 0.42), (-0.90, 0.00), (-0.80, -0.42), (-0.45, -0.78), (0.15, -0.80),
    (0.85, -0.58), (1.30, -0.32),
])
GILL_ANGLES = np.radians([30.0, 55.0, 80.0])        # abanico de branquias por lado
N_GILL_PTS = 6
BACK_SPOTS = np.array([2, 3, 4])                    # segmentos con manchas en el lomo
EYE_PTS = 8                                         # vértices de cada ojo


def _perp(v: np.ndarray) -> np.ndarray:
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _rotate(v: np.ndarray, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], axis=-1)


class Rig:
    def __init__(self, capacity: int, world_size: np.ndarray):
        self.size = world_size
        self.uid = np.full(capacity, -1, dtype=np.int64)
        self.spine = np.zeros((capacity, N_SEG, 2))
        self.feet = np.zeros((capacity, 4, 2))
        self.step_from = np.zeros((capacity, 4, 2))
        self.step_to = np.zeros((capacity, 4, 2))
        self.step_t = np.ones((capacity, 4))  # 1 = pie plantado

    # ---------- parámetros visuales por criatura ----------
    @staticmethod
    def body_params(radius: np.ndarray, x_acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spacing = radius * (0.65 + 0.4 * x_acc)          # aceleración = cuerpo más largo
        leg_len = radius * (0.9 + 0.55 * x_acc)          # y patas más largas (guepardo)
        return spacing, leg_len

    # ---------- ciclo de vida del rig ----------
    def sync(self, slots, uids, head, heading, radius, x_acc) -> None:
        """Crea el rig de las criaturas nuevas (o de slots reutilizados por otra)."""
        new = self.uid[slots] != uids
        if not new.any():
            return
        s = slots[new]
        spacing, _ = self.body_params(radius[new], x_acc[new])
        back = -np.stack([np.cos(heading[new]), np.sin(heading[new])], axis=1)
        dist = np.cumsum(SEG_LEN)[None, :, None] * spacing[:, None, None]
        self.spine[s] = head[new][:, None, :] + back[:, None, :] * dist
        self.uid[s] = uids[new]
        ideal, _ = self._foot_targets(s, radius[new], x_acc[new])
        self.feet[s] = ideal
        self.step_t[s] = 1.0

    def update(self, slots, head, radius, x_acc, dt, step_rate=9.0) -> None:
        if len(slots) == 0:
            return
        spacing, leg_len = self.body_params(radius, x_acc)

        # 1) la cabeza va a la posición nueva, sin saltar por el borde del toro
        old_head = self.spine[slots, 0]
        head_cont = old_head + torus_delta(old_head, head, self.size)
        self.spine[slots, 0] = head_cont

        # 2) seguir al líder: cada segmento a distancia fija del anterior, y con un
        #    ángulo máximo de doblez (si no, el cuerpo se dobla sobre sí mismo al girar)
        for k in range(1, N_SEG):
            prev = self.spine[slots, k - 1]
            v = self.spine[slots, k] - prev
            dist = np.linalg.norm(v, axis=1)
            vdir = v / np.maximum(dist, 1e-9)[:, None]
            if k >= 2:
                u = _normalize(prev - self.spine[slots, k - 2])   # hacia dónde iba la columna
                cos = np.clip((u * vdir).sum(axis=1), -1.0, 1.0)
                over = np.arccos(cos) > MAX_BEND[k]
                turn = np.sign(u[:, 0] * vdir[:, 1] - u[:, 1] * vdir[:, 0]) * MAX_BEND[k]
                vdir = np.where(over[:, None], _rotate(u, turn), vdir)
            link = spacing * SEG_LEN[k]
            length = np.clip(dist, link * 0.6, link)
            self.spine[slots, k] = prev + vdir * length[:, None]

        # 3) patas: ¿qué pies quedaron muy lejos de donde deberían estar?
        ideal, fwd = self._foot_targets(slots, radius, x_acc)
        feet = self.feet[slots]
        stepping = self.step_t[slots] < 1.0
        off = np.linalg.norm(feet - ideal, axis=2)
        lost = off > leg_len[:, None] * 2.5                 # se teletransportó: reubicar
        feet[lost] = ideal[lost]
        group_busy = np.stack([
            (stepping & (g == LEG_GROUP)).any(axis=1) for g in (0, 1)], axis=1)
        other_busy = group_busy[:, 1 - LEG_GROUP]           # [n, 4]
        start = (~stepping) & (off > leg_len[:, None] * 0.65) & (~other_busy)
        # solo una diagonal a la vez: si las dos quieren, gana la más atrasada
        both = start[:, LEG_GROUP == 0].any(axis=1) & start[:, LEG_GROUP == 1].any(axis=1)
        need = np.stack([np.where(g == LEG_GROUP, off, 0).max(axis=1) for g in (0, 1)], axis=1)
        loser = np.where(need[:, 0] >= need[:, 1], 1, 0)
        start &= ~(both[:, None] & (LEG_GROUP[None, :] == loser[:, None]))

        st = self.step_t[slots]
        sf, stt = self.step_from[slots], self.step_to[slots]
        sf[start] = feet[start]
        overshoot = fwd[:, LEG_SEG, :] * (leg_len[:, None, None] * 0.35)
        stt[start] = (ideal + overshoot)[start]
        st[start] = 0.0

        # 4) los pies en el aire avanzan con una curva suave (smoothstep)
        air = st < 1.0
        st[air] = np.minimum(st[air] + dt * step_rate, 1.0)
        u = st[..., None]
        u = u * u * (3 - 2 * u)
        feet = np.where(air[..., None], sf + (stt - sf) * u, feet)

        self.feet[slots], self.step_t[slots] = feet, st
        self.step_from[slots], self.step_to[slots] = sf, stt

        # 5) re-centrar si la cabeza cruzó un borde del mundo
        shift = head_cont - np.mod(head_cont, self.size)
        if np.any(shift):
            self.spine[slots] -= shift[:, None, :]
            self.feet[slots] -= shift[:, None, :]
            self.step_from[slots] -= shift[:, None, :]
            self.step_to[slots] -= shift[:, None, :]

    # ---------- marco local de la columna ----------
    def _frame(self, slots) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Columna, dirección de avance de cada segmento y su perpendicular (izquierda)."""
        sp = self.spine[slots]
        fwd = _normalize(sp[:, :-2] - sp[:, 2:])                       # diferencia centrada
        fwd = np.concatenate([_normalize(sp[:, :1] - sp[:, 1:2]), fwd, fwd[:, -1:]], axis=1)
        return sp, fwd, _perp(fwd)

    def _hips(self, sp, fwd, side, radius) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = fwd[:, LEG_SEG]
        s = side[:, LEG_SEG] * LEG_SIDE[None, :, None]
        hip = sp[:, LEG_SEG] + s * (radius[:, None, None] * SEG_RADII[LEG_SEG][None, :, None] * 0.6)
        return hip, f, s

    def _foot_targets(self, slots, radius, x_acc) -> tuple[np.ndarray, np.ndarray]:
        """Dónde 'quiere' estar cada pie: adelante y hacia afuera de su cadera."""
        _, leg_len = self.body_params(radius, x_acc)
        sp, fwd, side = self._frame(slots)
        hip, f, s = self._hips(sp, fwd, side, radius)
        ideal = hip + f * leg_len[:, None, None] * 0.3 + s * leg_len[:, None, None] * 0.75
        return ideal, fwd

    # ---------- geometría para dibujar ----------
    def legs(self, slots, radius, x_acc) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cadera, rodilla, pie y dedos de cada pata. La rodilla por IK de dos huesos.

        Dedos: [n, 4, 3, 2] puntas de tres garras que salen del pie.
        """
        _, leg_len = self.body_params(radius, x_acc)
        sp, fwd, side = self._frame(slots)
        hip, _, s = self._hips(sp, fwd, side, radius)
        foot = self.feet[slots]

        bone = leg_len[:, None] * 0.6                        # muslo = canilla
        d = foot - hip
        dist = np.clip(np.linalg.norm(d, axis=2), 1e-6, 2 * bone * 0.999)
        dir_ = _normalize(d)
        # ley del coseno con huesos iguales: la rodilla está a mitad de camino,
        # desplazada en perpendicular sqrt(hueso^2 - (dist/2)^2), doblada hacia atrás
        along = dist / 2
        h = np.sqrt(np.maximum(bone ** 2 - along ** 2, 0.0))
        bend = _perp(dir_) * (-LEG_SIDE)[None, :, None]
        knee = hip + dir_ * along[..., None] + bend * h[..., None]

        toe_dir = _normalize(foot - knee)[:, :, None, :]
        toes = foot[:, :, None, :] + _rotate(toe_dir, TOE_ANGLES[None, None, :]) * (
            leg_len[:, None, None, None] * 0.16)
        return hip, knee, foot, toes

    def outline(self, slots, radius, pad=0.0) -> np.ndarray:
        """Silueta del cuerpo (cuello a punta de la cola), un polígono por criatura.

        `pad` agranda la silueta hacia afuera (en unidades de mundo): dibujar primero
        la silueta agrandada en tinta y encima la normal da el contorno de tinta.
        """
        sp, fwd, side = self._frame(slots)
        pad = np.broadcast_to(np.asarray(pad, float), (len(sp),))
        width = radius[:, None, None] * SEG_RADII[None, :, None] + pad[:, None, None]
        left = sp + side * width
        right = sp - side * width
        # punta redonda adelante: semicírculo de 5 puntos delante del segmento 0
        ang = np.linspace(np.pi / 2, -np.pi / 2, 7)[1:-1]
        f0, s0, w0 = fwd[:, 0], side[:, 0], width[:, 0]
        cap = (sp[:, 0, None, :] + (f0[:, None, :] * np.cos(ang)[None, :, None]
                                     + s0[:, None, :] * np.sin(ang)[None, :, None]) * w0[:, None, :])
        tip = sp[:, -1] - fwd[:, -1] * (width[:, -1] * 3.0 + pad[:, None])   # cola en punta
        return np.concatenate([left[:, ::-1], cap, right, tip[:, None]], axis=1)

    def back_highlight(self, slots, radius, light_dir) -> np.ndarray:
        """Franja de brillo sobre el lomo, corrida hacia la luz: da volumen al cuerpo."""
        sp, fwd, side = self._frame(slots)
        width = radius[:, None, None] * SEG_RADII[None, :, None] * 0.45
        shift = np.asarray(light_dir)[None, None, :] * radius[:, None, None] * 0.3
        core = sp[:, 1:-1] + shift
        left = core + side[:, 1:-1] * width[:, 1:-1]
        right = core - side[:, 1:-1] * width[:, 1:-1]
        return np.concatenate([left[:, ::-1], right], axis=1)

    def head(self, slots, radius, pad=0.0) -> np.ndarray:
        """Silueta de la cabeza (cuña de lagartija), [n, len(HEAD_SHAPE), 2]."""
        sp, fwd, side = self._frame(slots)
        pad = np.broadcast_to(np.asarray(pad, float), (len(sp),))
        rh = radius * SEG_RADII[0] * HEAD_SCALE
        local = HEAD_SHAPE[None] * rh[:, None, None]
        local = local + _normalize(HEAD_SHAPE)[None] * pad[:, None, None]
        f, s = fwd[:, 0], side[:, 0]
        center = sp[:, 0] + f * (rh * 0.15)[:, None]
        return center[:, None, :] + local[..., :1] * f[:, None, :] + local[..., 1:] * s[:, None, :]

    def eyes(self, slots, radius, x_det) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ojos chiquitos de tinta en la cabeza; Detección los agranda."""
        sp, fwd, side = self._frame(slots)
        f, s = fwd[:, 0], side[:, 0]
        rh = radius * SEG_RADII[0] * HEAD_SCALE
        center = sp[:, 0] + f * (rh * 0.55)[:, None]
        er = radius * (0.09 + 0.13 * x_det)
        left = center + s * (rh * 0.5)[:, None]
        right = center - s * (rh * 0.5)[:, None]
        return left, right, er

    def eye_shapes(self, slots, radius, x_det, x_agg) -> np.ndarray:
        """Ojos como polígonos [n, 2, EYE_PTS, 2]: redondos si es dócil, rasgados e
        inclinados hacia el hocico si es agresivo (Agresividad pone la forma, Detección
        el tamaño)."""
        sp, fwd, side = self._frame(slots)
        f, s = fwd[:, 0], side[:, 0]
        left, right, er = self.eyes(slots, radius, x_det)
        ang = np.linspace(0, 2 * np.pi, EYE_PTS, endpoint=False)
        height = 1.0 - 0.6 * x_agg                       # rasgado: más bajo que ancho
        tilt = 0.5 * x_agg                               # inclinado hacia adelante
        out = []
        for center, sign in ((left, 1.0), (right, -1.0)):
            ex = np.cos(ang)[None, :] * er[:, None] * 1.15
            ey = np.sin(ang)[None, :] * er[:, None] * height[:, None]
            c, sn = np.cos(tilt * sign)[:, None], np.sin(tilt * sign)[:, None]
            lx, ly = ex * c - ey * sn, ex * sn + ey * c
            out.append(center[:, None, :] + lx[..., None] * f[:, None, :] + ly[..., None] * s[:, None, :])
        return np.stack(out, axis=1)

    def ears(self, slots, radius, x_agg) -> np.ndarray:
        """Orejas atrás de la cabeza [n, 2, 5, 2]: redondeadas si es dócil (conejo),
        largas y puntiagudas si es agresivo (felino)."""
        sp, fwd, side = self._frame(slots)
        f, s = fwd[:, 0], side[:, 0]
        rh = radius * SEG_RADII[0] * HEAD_SCALE
        length = rh * (0.55 + 0.6 * x_agg)
        width = rh * (0.55 - 0.25 * x_agg)
        bulge = 0.75 - 0.55 * x_agg                       # redondez de los costados
        out = []
        for sign in (1.0, -1.0):
            base = sp[:, 0] - f * (rh * 0.35)[:, None] + s * (rh * 0.62 * sign)[:, None]
            # la oreja apunta hacia atrás y afuera
            d = _normalize(-f * 1.0 + s * (0.8 * sign))
            p = _perp(d) * sign
            pts = [
                base + p * (width * 0.5)[:, None],
                base + d * (length * 0.55)[:, None] + p * (width * bulge)[:, None],
                base + d * length[:, None],
                base + d * (length * 0.55)[:, None] - p * (width * bulge * 0.6)[:, None],
                base - p * (width * 0.5)[:, None],
            ]
            out.append(np.stack(pts, axis=1))
        return np.stack(out, axis=1)

    def fangs(self, slots, radius, x_diet) -> np.ndarray:
        """Colmillos en la punta del hocico [n, 2, 3, 2] (más largos con más carne)."""
        sp, fwd, side = self._frame(slots)
        f, s = fwd[:, 0], side[:, 0]
        rh = radius * SEG_RADII[0] * HEAD_SCALE
        tip = sp[:, 0] + f * (rh * (0.15 + HEAD_SHAPE[0, 0] * 0.92))[:, None]
        length = rh * (0.15 + 0.45 * x_diet)
        out = []
        for sign in (1.0, -1.0):
            base = tip + s * (rh * 0.22 * sign)[:, None]
            pts = [base - s * (rh * 0.09)[:, None], base + f * length[:, None],
                   base + s * (rh * 0.09)[:, None]]
            out.append(np.stack(pts, axis=1))
        return np.stack(out, axis=1)

    def spots(self, slots, radius) -> tuple[np.ndarray, np.ndarray]:
        """Manchas sobre el lomo: posiciones [n, k, 2] y radios [n, k]."""
        sp = self.spine[slots]
        return sp[:, BACK_SPOTS], radius[:, None] * SEG_RADII[BACK_SPOTS][None, :] * 0.28

    def gills(self, slots, radius, x_mate, t) -> np.ndarray:
        """Branquias plumosas a los lados de la cabeza (gen Apareamiento), como un ajolote.

        Se mecen con el tiempo; más Apareamiento = más largas. [n, 6, N_GILL_PTS, 2]
        """
        sp, fwd, side = self._frame(slots)
        f, s = fwd[:, 0], side[:, 0]
        rh = radius * SEG_RADII[0] * HEAD_SCALE
        length = radius * (0.4 + 2.2 * x_mate)
        strands = []
        for sign in (1.0, -1.0):
            base = sp[:, 0] - f * (rh * 0.35)[:, None] + s * (rh * 0.8 * sign)[:, None]
            for i, ang in enumerate(GILL_ANGLES):
                back = _rotate(-f, np.full(len(f), -sign * ang))     # hacia atrás y afuera
                u = np.linspace(0.0, 1.0, N_GILL_PTS)[None, :, None]
                phase = slots[:, None, None] * 0.7 + i * 1.3 + sign
                sway = 0.25 * np.sin(t * 3.0 + phase + u * 2.5) * u
                curl = -0.35 * sign * u ** 2                         # se curvan hacia la cola
                d = back[:, None, :]
                pts = (base[:, None, :] + d * length[:, None, None] * u
                       + _perp(d) * length[:, None, None] * (sway + curl))
                strands.append(pts)
        return np.stack(strands, axis=1)
