"""Animación procedural estilo Rain World: columna que sigue a la cabeza + patas con IK.

Esto es SOLO visual: vive en el visor, no en la simulación. La simulación solo sabe
"esta criatura está en (x, y) mirando hacia θ"; el cuerpo, las patas y las plumas
salen de aquí. Todo está vectorizado: cada paso se hace para todas las criaturas a
la vez (las filas son slots del World).

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

N_SEG = 5
SEG_RADII = np.array([0.9, 1.0, 0.95, 0.75, 0.5])  # cabeza, pecho, ..., cola
LEG_SEG = np.array([1, 1, 3, 3])                     # en qué segmento nace cada pata
LEG_SIDE = np.array([1.0, -1.0, 1.0, -1.0])          # izquierda / derecha
LEG_GROUP = np.array([0, 1, 1, 0])                   # diagonales: (DI, TD) y (DD, TI)
N_PLUMES = 3
MAX_BEND = np.radians(40)                            # doblez máximo entre segmentos


def _torus_delta(frm, to, size):
    d = to - frm
    return d - size * np.round(d / size)


def _perp(v):
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


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
    def body_params(radius, x_acc):
        spacing = radius * (0.7 + 0.45 * x_acc)          # aceleración = cuerpo más largo
        leg_len = radius * (1.1 + 0.7 * x_acc)           # y patas más largas (guepardo)
        return spacing, leg_len

    # ---------- ciclo de vida del rig ----------
    def sync(self, slots, uids, head, heading, radius, x_acc):
        """Crea el rig de las criaturas nuevas (o de slots reutilizados por otra)."""
        new = self.uid[slots] != uids
        if not new.any():
            return
        s = slots[new]
        spacing, _ = self.body_params(radius[new], x_acc[new])
        back = -np.stack([np.cos(heading[new]), np.sin(heading[new])], axis=1)
        k = np.arange(N_SEG)[None, :, None]
        self.spine[s] = head[new][:, None, :] + back[:, None, :] * spacing[:, None, None] * k
        self.uid[s] = uids[new]
        ideal, _ = self._foot_targets(s, radius[new], x_acc[new])
        self.feet[s] = ideal
        self.step_t[s] = 1.0

    def update(self, slots, head, radius, x_acc, dt, step_rate=9.0):
        if len(slots) == 0:
            return
        spacing, leg_len = self.body_params(radius, x_acc)

        # 1) la cabeza va a la posición nueva, sin saltar por el borde del toro
        old_head = self.spine[slots, 0]
        head_cont = old_head + _torus_delta(old_head, head, self.size)
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
                over = np.arccos(cos) > MAX_BEND
                turn = np.sign(u[:, 0] * vdir[:, 1] - u[:, 1] * vdir[:, 0]) * MAX_BEND
                c, s_ = np.cos(turn), np.sin(turn)
                clamped = np.stack([u[:, 0] * c - u[:, 1] * s_, u[:, 0] * s_ + u[:, 1] * c], axis=1)
                vdir = np.where(over[:, None], clamped, vdir)
            length = np.clip(dist, spacing * 0.6, spacing)
            self.spine[slots, k] = prev + vdir * length[:, None]

        # 3) patas: ¿qué pies quedaron muy lejos de donde deberían estar?
        ideal, fwd = self._foot_targets(slots, radius, x_acc)
        feet = self.feet[slots]
        stepping = self.step_t[slots] < 1.0
        off = np.linalg.norm(feet - ideal, axis=2)
        lost = off > leg_len[:, None] * 2.5                 # se teletransportó: reubicar
        feet[lost] = ideal[lost]
        group_busy = np.stack([
            (stepping & (LEG_GROUP == g)).any(axis=1) for g in (0, 1)], axis=1)
        other_busy = group_busy[:, 1 - LEG_GROUP]           # [n, 4]
        start = (~stepping) & (off > leg_len[:, None] * 0.65) & (~other_busy)
        # solo una diagonal a la vez: si las dos quieren, gana la más atrasada
        both = start[:, LEG_GROUP == 0].any(axis=1) & start[:, LEG_GROUP == 1].any(axis=1)
        need = np.stack([np.where(LEG_GROUP == g, off, 0).max(axis=1) for g in (0, 1)], axis=1)
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

    # ---------- geometría derivada (para dibujar) ----------
    def _foot_targets(self, slots, radius, x_acc):
        """Dónde 'quiere' estar cada pie: adelante y hacia afuera de su cadera."""
        _, leg_len = self.body_params(radius, x_acc)
        sp = self.spine[slots]
        fwd = _normalize(sp[:, :-2] - sp[:, 2:])            # dirección de cada segmento
        fwd = np.concatenate([fwd[:, :1], fwd, fwd[:, -1:]], axis=1)  # [n, N_SEG, 2]
        f = fwd[:, LEG_SEG]
        side = _perp(f) * LEG_SIDE[None, :, None]
        hip = sp[:, LEG_SEG] + side * radius[:, None, None] * 0.55
        ideal = hip + f * leg_len[:, None, None] * 0.3 + side * leg_len[:, None, None] * 0.75
        return ideal, fwd

    def legs(self, slots, radius, x_acc):
        """Cadera, rodilla y pie de cada pata. La rodilla por IK de dos huesos."""
        _, leg_len = self.body_params(radius, x_acc)
        sp = self.spine[slots]
        fwd = _normalize(sp[:, :-2] - sp[:, 2:])
        fwd = np.concatenate([fwd[:, :1], fwd, fwd[:, -1:]], axis=1)
        f = fwd[:, LEG_SEG]
        side = _perp(f) * LEG_SIDE[None, :, None]
        hip = sp[:, LEG_SEG] + side * radius[:, None, None] * 0.55
        foot = self.feet[slots]

        bone = leg_len[:, None] * 0.6                        # muslo = canilla
        d = foot - hip
        dist = np.clip(np.linalg.norm(d, axis=2), 1e-6, 2 * bone * 0.999)
        dir_ = d / np.linalg.norm(d, axis=2, keepdims=True).clip(1e-9)
        # ley del coseno con huesos iguales: la rodilla está a mitad de camino,
        # desplazada en perpendicular sqrt(hueso^2 - (dist/2)^2), doblada hacia afuera
        along = dist / 2
        h = np.sqrt(np.maximum(bone ** 2 - along ** 2, 0.0))
        bend = _perp(dir_) * (-LEG_SIDE)[None, :, None]
        knee = hip + dir_ * along[..., None] + bend * h[..., None]
        return hip, knee, foot

    def outline(self, slots, radius):
        """Silueta suave alrededor de la columna: borde izquierdo, punta de la cabeza,
        borde derecho y punta de la cola. Un solo polígono por criatura."""
        sp = self.spine[slots]
        fwd = _normalize(sp[:, :-2] - sp[:, 2:])
        fwd = np.concatenate([fwd[:, :1], fwd, fwd[:, -1:]], axis=1)    # [n, N_SEG, 2]
        side = _perp(fwd)
        width = radius[:, None, None] * SEG_RADII[None, :, None]
        left = sp + side * width
        right = sp - side * width
        # punta redonda de la cabeza: semicírculo de 5 puntos delante del segmento 0
        ang = np.linspace(np.pi / 2, -np.pi / 2, 7)[1:-1]
        f0, s0, w0 = fwd[:, 0], side[:, 0], width[:, 0]
        cap = (sp[:, 0, None, :] + (f0[:, None, :] * np.cos(ang)[None, :, None]
                                     + s0[:, None, :] * np.sin(ang)[None, :, None]) * w0[:, None, :])
        tip = sp[:, -1] - fwd[:, -1] * width[:, -1] * 1.6               # cola en punta
        return np.concatenate([left[:, ::-1], cap, right, tip[:, None]], axis=1)

    def plumes(self, slots, radius, x_mate, t):
        """Plumas largas de la cola (gen Apareamiento) que se mecen al caminar."""
        sp = self.spine[slots]
        tail = sp[:, -1] + _normalize(sp[:, -1] - sp[:, -2]) * radius[:, None] * SEG_RADII[-1] * 1.4
        back = _normalize(sp[:, -1] - sp[:, -2])
        side = _perp(back)
        length = radius * (0.5 + 2.5 * x_mate)
        s = np.linspace(0, 1, 7)[None, None, :, None]            # a lo largo de la pluma
        spread = np.array([-0.35, 0.0, 0.35])[None, :, None, None]
        phase = (slots[:, None] * 0.7 + np.arange(N_PLUMES)[None, :] * 1.3)[..., None, None]
        sway = 0.18 * np.sin(t * 3.0 + phase + s * 2.5)
        L = length[:, None, None, None]
        pts = (tail[:, None, None, :]
               + back[:, None, None, :] * L * s
               + side[:, None, None, :] * L * s * (spread + sway))
        return pts                                                # [n, plumas, 7, 2]

    def eyes(self, slots, radius, x_det):
        sp = self.spine[slots]
        f = _normalize(sp[:, 0] - sp[:, 1])
        side = _perp(f)
        r0 = radius * SEG_RADII[0]
        center = sp[:, 0] + f * r0[:, None] * 0.35
        er = radius * (0.14 + 0.2 * x_det)
        left = center + side * r0[:, None] * 0.45
        right = center - side * r0[:, None] * 0.45
        return left, right, er
