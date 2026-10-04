"""El rig de animación y la paleta son numpy puro: se testean sin ventana."""
import numpy as np
import pytest

from viewer import palette as pal
from viewer.animation import N_SEG, SEG_LEN, Rig


@pytest.fixture
def walking_rig():
    size = np.array([1600.0, 1000.0])
    rig = Rig(capacity=4, world_size=size)
    slots, uids = np.array([0, 1]), np.array([10, 11])
    radius, x_acc = np.array([4.0, 8.0]), np.array([0.2, 0.9])
    head = np.array([[100.0, 100.0], [1595.0, 500.0]])   # la segunda cruza el borde
    rig.sync(slots, uids, head, np.array([0.0, 0.0]), radius, x_acc)
    for _ in range(60):                       # caminar en +x
        head = (head + [2.0, 0.0]) % size
        rig.update(slots, head, radius, x_acc, dt=1 / 60)
    return rig, slots, radius, x_acc


def test_spine_links_never_stretch(walking_rig):
    rig, slots, radius, x_acc = walking_rig
    spacing, _ = Rig.body_params(radius, x_acc)
    links = np.linalg.norm(np.diff(rig.spine[slots], axis=1), axis=2)
    assert (links <= spacing[:, None] * SEG_LEN[None, 1:] + 1e-6).all()


def test_geometry_is_finite_and_well_shaped(walking_rig):
    rig, slots, radius, x_acc = walking_rig
    assert np.isfinite(rig.outline(slots, radius)).all()
    assert np.isfinite(rig.head(slots, radius, pad=1.0)).all()
    hip, knee, foot, toes = rig.legs(slots, radius, x_acc)
    assert hip.shape == knee.shape == foot.shape == (2, 4, 2) and toes.shape == (2, 4, 3, 2)
    assert rig.gills(slots, radius, np.array([0.5, 1.0]), t=1.0).shape[:2] == (2, 6)
    assert rig.spine.shape[1] == N_SEG


def test_padded_outline_is_bigger(walking_rig):
    rig, slots, radius, _ = walking_rig
    area = []
    for pad in (0.0, 1.0):
        p = rig.outline(slots, radius, pad)[0]
        x, y = p[:, 0], p[:, 1]
        area.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    assert area[1] > area[0]


def test_new_creature_in_reused_slot_gets_a_fresh_rig(walking_rig):
    rig, _, radius, x_acc = walking_rig
    head = np.array([[800.0, 800.0]])
    rig.sync(np.array([0]), np.array([99]), head, np.array([1.0]), radius[:1], x_acc[:1])
    assert rig.uid[0] == 99
    assert np.allclose(rig.spine[0, 0], head[0])


def test_hunger_greys_out_the_body():
    full, empty = pal.hungry_body((120, 220, 140), np.array([1.0, 0.0]))
    spread = lambda c: c.max() - c.min()  # noqa: E731  (saturación aproximada)
    assert spread(empty) < spread(full)


def test_species_palette_is_valid_rgb():
    c = pal.species_colors((255, 170, 90))
    for color in (c.body, c.light, c.shade, c.accent, c.accent_light):
        assert len(color) == 3 and all(0 <= v <= 255 for v in color)


def test_creature_card_formats_for_players(cfg):
    from especies.metrics import describe_creature
    from especies.state import create_world
    from viewer.app import creature_card
    w = create_world(cfg, seed=0)
    lines = creature_card(describe_creature(w, int(w.alive_idx()[0])))
    assert lines[0].startswith("criatura #") and any("fundador" in line for line in lines)
