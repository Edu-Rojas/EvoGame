"""El visor completo dibuja sin ventana (SDL dummy): atrapa regresiones de render."""
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")


@pytest.fixture
def viewer(cfg):
    from especies.state import create_world
    from viewer.app import Viewer
    pygame.init()
    v = Viewer(create_world(cfg, seed=5), tps=20)
    for _ in range(60):
        v.advance(1 / 20)
    yield v
    pygame.quit()


@pytest.mark.parametrize("zoom", [1.0, 4.0, 12.0])
def test_draws_at_every_level_of_detail(viewer, zoom):
    viewer.cam.zoom = min(viewer.cam.min_zoom * zoom, 12.0)
    viewer.cam.center = viewer.w.pos[viewer.w.alive_idx()[0]].copy()
    for _ in range(3):
        viewer.time += 1 / 60
        viewer.draw(1 / 60)
    # algo se dibujó: el lienzo no es de un solo color
    px = pygame.surfarray.array3d(viewer.canvas)
    assert len(np.unique(px.reshape(-1, 3), axis=0)) > 10


def test_selection_card_and_pixel_switch(viewer):
    viewer.selected_uid = int(viewer.w.uid[viewer.w.alive_idx()[0]])
    viewer.draw(1 / 60)
    viewer.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
    viewer.draw(1 / 60)
    assert viewer.pixel != 3
