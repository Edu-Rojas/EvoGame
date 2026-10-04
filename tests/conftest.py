from dataclasses import replace

import pytest

from especies.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def small_cfg(cfg):
    """Mundo chico y rápido para tests de integración."""
    return replace(cfg, sim=replace(cfg.sim, capacity=400))
