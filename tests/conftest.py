import sys
from dataclasses import replace
from pathlib import Path

import pytest

# Permite correr pytest sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from especies.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def small_cfg(cfg):
    """Mundo chico y rápido para tests de integración."""
    return replace(cfg, sim=replace(cfg.sim, capacity=400))
