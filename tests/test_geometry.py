import numpy as np
import pytest

from especies.geometry import torus_delta, wrap


def test_torus_delta_takes_shortest_path():
    size = np.array([100.0, 100.0])
    d = torus_delta(np.array([[95.0, 50.0]]), np.array([[5.0, 50.0]]), size)
    assert d[0, 0] == pytest.approx(10.0)   # cruza el borde en vez de ir 90 hacia atrás


def test_torus_delta_is_antisymmetric():
    size = np.array([100.0, 80.0])
    a, b = np.array([[3.0, 70.0]]), np.array([[90.0, 5.0]])
    assert np.allclose(torus_delta(a, b, size), -torus_delta(b, a, size))


def test_wrap_never_returns_the_edge():
    size = np.array([1600.0, 1000.0])
    p = wrap(np.array([[-1e-17, 1000.0], [1600.0, -0.5], [-1e-14, 3205.0]]), size)
    assert (p >= 0).all() and (p < size).all()
