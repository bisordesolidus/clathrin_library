r"""
Ворота T11 — операторы объёма и площади на правильных телах (точные значения).
"""
import numpy as np
import pytest

import lattice as lat
import volume as vol


@pytest.fixture(scope="module")
def dodeca():
    return lat.dodecahedron(edge=1.0)          # единичное ребро — сверка с формулами


# --------------------------------------------------------------------------- #
#  Точные аналитические значения для додекаэдра (ребро a=1)
# --------------------------------------------------------------------------- #
def test_dodecahedron_volume(dodeca):
    V = vol.enclosed_volume(dodeca.coords, dodeca.faces)
    exact = (15 + 7 * np.sqrt(5)) / 4                     # объём додекаэдра, a=1
    assert np.isclose(V, exact, rtol=1e-6)


def test_dodecahedron_area(dodeca):
    A = vol.surface_area(dodeca.coords, dodeca.faces)
    exact = 3 * np.sqrt(25 + 10 * np.sqrt(5))            # площадь додекаэдра, a=1
    assert np.isclose(A, exact, rtol=1e-6)


def test_volume_positive_and_scales(dodeca):
    """Объём положителен и масштабируется как ребро³."""
    assert vol.enclosed_volume(dodeca.coords, dodeca.faces) > 0
    d2 = lat.dodecahedron(edge=2.0)
    V1 = vol.enclosed_volume(dodeca.coords, dodeca.faces)
    V2 = vol.enclosed_volume(d2.coords, d2.faces)
    assert np.isclose(V2, 8 * V1, rtol=1e-6)             # (×2)³


def test_area_scales_as_edge_squared(dodeca):
    d2 = lat.dodecahedron(edge=2.0)
    A1 = vol.surface_area(dodeca.coords, dodeca.faces)
    A2 = vol.surface_area(d2.coords, d2.faces)
    assert np.isclose(A2, 4 * A1, rtol=1e-6)             # (×2)²


# --------------------------------------------------------------------------- #
#  Инвариантность к SE(3) (трансляция + вращение)
# --------------------------------------------------------------------------- #
def test_volume_area_invariant_to_rigid_motion(dodeca):
    import se3
    R = se3.exp_so3([0.3, -0.7, 0.2])
    moved = dodeca.coords @ R.T + np.array([5.0, -3.0, 2.0])
    assert np.isclose(vol.enclosed_volume(moved, dodeca.faces),
                      vol.enclosed_volume(dodeca.coords, dodeca.faces), rtol=1e-9)
    assert np.isclose(vol.surface_area(moved, dodeca.faces),
                      vol.surface_area(dodeca.coords, dodeca.faces), rtol=1e-9)


# --------------------------------------------------------------------------- #
#  Градиенты: ∇V направлен наружу (раздувание увеличивает объём)
# --------------------------------------------------------------------------- #
def test_volume_gradient_points_outward(dodeca):
    g = vol.volume_gradient(dodeca.coords, dodeca.faces).reshape(-1, 3)
    # ∇V в каждой вершине сонаправлен с радиальным ортом (объём растёт при раздувании)
    rad = dodeca.coords / np.linalg.norm(dodeca.coords, axis=1, keepdims=True)
    dots = np.sum(g * rad, axis=1)
    assert np.all(dots > 0)


def test_area_gradient_matches_numeric(dodeca):
    """Аналитической формулы нет — проверяем согласованность: ∇A·δ ≈ ΔA."""
    g = vol.area_gradient(dodeca.coords, dodeca.faces)
    rng = np.random.default_rng(0)
    d = rng.normal(size=dodeca.coords.shape) * 1e-4
    dA = (vol.surface_area(dodeca.coords + d, dodeca.faces)
          - vol.surface_area(dodeca.coords - d, dodeca.faces)) / 2
    assert np.isclose(g @ d.ravel(), dA, rtol=1e-4, atol=1e-9)
