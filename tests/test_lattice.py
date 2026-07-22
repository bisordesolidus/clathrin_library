r"""
Ворота геометрии решётки (до вехи M1) для lattice.py.

Додекаэдр: 20 вершин / 30 рёбер / 12 пятиугольников, Эйлер = 2, 3-регулярен,
рёбра равной длины, вершины на сфере, грани планарны и ориентированы наружу,
каждое ребро в двух гранях. Частичная проверка Iₕ — инвариантность множества
координат к перестановке осей и отражениям (полный спектр Iₕ — веха M1).
"""
import numpy as np
import pytest

import lattice as lat
from triskelion import EDGE_NM


@pytest.fixture(scope="module")
def dodeca():
    return lat.dodecahedron()


# --------------------------------------------------------------------------- #
#  Счётные инварианты
# --------------------------------------------------------------------------- #
def test_counts(dodeca):
    assert dodeca.n_vertices == 20
    assert dodeca.n_edges == 30
    assert dodeca.n_faces == 12


def test_euler(dodeca):
    assert dodeca.euler() == 2


def test_three_regular(dodeca):
    assert dodeca.is_three_regular()
    assert np.all(dodeca.degrees() == 3)


def test_all_faces_pentagons(dodeca):
    assert dodeca.face_type_counts() == {5: 12}


def test_planar(dodeca):
    import networkx as nx
    ok, _ = nx.check_planarity(dodeca.graph)
    assert ok


def test_validate_report(dodeca):
    r = dodeca.validate()
    assert r["V"] == 20 and r["E"] == 30 and r["F"] == 12
    assert r["euler"] == 2
    assert r["three_regular"] is True
    assert r["planar"] is True
    assert r["every_edge_in_two_faces"] is True
    assert r["edge_len_uniform"] is True


# --------------------------------------------------------------------------- #
#  Геометрия
# --------------------------------------------------------------------------- #
def test_edge_length_calibrated(dodeca):
    L = dodeca.edge_lengths()
    assert np.allclose(L, EDGE_NM, atol=1e-9)


def test_custom_edge_length():
    d = lat.dodecahedron(edge=10.0)
    assert np.allclose(d.edge_lengths(), 10.0, atol=1e-9)


def test_vertices_on_sphere(dodeca):
    R = dodeca.circumradii()
    assert np.ptp(R) < 1e-9 * R.mean()


def test_centered_at_origin(dodeca):
    assert np.allclose(dodeca.coords.mean(axis=0), 0.0, atol=1e-9)


def test_radial_directions_unit_and_outward(dodeca):
    d = dodeca.radial_directions()
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    # радиальный орт совпадает по направлению с самой вершиной (центр в 0)
    for i in range(dodeca.n_vertices):
        assert d[i] @ dodeca.coords[i] > 0


# --------------------------------------------------------------------------- #
#  Грани: планарность, ориентация наружу, каждое ребро в двух гранях
# --------------------------------------------------------------------------- #
def test_faces_are_planar(dodeca):
    for f in dodeca.faces:
        pts = dodeca.coords[f]
        c = pts.mean(axis=0)
        # SVD: наименьшее сингулярное значение ≈ 0 ⇒ точки компланарны
        s = np.linalg.svd(pts - c, compute_uv=False)
        assert s[-1] < 1e-9 * s[0]


def test_faces_oriented_outward(dodeca):
    r"""Нормаль грани (метод Ньюэлла) смотрит наружу (·центроид > 0)."""
    for f in dodeca.faces:
        pts = dodeca.coords[f]
        n = np.zeros(3)
        for k in range(len(f)):
            p, q = pts[k], pts[(k + 1) % len(f)]
            n += np.cross(p, q)
        assert n @ pts.mean(axis=0) > 0


def test_every_edge_in_two_faces(dodeca):
    inc = dodeca.edge_face_incidence()
    assert len(inc) == 30
    assert all(v == 2 for v in inc.values())


def test_faces_edges_consistent_with_graph(dodeca):
    r"""Рёбра, выведенные из граней, совпадают с рёбрами графа."""
    from_faces = set()
    for f in dodeca.faces:
        for k in range(len(f)):
            i, j = f[k], f[(k + 1) % len(f)]
            from_faces.add((min(i, j), max(i, j)))
    assert from_faces == set(dodeca.edges)


# --------------------------------------------------------------------------- #
#  Частичная проверка симметрии Iₕ (полный спектр — веха M1)
# --------------------------------------------------------------------------- #
def _coord_set(coords, tol=6):
    return {tuple(np.round(p, tol)) for p in coords}


def test_symmetry_axis_cycle(dodeca):
    r"""Множество вершин инвариантно к циклу осей (x,y,z)→(y,z,x) — C₃ из Iₕ."""
    S = _coord_set(dodeca.coords)
    cycled = _coord_set(dodeca.coords[:, [1, 2, 0]])
    assert S == cycled


def test_symmetry_inversion_and_reflections(dodeca):
    r"""Инвариантность к инверсии и к отражениям в координатных плоскостях."""
    S = _coord_set(dodeca.coords)
    assert S == _coord_set(-dodeca.coords)                      # инверсия
    for ax in range(3):
        flipped = dodeca.coords.copy()
        flipped[:, ax] *= -1
        assert S == _coord_set(flipped)                         # отражение


def test_enantiomer_note_chi_not_here(dodeca):
    r"""Геометрия решётки ахиральна сама по себе; хиральность войдёт с
    трискелионами (χ) в M2 — здесь координаты симметричны к отражению."""
    assert _coord_set(dodeca.coords) == _coord_set(dodeca.coords * [-1, 1, 1])
