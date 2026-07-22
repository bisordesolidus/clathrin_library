r"""
Ворота вехи M2a — вращательная система, маршрутизация ног (голова-хвост),
хаб-реперы. Главное: правило голова-хвост = трассировка граней; каждое ребро
несёт ровно 2 проксимальных + 2 дистальных сегмента; смешанные повороты
запрещены (сливают грани).
"""
import numpy as np
import pytest

import lattice as lat
import routing as rt
from triskelion import Triskelion


@pytest.fixture(scope="module")
def setup():
    d = lat.dodecahedron()
    rot = rt.rotation_system(d)
    return d, rot


# --------------------------------------------------------------------------- #
#  Вращательная система и перестановки карты
# --------------------------------------------------------------------------- #
def test_rotation_system_three_neighbors(setup):
    d, rot = setup
    for i in range(d.n_vertices):
        assert len(rot[i]) == 3
        assert set(rot[i]) == set(d.graph.neighbors(i))


def test_alpha_involution(setup):
    d, rot = setup
    for dart in rt.darts(d):
        assert rt.alpha(rt.alpha(dart)) == dart


def test_sigma_is_3cycle(setup):
    """σ на дротиках из одной вершины — цикл длины 3."""
    d, rot = setup
    for i in range(d.n_vertices):
        dart = (i, rot[i][0])
        assert rt.sigma(rt.sigma(rt.sigma(dart, rot), rot), rot) == dart


def test_dart_count(setup):
    d, rot = setup
    assert len(rt.darts(d)) == 2 * d.n_edges == 3 * d.n_vertices


# --------------------------------------------------------------------------- #
#  ГЛАВНОЕ: голова-хвост = трассировка граней
# --------------------------------------------------------------------------- #
def test_orbits_equal_faces_both_chiralities(setup):
    """φ = σ∘α обходит грани для обеих хиральностей (энантиомеры)."""
    d, rot = setup
    for hand in (+1, -1):
        orbits = rt.trace_orbits(d, rot, hand)
        assert len(orbits) == 12
        assert all(len(o) == 5 for o in orbits)
        assert rt.is_head_to_tail_consistent(d, rot, hand)


def test_orbit_vertex_cycles_are_faces(setup):
    """Циклы вершин орбит совпадают с гранями (как множества)."""
    d, rot = setup
    orbits = rt.trace_orbits(d, rot, +1)
    assert rt.orbit_vertex_sets(orbits) == {frozenset(f) for f in d.faces}


def test_mixed_handedness_forbidden(setup):
    """Смешанные повороты сливают грани ⇒ орбиты ≠ грани (запрет)."""
    d, rot = setup
    mixed = {i: (-1 if i == 0 else +1) for i in range(d.n_vertices)}
    assert not rt.is_head_to_tail_consistent(d, rot, mixed)
    orbits = rt.trace_orbits(d, rot, mixed)
    sizes = sorted(len(o) for o in orbits)
    assert sizes != [5] * 12          # структура сломана
    assert sum(sizes) == 60            # но дротиков по-прежнему 3N


# --------------------------------------------------------------------------- #
#  Состав ребра: 2 проксимальных + 2 дистальных, антипараллельны
# --------------------------------------------------------------------------- #
def test_every_edge_two_proximal_two_distal(setup):
    d, rot = setup
    seg = rt.edge_segments(d, rot, +1)
    assert len(seg) == 30
    for e, s in seg.items():
        assert len(s["proximal"]) == 2
        assert len(s["distal"]) == 2


def test_proximal_antiparallel(setup):
    """Два проксимальных дротика ребра {i,j} — это (i,j) и (j,i) (антипараллельны)."""
    d, rot = setup
    seg = rt.edge_segments(d, rot, +1)
    for e, s in seg.items():
        p = s["proximal"]
        assert {p[0], p[1]} == {p[0], rt.alpha(p[0])}


def test_distal_antiparallel(setup):
    """Два дистальных сегмента ребра идут в противоположных направлениях."""
    d, rot = setup
    routing = rt.leg_routing(d, rot, +1)
    seg = rt.edge_segments(d, rot, +1)
    for e, s in seg.items():
        dd = [routing[leg] for leg in s["distal"]]   # дистальные дротики
        assert {dd[0], dd[1]} == {dd[0], rt.alpha(dd[0])}


def test_leg_count_per_triskelion(setup):
    """У каждого трискелиона ровно 3 ноги = 3 дротика наружу."""
    d, rot = setup
    routing = rt.leg_routing(d, rot, +1)
    from collections import Counter
    c = Counter(dart[0] for dart in routing)
    assert all(c[i] == 3 for i in range(d.n_vertices))


# --------------------------------------------------------------------------- #
#  Хаб-реперы
# --------------------------------------------------------------------------- #
def test_hub_frames_are_SO3(setup):
    d, rot = setup
    R = rt.hub_frames(d, rot)
    for i in range(d.n_vertices):
        assert np.allclose(R[i] @ R[i].T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R[i]), 1.0, atol=1e-12)


def test_hub_frame_e3_is_radial(setup):
    d, rot = setup
    R = rt.hub_frames(d, rot)
    rh = d.radial_directions()
    for i in range(d.n_vertices):
        assert np.allclose(R[i][:, 2], rh[i], atol=1e-12)


def test_hub_frame_e1_toward_first_neighbor(setup):
    """ê₁ указывает (касательно) на первого соседа rot[i][0]."""
    d, rot = setup
    R = rt.hub_frames(d, rot)
    C = d.coords
    for i in range(d.n_vertices):
        t = C[rot[i][0]] - C[i]
        t = t - (t @ R[i][:, 2]) * R[i][:, 2]
        t /= np.linalg.norm(t)
        assert np.allclose(R[i][:, 0], t, atol=1e-12)


def test_legs_point_at_neighbors(setup):
    """Три ноги трискелиона (азимуты 0/120/240°) смотрят на трёх соседей."""
    d, rot = setup
    R = rt.hub_frames(d, rot)
    C = d.coords
    T = Triskelion()
    for i in range(d.n_vertices):
        e1, e2, e3 = R[i][:, 0], R[i][:, 1], R[i][:, 2]
        for a in range(3):
            # азимут мировой ноги a в касательном репере
            w = R[i] @ T.leg_axis(a)
            az_leg = np.arctan2(w @ e2, w @ e1)
            # азимут соседа a
            t = C[rot[i][a]] - C[i]
            az_nb = np.arctan2(t @ e2, t @ e1)
            assert np.isclose((az_leg - az_nb) % (2 * np.pi), 0.0, atol=1e-9) or \
                   np.isclose((az_leg - az_nb) % (2 * np.pi), 2 * np.pi, atol=1e-9)
