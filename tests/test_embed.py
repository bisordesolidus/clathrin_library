r"""
Ворота вехи M3b — вложение фуллеренов и валидация точечных групп.

Главное: додекаэдр из спирали ≅ координатному M0; названные клетки из статьи
дают свои точечные группы как вырождения ANM-спектра (I_h→{1,3,4,5},
D_6h барабан→{1,2}, T_d мини-кот→{1,2,3}); топология любого изомера корректна.
"""
import networkx as nx
import pytest

import embed as em
import fullerene as fu
import lattice as lat


# --------------------------------------------------------------------------- #
#  Перекрёстная сверка с M0
# --------------------------------------------------------------------------- #
def test_dodecahedron_from_spiral_matches_m0():
    d = em.fullerene_lattice(em.DODECAHEDRON, name="dodeca-spiral")
    assert d.n_vertices == 20 and d.n_edges == 30 and d.n_faces == 12
    assert nx.is_isomorphic(d.graph, lat.dodecahedron().graph)


def test_dodecahedron_from_spiral_validates():
    d = em.fullerene_lattice(em.DODECAHEDRON)
    r = d.validate()
    assert r["euler"] == 2
    assert r["three_regular"] is True
    assert r["face_types"] == {5: 12}
    assert r["every_edge_in_two_faces"] is True


# --------------------------------------------------------------------------- #
#  Точечные группы = вырождения ANM-спектра (мишень из статьи)
# --------------------------------------------------------------------------- #
def test_dodecahedron_Ih():
    nz, sig = em.anm_signature(em.fullerene_lattice(em.DODECAHEDRON))
    assert nz == 6
    assert set(sig) == {1, 3, 4, 5}                 # I_h


def test_barrel_36_15_D6h():
    barrel = em.fullerene_lattice(em.BARREL_36_15, name="barrel-36-15")
    assert barrel.n_vertices == 36
    assert barrel.face_type_counts() == {5: 12, 6: 8}
    assert barrel.euler() == 2
    nz, sig = em.anm_signature(barrel)
    assert nz == 6
    assert set(sig) == {1, 2}                        # D_6h


def test_minicoat_28_Td_exists():
    """Среди 2 изомеров C28 один — мини-кот 28-2 с группой T_d ({1,2,3})."""
    sigs = [set(em.anm_signature(em.isomer_lattice(28, k))[1])
            for k in range(fu.n_isomers(28))]
    assert {1, 2, 3} in sigs                          # T_d мини-кот


def test_find_cage_by_signature():
    """find_cage находит клетку по сигнатуре группы."""
    minicoat = em.find_cage(28, {1, 2, 3})
    assert minicoat.n_vertices == 28
    assert em.anm_signature(minicoat)[1] == [1, 2, 3]


# --------------------------------------------------------------------------- #
#  Топология любого изомера корректна (12 пятиугольников, Эйлер)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [24, 28, 30, 32])
def test_all_isomers_valid_topology(n):
    for k in range(fu.n_isomers(n)):
        latt = em.isomer_lattice(n, k)
        assert latt.n_vertices == n
        assert latt.face_type_counts().get(5, 0) == 12       # 12 пятиугольников
        assert latt.face_type_counts().get(6, 0) == (n - 20) // 2
        assert latt.euler() == 2
        assert latt.is_three_regular()


def test_isomers_have_six_zero_modes():
    """Точечный ANM на любой клетке даёт ровно 6 нулевых мод (жёсткая сеть)."""
    for n in [24, 28, 32]:
        for k in range(fu.n_isomers(n)):
            nz, _ = em.anm_signature(em.isomer_lattice(n, k))
            assert nz == 6


# --------------------------------------------------------------------------- #
#  Масштаб и центрирование
# --------------------------------------------------------------------------- #
def test_edge_length_scaled():
    import numpy as np
    d = em.fullerene_lattice(em.DODECAHEDRON, edge=18.5)
    assert np.isclose(d.edge_lengths().mean(), 18.5, rtol=1e-6)
    barrel = em.fullerene_lattice(em.BARREL_36_15, edge=18.5)
    assert np.isclose(barrel.edge_lengths().mean(), 18.5, rtol=1e-6)
