r"""Юнит-тесты сборщика гессиана и точечного ANM."""
import numpy as np
import pytest

import hamiltonian as ham
import lattice as lat


# --------------------------------------------------------------------------- #
#  Сборщик H = Σ k J Jᵀ
# --------------------------------------------------------------------------- #
def test_assemble_single_rank1():
    """Один вклад: H = k·J Jᵀ точно."""
    J = np.array([1.0, -2.0, 0.5, 3.0])
    idx = [0, 1, 2, 3]
    H = ham.assemble(4, [(idx, J, 2.0)]).toarray()
    assert np.allclose(H, 2.0 * np.outer(J, J))


def test_assemble_accumulates_and_scatters():
    """Два вклада на разных DOF складываются в нужные блоки."""
    H = ham.assemble(4, [([0, 1], np.array([1.0, 1.0]), 1.0),
                         ([2, 3], np.array([1.0, -1.0]), 3.0)]).toarray()
    expect = np.zeros((4, 4))
    expect[np.ix_([0, 1], [0, 1])] += np.outer([1, 1], [1, 1])
    expect[np.ix_([2, 3], [2, 3])] += 3.0 * np.outer([1, -1], [1, -1])
    assert np.allclose(H, expect)


def test_assemble_symmetric_psd():
    H = ham.assemble(6, [([0, 1, 2, 3, 4, 5],
                          np.random.default_rng(0).normal(size=6), 1.7)]).toarray()
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() > -1e-12


# --------------------------------------------------------------------------- #
#  Точечный ANM
# --------------------------------------------------------------------------- #
def test_anm_pairs_shells():
    """Отсечка по оболочкам додекаэдра даёт нужное число пар."""
    d = lat.dodecahedron()
    a = d.edge_lengths().mean()
    assert len(ham.anm_pairs(d.coords, 1.2 * a)) == 30          # только рёбра
    assert len(ham.anm_pairs(d.coords, 2.0 * a)) == 30 + 60     # + диагонали граней


def test_point_node_hessian_symmetric_psd():
    d = lat.dodecahedron()
    H = ham.anm_from_lattice(d).toarray()
    assert np.allclose(H, H.T, atol=1e-12)
    assert np.linalg.eigvalsh(H).min() > -1e-9 * abs(np.linalg.eigvalsh(H)).max()


def test_anm_translation_invariance():
    """Сдвиг всех узлов на общий вектор — нулевая энергия: H·T = 0."""
    d = lat.dodecahedron()
    H = ham.anm_from_lattice(d).toarray()
    N = d.n_vertices
    for a in range(3):
        T = np.zeros(3 * N)
        T[a::3] = 1.0
        assert np.allclose(H @ T, 0.0, atol=1e-10)


def test_anm_rotation_invariance():
    """Бесконечно-малое вращение — нулевая энергия: H·(ê×r) = 0."""
    d = lat.dodecahedron()
    H = ham.anm_from_lattice(d).toarray()
    c = d.coords - d.coords.mean(0)
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        R = np.cross(e, c).ravel()
        assert np.allclose(H @ R, 0.0, atol=1e-10)


def test_edge_only_is_underconstrained():
    """Максвелл: только рёбра ⇒ много нулевых мод (недоопределённость)."""
    d = lat.dodecahedron()
    a = d.edge_lengths().mean()
    H = ham.point_node_hessian(d.coords, 1.2 * a).toarray()
    w = np.linalg.eigvalsh(H)
    n_zero = int(np.sum(np.abs(w) < 1e-8 * abs(w).max()))
    assert n_zero == 30      # 24 механизма Максвелла + 6 rigid-body
