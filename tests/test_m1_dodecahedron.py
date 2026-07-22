r"""
Ворота вехи M1 — точечный ANM baseline на додекаэдре.

Главное: ровно 6 нулевых мод = rigid-body ядро; H ⪰ 0; вырождения ненулевого
спектра идут группами кратности ∈ {1,3,4,5} — размерности неприводимых
представлений группы Iₕ (первая настоящая проверка симметрии клетки).
"""
import numpy as np
import pytest

import diagonalize as dg
import hamiltonian as ham
import lattice as lat


@pytest.fixture(scope="module")
def spec():
    d = lat.dodecahedron()
    H = ham.anm_from_lattice(d)          # отсечка 2·a: рёбра + диагонали граней
    w, V = dg.spectrum(H)
    return d, H, w, V


# --------------------------------------------------------------------------- #
#  T1 — ровно 6 нулевых мод
# --------------------------------------------------------------------------- #
def test_exactly_six_zero_modes(spec):
    _, _, w, _ = spec
    assert dg.n_zero_modes(w) == 6


def test_psd(spec):
    _, _, w, _ = spec
    assert dg.is_psd(w)
    assert w.min() > -1e-9 * abs(w).max()


# --------------------------------------------------------------------------- #
#  T2 — ядро = rigid-body подпространство
# --------------------------------------------------------------------------- #
def test_kernel_is_rigid_body(spec):
    d, _, w, V = spec
    ker = V[:, dg.zero_mode_indices(w)]
    rb = dg.rigid_body_modes(d.coords)
    assert ker.shape[1] == 6 and rb.shape[1] == 6
    # численное ядро совпадает с аналитическим rigid-body пространством
    assert dg.subspace_angle_deg(ker, rb) < 1e-4


def test_rigid_body_modes_annihilated(spec):
    _, H, _, _ = spec
    d = lat.dodecahedron()
    rb = dg.rigid_body_modes(d.coords)
    assert np.allclose(H @ rb, 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
#  T5 — вырождения Iₕ ∈ {1,3,4,5}
# --------------------------------------------------------------------------- #
def test_Ih_degeneracy_multiplicities(spec):
    _, _, w, _ = spec
    mults = dg.multiplicities(w)
    # все кратности — размерности ирепов Iₕ
    assert set(mults) <= {1, 3, 4, 5}
    # внутренних мод ровно 60 − 6 = 54
    assert sum(mults) == 54


def test_lowest_nonzero_mode_is_degenerate(spec):
    """Низшая ненулевая мода вырождена (не одиночная) — коллективная деформация."""
    _, _, w, _ = spec
    groups = dg.degeneracy_groups(w)
    val, mult = groups[0]
    assert mult in {3, 4, 5}
    assert val > 0


def test_spectrum_structure_stable_to_cutoff(spec):
    """Спектр (6 нулевых + вырождения Iₕ) не зависит от точного R_c внутри окна."""
    d = lat.dodecahedron()
    for cf in [1.7, 2.0, 2.2]:
        w, _ = dg.spectrum(ham.anm_from_lattice(d, cutoff_factor=cf))
        assert dg.n_zero_modes(w) == 6
        assert set(dg.multiplicities(w)) <= {1, 3, 4, 5}


# --------------------------------------------------------------------------- #
#  Масштабирование жёсткости
# --------------------------------------------------------------------------- #
def test_stiffness_scales_spectrum(spec):
    """Спектр линеен по жёсткости: k·H ⇒ k·λ."""
    d = lat.dodecahedron()
    w1, _ = dg.spectrum(ham.anm_from_lattice(d, stiffness=1.0))
    w3, _ = dg.spectrum(ham.anm_from_lattice(d, stiffness=3.0))
    nz1 = np.sort(w1)[6:]
    nz3 = np.sort(w3)[6:]
    assert np.allclose(nz3, 3.0 * nz1, rtol=1e-9)
