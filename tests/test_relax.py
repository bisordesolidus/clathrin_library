r"""
Ворота вехи M4 — фрустрация и релаксация.

Релаксация уменьшает энергию и сходится; релаксированная симметричная клетка —
истинный минимум (гессиан ⪰0, 6 нулевых мод); фрустрация зависит от клетки;
энантиомеры дают одинаковую фрустрацию.
"""
import numpy as np
import pytest

import diagonalize as dg
import embed as em
import lattice as lat
import relax as rx
from triskelion import Triskelion


@pytest.fixture(scope="module")
def dodeca_relaxed():
    m = rx.build_frustration_model(lat.dodecahedron())
    return m, rx.relax(m)


# --------------------------------------------------------------------------- #
#  Релаксация работает
# --------------------------------------------------------------------------- #
def test_relaxation_decreases_energy(dodeca_relaxed):
    _, res = dodeca_relaxed
    assert res.E_min < res.E0
    assert res.converged


def test_frustration_nonzero(dodeca_relaxed):
    """Собственная геометрия покоя ⇒ клетка фрустрирована (E_min > 0)."""
    _, res = dodeca_relaxed
    assert res.E_min > 1e-6


# --------------------------------------------------------------------------- #
#  Релаксированная клетка — истинный устойчивый минимум
# --------------------------------------------------------------------------- #
def test_relaxed_hessian_psd_six_zero_modes(dodeca_relaxed):
    m, res = dodeca_relaxed
    H = rx.hessian(m, res.g)                      # истинный гессиан (не GN)
    w, _ = dg.spectrum(H)
    assert dg.is_psd(w)
    assert dg.n_zero_modes(w) == 6


# --------------------------------------------------------------------------- #
#  Фрустрация зависит от клетки (основа T10)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_frustration_is_cage_dependent():
    cages = [lat.dodecahedron(),
             em.find_cage(28, {1, 2, 3}),          # мини-кот T_d
             em.fullerene_lattice(em.BARREL_36_15)]  # барабан D_6h
    fpv = [rx.frustration_per_vertex(rx.build_frustration_model(c)) for c in cages]
    # все различны (клетки различимы по релаксированной фрустрации)
    assert len(set(np.round(fpv, 3))) == len(fpv)
    assert all(f > 0 for f in fpv)


# --------------------------------------------------------------------------- #
#  Энантиомеры — одинаковая фрустрация (зеркальная симметрия)
# --------------------------------------------------------------------------- #
def test_enantiomers_same_frustration():
    mL = rx.build_frustration_model(lat.dodecahedron(),
                                    tris=Triskelion(chi=np.deg2rad(30)), hand=+1)
    mR = rx.build_frustration_model(lat.dodecahedron(),
                                    tris=Triskelion(chi=np.deg2rad(-30)), hand=-1)
    assert np.isclose(rx.relax(mL).E_min, rx.relax(mR).E_min, rtol=1e-6)


# --------------------------------------------------------------------------- #
#  Гессиан GN симметричен и разрежён
# --------------------------------------------------------------------------- #
def test_hessian_symmetric(dodeca_relaxed):
    m, res = dodeca_relaxed
    H = rx.hessian(m, res.g)
    assert np.allclose(H, H.T, atol=1e-10)
    assert H.shape == (6 * 20, 6 * 20)
