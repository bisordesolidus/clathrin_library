r"""
Ворота вехи M2b — SE(3) rigid-body гессиан на додекаэдре («скелет»).

T1  ровно 6 нулевых мод;
T2  ядро = телесные генераторы se(3): ξ_i = Ad(g_i^{-1})·η;
T4  энантиомеры (χ→−χ, hand→−hand) изоспектральны;
T5  вырождения Iₕ ∈ {1,3,4,5}, сумма 114;
    H0 ⪰ 0; хиральность χ входит в спектр.
"""
import numpy as np
import pytest

import diagonalize as dg
import rigidbody as rb
from triskelion import Triskelion


@pytest.fixture(scope="module")
def model():
    return rb.build_se3_model()


@pytest.fixture(scope="module")
def spec(model):
    return dg.spectrum(model.H)


# --------------------------------------------------------------------------- #
#  T1 — ровно 6 нулевых мод, H0 ⪰ 0
# --------------------------------------------------------------------------- #
def test_exactly_six_zero_modes(spec):
    w, _ = spec
    assert dg.n_zero_modes(w) == 6


def test_psd(spec):
    w, _ = spec
    assert dg.is_psd(w)
    assert w.min() > -1e-9 * abs(w).max()


def test_hessian_dimension(model):
    assert model.H.shape == (6 * 20, 6 * 20)


# --------------------------------------------------------------------------- #
#  T2 — ядро = телесные генераторы se(3): Ad(g^{-1})·η
# --------------------------------------------------------------------------- #
def test_se3_generators_are_zero_modes(model):
    Z = rb.se3_zero_modes(model)
    assert Z.shape == (6 * 20, 6)
    assert np.abs(model.H @ Z).max() < 1e-9


def test_kernel_equals_se3_generators(model, spec):
    w, V = spec
    ker = V[:, dg.zero_mode_indices(w)]
    Z = rb.se3_zero_modes(model)
    assert ker.shape[1] == 6
    assert dg.subspace_angle_deg(ker, Z) < 1e-4


def test_se3_generators_are_independent(model):
    """6 мод Ad(g^{-1})·η линейно независимы (ранг 6)."""
    Z = rb.se3_zero_modes(model)
    assert np.linalg.matrix_rank(Z, tol=1e-8) == 6


# --------------------------------------------------------------------------- #
#  T5 — вырождения Iₕ
# --------------------------------------------------------------------------- #
def test_Ih_degeneracies(spec):
    w, _ = spec
    mults = dg.multiplicities(w)                 # дефолт deg_rtol=1e-6
    assert set(mults) <= {1, 3, 4, 5}
    assert sum(mults) == 6 * 20 - 6              # 114 внутренних мод


def test_lowest_nonzero_mode_degenerate(spec):
    w, _ = spec
    _, mult = dg.degeneracy_groups(w)[0]
    assert mult in {3, 4, 5}


# --------------------------------------------------------------------------- #
#  Хиральность: χ входит в спектр
# --------------------------------------------------------------------------- #
def test_chirality_enters_spectrum():
    w_chi, _ = dg.spectrum(rb.build_se3_model(tris=Triskelion(chi=np.deg2rad(30))).H)
    w_0, _ = dg.spectrum(rb.build_se3_model(tris=Triskelion(chi=0.0)).H)
    assert not np.allclose(np.sort(w_chi)[6:], np.sort(w_0)[6:], atol=1e-6)


# --------------------------------------------------------------------------- #
#  T4 — энантиомеры изоспектральны
# --------------------------------------------------------------------------- #
def test_enantiomers_isospectral():
    """Зеркальный энантиомер (χ→−χ, hand→−hand) даёт тот же спектр."""
    m_L = rb.build_se3_model(tris=Triskelion(chi=np.deg2rad(30)), hand=+1)
    m_R = rb.build_se3_model(tris=Triskelion(chi=np.deg2rad(-30)), hand=-1)
    wL, _ = dg.spectrum(m_L.H)
    wR, _ = dg.spectrum(m_R.H)
    assert np.allclose(np.sort(wL), np.sort(wR), atol=1e-8)


def test_both_chiralities_have_six_zero_modes():
    for hand in (+1, -1):
        w, _ = dg.spectrum(rb.build_se3_model(hand=hand).H)
        assert dg.n_zero_modes(w) == 6


# --------------------------------------------------------------------------- #
#  Жёсткость масштабирует спектр
# --------------------------------------------------------------------------- #
def test_stiffness_scales_spectrum():
    w1, _ = dg.spectrum(rb.build_se3_model(stiffness=1.0).H)
    w2, _ = dg.spectrum(rb.build_se3_model(stiffness=2.5).H)
    assert np.allclose(np.sort(w2)[6:], 2.5 * np.sort(w1)[6:], rtol=1e-9)
