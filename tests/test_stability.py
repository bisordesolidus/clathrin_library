r"""
Ворота §9 — критическое натяжение (linear buckling).
"""
import numpy as np
import pytest

import lattice as lat
import stability as st


@pytest.fixture(scope="module")
def crit():
    d = lat.dodecahedron()
    sigma_c, v_c = st.critical_tension(d)
    return d, sigma_c, v_c


def test_critical_tension_finite_positive(crit):
    _, sigma_c, _ = crit
    assert np.isfinite(sigma_c)
    assert sigma_c > 0


def test_buckling_mode_unit_vector(crit):
    d, _, v_c = crit
    assert v_c is not None
    assert v_c.shape == (3 * d.n_vertices,)
    assert np.isclose(np.linalg.norm(v_c), 1.0)


def test_effective_hessian_singular_at_sigma_c(crit):
    """При σ=σ_c у H_eff появляется 7-я нулевая мода (сверх 6 rigid-body)."""
    d, sigma_c, _ = crit
    H = st.eff_hessian_at(d, sigma_c)
    w = np.sort(np.abs(np.linalg.eigvalsh(H)))
    scale = np.abs(np.linalg.eigvalsh(H)).max()
    # 6 rigid-body ~0, плюс критическая мода ~0 → 7-е с.з. мало
    assert w[6] < 1e-5 * scale


def test_stable_below_critical(crit):
    """Ниже σ_c оболочка устойчива (H_eff ≻ 0 на внутренних модах)."""
    d, sigma_c, _ = crit
    H = st.eff_hessian_at(d, 0.5 * sigma_c)
    w = np.sort(np.linalg.eigvalsh(H))
    assert w[6] > 1e-6 * abs(w).max()          # 7-я мода положительна
