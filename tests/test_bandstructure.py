r"""
Ворота §15 (восприимчивости объёма): T12 χ_VA vs конечная разность,
T13 ⟨δV²⟩ vs Монте-Карло.
"""
import numpy as np
import pytest

import bandstructure as bs
import hamiltonian as ham
import lattice as lat
import volume as vol


@pytest.fixture(scope="module")
def dodeca():
    return lat.dodecahedron()


# --------------------------------------------------------------------------- #
#  T12 — восприимчивость объёма vs точный отклик (релаксация под натяжением)
# --------------------------------------------------------------------------- #
def test_susceptibility_vs_finite_diff(dodeca):
    chi = bs.volume_susceptibility(dodeca)
    V0 = vol.enclosed_volume(dodeca.coords, dodeca.faces)
    sigma = 1e-3
    C = bs.relax_under_tension(dodeca, sigma)
    Vs = vol.enclosed_volume(C, dodeca.faces)
    assert np.isclose((V0 - Vs) / sigma, chi, rtol=1e-2)     # V(σ)=V₀−σχ_VA


def test_susceptibility_positive(dodeca):
    """Натяжение СЖИМАЕТ клетку ⇒ χ_VA > 0 (объём убывает с σ)."""
    assert bs.volume_susceptibility(dodeca) > 0


def test_volume_at_tension_linear(dodeca):
    chi = bs.volume_susceptibility(dodeca)
    V0 = vol.enclosed_volume(dodeca.coords, dodeca.faces)
    assert np.isclose(bs.volume_at_tension(dodeca, 0.01), V0 - 0.01 * chi)


# --------------------------------------------------------------------------- #
#  T13 — тепловая дисперсия объёма vs Монте-Карло
# --------------------------------------------------------------------------- #
def test_thermal_variance_vs_monte_carlo(dodeca):
    kT = 1.0
    var = bs.thermal_volume_variance(dodeca, kT=kT)
    pairs, rest = bs.anm_setup(dodeca)
    H0 = ham.anm_full_hessian(dodeca.coords, pairs, rest)
    w, V = np.linalg.eigh(H0)
    active = w > 1e-6 * w.max()
    gV = vol.volume_gradient(dodeca.coords, dodeca.faces)
    # выборка q ~ N(0, kT H₀⁺): q = √kT Σ z_k v_k/√λ_k
    rng = np.random.default_rng(0)
    Va, wa = V[:, active], w[active]
    z = rng.normal(size=(active.sum(), 30000))
    q = Va @ (z / np.sqrt(wa)[:, None]) * np.sqrt(kT)
    proj = gV @ q
    assert np.isclose(np.var(proj), var, rtol=0.05)


def test_variance_scales_with_kT(dodeca):
    v1 = bs.thermal_volume_variance(dodeca, kT=1.0)
    v3 = bs.thermal_volume_variance(dodeca, kT=3.0)
    assert np.isclose(v3, 3.0 * v1, rtol=1e-9)


# --------------------------------------------------------------------------- #
#  T14 — зонная структура: дискретные зоны, запрещённые промежутки, кроссовер
# --------------------------------------------------------------------------- #
def test_volume_band_single(dodeca):
    b = bs.volume_band(dodeca, sigma_max=0.3)
    assert b["V0"] > 0 and b["chi"] > 0 and b["dV"] > 0
    assert b["lo"] < b["V0"] < b["hi"] + b["chi"] * 0.3   # зона ниже V₀ по σ


def test_band_structure_discrete_and_gaps():
    """Малые клетки дают дискретные зоны с запрещёнными промежутками (§15.6)."""
    import embed as em
    cages = [em.fullerene_lattice([5] * 12, name="C20"),
             em.isomer_lattice(24, 0), em.isomer_lattice(28, 1)]
    bstr = bs.band_structure(cages, sigma_max=0.1)
    assert len(bstr["bands"]) == 3
    # объёмы строго возрастают с N (дискретная лесенка V∝N^{3/2})
    V0s = [b["V0"] for b in bstr["bands"]]
    assert V0s[0] < V0s[1] < V0s[2]
    # между малыми клетками — запрещённые промежутки (не перекрываются)
    assert all(not f["overlap"] for f in bstr["forbidden"])
    assert all(f["width"] > 0 for f in bstr["forbidden"])

