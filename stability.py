r"""
stability.py — критическое натяжение (linear buckling analysis, MODEL.md §9).

Эффективный гессиан под натяжением аффинен по σ:
        H_eff(σ) = H₀ + σ·H⁽¹⁾_σ      (H⁽¹⁾_σ — эффективный вклад натяжения, §8.2).
Оболочка устойчива ⟺ H_eff(σ) ≻ 0 на дополнении к нулевым модам. Критическое
натяжение — наименьшее σ, при котором появляется нулевое с.з.:
        H₀ v = μ (−H⁽¹⁾_σ) v,     σ_c = наименьшее положительное σ,
собственный вектор v_c = МОДА ПОТЕРИ УСТОЙЧИВОСТИ.

(Для замкнутых клеток σ_c — предел механической устойчивости под натяжением;
переход flat→curved требует плоской решётки — M3c.)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig

import hamiltonian as ham
from bandstructure import anm_setup
from perturbation import tension_h1


def critical_tension(latt, stiffness: float = 1.0):
    r"""Критическое натяжение σ_c и мода потери устойчивости.

    Возвращает (σ_c, v_c): σ_c — наименьшее σ>0, при котором мода H_eff(σ) → 0
    (np.inf, если оболочка устойчива при любом натяжении); v_c — мода в
    исходных 3N координатах."""
    pairs, rest = anm_setup(latt)
    H0 = ham.anm_full_hessian(latt.coords, pairs, rest, stiffness)
    Weff, _ = tension_h1(latt.coords, latt.faces, pairs, rest, stiffness)

    # проекция на ненулевое подпространство H₀
    w, V = np.linalg.eigh(0.5 * (H0 + H0.T))
    scale = float(w.max())
    P = V[:, w > 1e-6 * scale]
    H0r = P.T @ H0 @ P
    Wr = P.T @ Weff @ P

    # H_eff_r(σ) = H0r + σ Wr вырождается при det = 0 ⇒ обобщённые с.з. (H0r, −Wr)
    vals, vecs = eig(H0r, -Wr)
    sig = vals.real[np.abs(vals.imag) < 1e-8 * (np.abs(vals.real) + 1)]
    positive = sig[sig > 1e-12]
    if positive.size == 0:
        return np.inf, None
    k = int(np.argmin(np.where(sig > 1e-12, sig, np.inf)))
    sigma_c = float(sig[k])
    v_c = P @ vecs[:, k].real
    return sigma_c, v_c / np.linalg.norm(v_c)


def eff_hessian_at(latt, sigma: float, stiffness: float = 1.0) -> np.ndarray:
    r"""H_eff(σ) = H₀ + σ·H⁽¹⁾_σ (для проверки: при σ=σ_c имеет нулевое с.з.)."""
    pairs, rest = anm_setup(latt)
    H0 = ham.anm_full_hessian(latt.coords, pairs, rest, stiffness)
    Weff, _ = tension_h1(latt.coords, latt.faces, pairs, rest, stiffness)
    return H0 + sigma * Weff
