r"""
bandstructure.py — восприимчивости объёма к натяжению и зонная структура (§15).

Линейный отклик объёма на натяжение (доминируется мягкой дыхательной модой):
        V(σ) = V₀ − σ·χ_VA,     χ_VA = (∇V)ᵀ H₀⁺ (∇A).
Тепловая флуктуация объёма (полуширина зоны):
        ⟨δV²⟩ = k_BT (∇V)ᵀ H₀⁺ (∇V).
Обе — через псевдообратную H₀⁺ точечного ANM (мягкие моды дают главный вклад).

Зоны доступных объёмов (§15.5): каждый разрешённый изомер при σ∈[σ_min,σ_max]
заметает отрезок V_iso(σ); объединение — «зоны», дополнение — «запрещённые зоны».
Здесь — операторный слой (восприимчивости и отклик); построение самих зон по
набору клеток — поверх (использует `schein` как фильтр допустимых).
"""

from __future__ import annotations

import numpy as np

import hamiltonian as ham
import volume as vol
from perturbation import _pinv_drop_null


def anm_setup(latt, cutoff_factor: float = 2.0):
    r"""Пары и длины покоя точечного ANM (длина покоя = фактическое расстояние)."""
    a = float(latt.edge_lengths().mean())
    pairs = ham.anm_pairs(latt.coords, cutoff_factor * a)
    C = latt.coords
    rest = np.array([np.linalg.norm(C[int(i)] - C[int(j)]) for i, j in pairs])
    return pairs, rest


def _H0_pinv(latt, stiffness: float):
    pairs, rest = anm_setup(latt)
    H0 = ham.anm_full_hessian(latt.coords, pairs, rest, stiffness)
    return _pinv_drop_null(H0)


def volume_susceptibility(latt, stiffness: float = 1.0) -> float:
    r"""χ_VA = (∇V)ᵀ H₀⁺ (∇A): восприимчивость объёма к натяжению (V=V₀−σχ_VA)."""
    H0p = _H0_pinv(latt, stiffness)
    gV = vol.volume_gradient(latt.coords, latt.faces)
    gA = vol.area_gradient(latt.coords, latt.faces)
    return float(gV @ H0p @ gA)


def thermal_volume_variance(latt, kT: float = 1.0, stiffness: float = 1.0) -> float:
    r"""⟨δV²⟩ = k_BT (∇V)ᵀ H₀⁺ (∇V): тепловая дисперсия объёма (полуширина зоны)."""
    H0p = _H0_pinv(latt, stiffness)
    gV = vol.volume_gradient(latt.coords, latt.faces)
    return float(kT * gV @ H0p @ gV)


def volume_at_tension(latt, sigma: float, stiffness: float = 1.0) -> float:
    r"""Объём под натяжением в линейном отклике: V(σ) = V₀ − σ·χ_VA."""
    V0 = vol.enclosed_volume(latt.coords, latt.faces)
    return V0 - sigma * volume_susceptibility(latt, stiffness)


def volume_band(latt, sigma_max: float, kT: float = 1.0,
                stiffness: float = 1.0) -> dict:
    r"""Зона доступных объёмов одной клетки (§15.5).

    При σ∈[0,σ_max] объём заметает отрезок [V₀−σ_max·χ_VA, V₀], уширенный
    тепловым ±√⟨δV²⟩. Возвращает {V0, chi, dV, lo, hi, N}."""
    from volume import enclosed_volume
    V0 = enclosed_volume(latt.coords, latt.faces)
    chi = volume_susceptibility(latt, stiffness)
    dV = float(np.sqrt(abs(thermal_volume_variance(latt, kT, stiffness))))
    return {"name": latt.name, "N": latt.n_vertices, "V0": V0, "chi": chi,
            "dV": dV, "lo": V0 - sigma_max * chi - dV, "hi": V0 + dV}


def band_structure(lattices, sigma_max: float, kT: float = 1.0,
                   stiffness: float = 1.0) -> dict:
    r"""Зонная структура объёмов по набору клеток (§15.5–15.6).

    Возвращает {bands: [...] по возрастанию V₀, forbidden: [(lo,hi,ниже,выше)]}.
    Запрещённая зона — промежуток между верхом одной зоны и низом следующей;
    отрицательная ширина = зоны перекрываются (кроссовер §15.6)."""
    bands = sorted((volume_band(L, sigma_max, kT, stiffness) for L in lattices),
                   key=lambda b: b["V0"])
    forbidden = []
    for a, b in zip(bands, bands[1:]):
        gap = b["lo"] - a["hi"]
        forbidden.append({"lo": a["hi"], "hi": b["lo"], "width": gap,
                          "below": a["name"], "above": b["name"],
                          "overlap": gap < 0})
    return {"bands": bands, "forbidden": forbidden}


def relax_under_tension(latt, sigma: float, stiffness: float = 1.0,
                        iters: int = 400, tol: float = 1e-11) -> np.ndarray:
    r"""Точное новое равновесие под натяжением: min ½Σk(|r_ij|−d₀)² + σ·A(r).
    Квази-ньютон с фиксированной H₀⁺ (сходится для малых σ) — эталон для χ_VA."""
    pairs, rest = anm_setup(latt)
    H0p = _pinv_drop_null(ham.anm_full_hessian(latt.coords, pairs, rest, stiffness))
    C = latt.coords.copy()
    for _ in range(iters):
        g = np.zeros_like(C)
        for idx, (i, j) in enumerate(pairs):
            i, j = int(i), int(j)
            r = C[i] - C[j]; L = np.linalg.norm(r)
            f = stiffness * (L - rest[idx]) * r / L
            g[i] += f; g[j] -= f
        g = g + sigma * vol.area_gradient(C, latt.faces).reshape(C.shape)
        C = C - (H0p @ g.ravel()).reshape(C.shape)
        if np.linalg.norm(g) < tol:
            break
    return C
