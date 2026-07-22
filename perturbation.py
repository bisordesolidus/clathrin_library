r"""
perturbation.py — стационарная теория возмущений (MODEL.md §8).

Классический аналог квантовомеханической ТВ, применённый к гармонической матрице
H₀ (гессиан). Возмущение H' сдвигает собственные значения (жёсткости мод):

  λ_n⁽¹⁾ = ⟨n|H'|n⟩
  λ_n⁽²⁾ = Σ_{m≠n} |⟨m|H'|n⟩|² / (λ_n⁽⁰⁾ − λ_m⁽⁰⁾)
  |n⁽¹⁾⟩ = Σ_{m≠n} ⟨m|H'|n⟩/(λ_n⁽⁰⁾ − λ_m⁽⁰⁾) |m⟩

ВЫРОЖДЕННАЯ ТВ (вырождения по симметрии в наших клетках повсеместны): в
вырожденном подпространстве сначала ДИАГОНАЛИЗУЕМ H' → правильный нулевой базис
и λ⁽¹⁾ = его собственные значения; затем обычная ТВ 2-го порядка.

Нулевые моды H₀ (rigid-body) исключаются из сумм (деление на 0) — передаются
через `active`. Валидация (T6/T7): ошибка ТВ¹ = O(ε²) (наклон 2), ТВ² = O(ε³)
(наклон 3) против точной диагонализации H₀+εH'.

  [Замечание §8.2: если возмущение имеет ЛИНЕЙНЫЙ член (натяжение: f=σ∇A≠0),
   эффективный H⁽¹⁾ = W − C[H₀⁺f,·,·] включает ангармонический тензор C —
   собирать H' надо в СДВИНУТОМ минимуме. Здесь машинерия ТВ для готового H';
   сборка физических H' (натяжение и т.д.) — `stability.py`/расширения.]
"""

from __future__ import annotations

import numpy as np


def _degenerate_groups(w: np.ndarray, active: np.ndarray,
                       rtol: float) -> list[list[int]]:
    r"""Сгруппировать активные моды по вырождению (близкие λ⁽⁰⁾)."""
    scale = max(float(np.abs(w).max()), 1.0)
    order = active[np.argsort(w[active])]
    groups: list[list[int]] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and (w[order[j + 1]] - w[order[i]]) < rtol * scale:
            j += 1
        groups.append([int(x) for x in order[i:j + 1]])
        i = j + 1
    return groups


def pt_corrections(w0: np.ndarray, V0: np.ndarray, Hp: np.ndarray,
                   active: np.ndarray | None = None,
                   deg_rtol: float = 1e-6):
    r"""Поправки ТВ 1-го и 2-го порядка (с вырожденной ТВ).

    w0, V0 — спектр H₀ (собственные значения, векторы столбцами).
    Hp — матрица возмущения H'. active — индексы мод, участвующих в ТВ
    (по умолчанию все; для клеток передавать НЕнулевые моды).
    Возвращает (λ⁽¹⁾, λ⁽²⁾, V_corr) — поправки (nan для неактивных мод) и
    исправленный (вырожденной ТВ) базис."""
    n = len(w0)
    if active is None:
        active = np.arange(n)
    active = np.asarray(active)
    Hpb = V0.T @ Hp @ V0
    lam1 = np.full(n, np.nan)
    lam2 = np.full(n, np.nan)
    Vcorr = V0.copy()

    groups = _degenerate_groups(w0, active, deg_rtol)
    # 1-й порядок + правильный базис в вырожденных подпространствах
    for g in groups:
        if len(g) == 1:
            lam1[g[0]] = Hpb[g[0], g[0]]
        else:
            ev, evec = np.linalg.eigh(Hpb[np.ix_(g, g)])
            for k, m in enumerate(g):
                lam1[m] = ev[k]
            Vcorr[:, g] = V0[:, g] @ evec      # поворот к правильному нулевому базису

    # 2-й порядок в исправленном базисе (суммы только по ДРУГИМ группам)
    Hpc = Vcorr.T @ Hp @ Vcorr
    for g in groups:
        for m in g:
            s = 0.0
            for h in groups:
                if h is g:
                    continue
                for k in h:
                    s += Hpc[k, m] ** 2 / (w0[m] - w0[k])
            lam2[m] = s
    return lam1, lam2, Vcorr


def pt_eigenvalues(w0: np.ndarray, V0: np.ndarray, Hp: np.ndarray, eps: float,
                   active: np.ndarray | None = None, order: int = 2,
                   deg_rtol: float = 1e-6) -> np.ndarray:
    r"""Собственные значения H₀+εH', оценённые ТВ (для активных мод)."""
    lam1, lam2, _ = pt_corrections(w0, V0, Hp, active, deg_rtol)
    out = w0.copy().astype(float)
    m = ~np.isnan(lam1)
    out[m] = w0[m] + eps * lam1[m]
    if order >= 2:
        out[m] = out[m] + eps ** 2 * lam2[m]
    return out


def exact_eigenvalues(H0: np.ndarray, Hp: np.ndarray, eps: float) -> np.ndarray:
    r"""Точные собственные значения H₀+εH' (эталон для валидации ТВ)."""
    A = H0 + eps * Hp
    return np.linalg.eigvalsh(0.5 * (A + A.T))


def _pinv_drop_null(H: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    r"""Псевдообратная H с отброшенными нулевыми модами (rigid-body)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    scale = max(float(np.abs(w).max()), 1.0)
    inv = np.where(np.abs(w) > rtol * scale, 1.0 / w, 0.0)
    return (V * inv) @ V.T


def tension_h1(coords: np.ndarray, faces, pairs: np.ndarray, rest: np.ndarray,
               stiffness: float = 1.0, h: float = 1e-4) -> np.ndarray:
    r"""Эффективный H⁽¹⁾ мембранного натяжения (§8.2), точечный ANM 3N.

    Возмущение — площадь A(r): H = H₀ + σ·A. Линейный член f=∇A ≠ 0 СДВИГАЕТ
    равновесие на δq* = −σ H₀⁺∇A, и это входит в спектр через ангармонику:
        H⁽¹⁾ = ∇²A − C[H₀⁺∇A],
    где C[v] = производная ANM-гессиана вдоль v. НАИВНЫЙ H⁽¹⁾=∇²A (забыт сдвиг)
    даёт ТВ наклон 1 вместо 2 — проверено.
    Возвращает (H1_correct, H1_naive)."""
    import hamiltonian as ham
    import volume as vol
    coords = np.asarray(coords, dtype=float)
    H0 = ham.anm_full_hessian(coords, pairs, rest, stiffness)   # = проектир. (равновесие)
    gradA = vol.area_gradient(coords, faces)
    HA = vol.area_hessian(coords, faces)
    shift = _pinv_drop_null(H0) @ gradA
    flat = coords.ravel()
    Cshift = (ham.anm_full_hessian((flat + h * shift).reshape(coords.shape),
                                   pairs, rest, stiffness)
              - ham.anm_full_hessian((flat - h * shift).reshape(coords.shape),
                                     pairs, rest, stiffness)) / (2 * h)
    return HA - Cshift, HA


def first_order_mode_mixing(w0: np.ndarray, V0: np.ndarray, Hp: np.ndarray,
                            n_mode: int, active: np.ndarray | None = None,
                            deg_rtol: float = 1e-6) -> np.ndarray:
    r"""Поправка 1-го порядка к моде |n⁽¹⁾⟩ = Σ_{m≠n} ⟨m|H'|n⟩/(λn−λm)|m⟩."""
    n = len(w0)
    if active is None:
        active = np.arange(n)
    scale = max(float(np.abs(w0).max()), 1.0)
    Hpb = V0.T @ Hp @ V0
    corr = np.zeros(n)
    for m in active:
        if m == n_mode:
            continue
        denom = w0[n_mode] - w0[m]
        if abs(denom) < deg_rtol * scale:
            continue
        corr += (Hpb[m, n_mode] / denom) * V0[:, m]
    return corr
