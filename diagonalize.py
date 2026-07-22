r"""
diagonalize.py — спектр гессиана и классификация мод (MODEL.md §6).

Пока: плотный симметричный eigh (для малых/средних сетей), фильтр нулевых мод,
аналитические rigid-body моды (6-мерное ядро) и группировка по вырождению —
достаточно для ворот вехи M1 (ровно 6 нулевых, вырождения Iₕ ∈ {1,3,4,5}).
Разрежённый eigsh(shift-invert), проекция на VSH и на ирепы точечной группы —
вехи M4+.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse


# --------------------------------------------------------------------------- #
#  Спектр
# --------------------------------------------------------------------------- #
def spectrum(H) -> tuple[np.ndarray, np.ndarray]:
    r"""Полный спектр симметричного H: (собственные значения ↑, векторы столбцами)."""
    A = H.toarray() if issparse(H) else np.asarray(H, dtype=float)
    A = 0.5 * (A + A.T)
    return np.linalg.eigh(A)


def _scale(w: np.ndarray) -> float:
    return max(float(np.abs(w).max()), 1.0)


# --------------------------------------------------------------------------- #
#  Нулевые моды и знакоопределённость
# --------------------------------------------------------------------------- #
def zero_mode_indices(w: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    r"""Индексы (почти) нулевых собственных значений — размерность ядра."""
    return np.where(np.abs(w) < rtol * _scale(w))[0]


def n_zero_modes(w: np.ndarray, rtol: float = 1e-6) -> int:
    return int(zero_mode_indices(w, rtol).size)


def is_psd(w: np.ndarray, rtol: float = 1e-6) -> bool:
    r"""H ⪰ 0: наименьшее с.з. не ниже −rtol·scale."""
    return bool(w.min() > -rtol * _scale(w))


# --------------------------------------------------------------------------- #
#  Rigid-body моды (аналитическое ядро)
# --------------------------------------------------------------------------- #
def rigid_body_modes(coords: np.ndarray) -> np.ndarray:
    r"""6 нулевых мод жёсткого тела как ортонормированные столбцы (3N, 6):
    3 трансляции + 3 бесконечно-малых вращения δr_i = ê_a × (r_i − r̄)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    c = coords - coords.mean(axis=0)
    M = np.zeros((3 * n, 6))
    for a in range(3):
        M[a::3, a] = 1.0                                  # трансляции
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        M[:, 3 + a] = np.cross(e, c).ravel()              # вращения
    Q, _ = np.linalg.qr(M)
    return Q


def subspace_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    r"""Наибольший главный угол между линейными оболочками столбцов A и B
    (0° ⇔ подпространства совпадают)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


# --------------------------------------------------------------------------- #
#  Группировка по вырождению
# --------------------------------------------------------------------------- #
def degeneracy_groups(w: np.ndarray, zero_rtol: float = 1e-6,
                      deg_rtol: float = 1e-6,
                      drop_zero: bool = True) -> list[tuple[float, int]]:
    r"""Сгруппировать ненулевые с.з. в вырожденные кластеры.
    Возвращает [(среднее значение, кратность), …] по возрастанию."""
    scale = _scale(w)
    ww = np.sort(w)
    if drop_zero:
        ww = ww[np.abs(ww) >= zero_rtol * scale]
    groups: list[tuple[float, int]] = []
    i = 0
    while i < len(ww):
        j = i
        while j + 1 < len(ww) and (ww[j + 1] - ww[j]) < deg_rtol * scale:
            j += 1
        groups.append((float(ww[i:j + 1].mean()), j - i + 1))
        i = j + 1
    return groups


def multiplicities(w: np.ndarray, **kw) -> list[int]:
    r"""Кратности вырожденных групп ненулевых мод (для проверки симметрии)."""
    return [m for _, m in degeneracy_groups(w, **kw)]
