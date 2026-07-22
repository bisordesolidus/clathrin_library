r"""
se3.py — группа SE(3) = R^3 ⋊ SO(3) и её алгебра Ли se(3).

Фундамент всей модели: узел-трискелион — твёрдое тело g = (R, r) ∈ SE(3),
вариации живут в se(3) ≅ R^6 (см. MODEL.md §2). Здесь — exp/log/Ad/ad с
численно устойчивыми рядами при малых углах и жёстко зафиксированными
соглашениями.

СОГЛАШЕНИЯ (одинаковые во всём проекте!)
----------------------------------------
* Вектор алгебры:  ξ = [u; φ] ∈ R^6,  u — ТРАНСЛЯЦИЯ (первая), φ — ПОВОРОТ.
  (Тот же порядок, что в гессиане MODEL.md §5.2: блок трансляции, затем блок вращения.)
* Элемент группы:  T = [[R, r],
                        [0, 1]]  ∈ R^{4×4},  R ∈ SO(3), r ∈ R^3.
* hat:  ξ̂ = [[φ̂, u],
             [0, 0]] ∈ R^{4×4},   где φ̂ v = φ × v  (so(3)).

ФОРМУЛЫ
-------
so(3):  R = exp(φ̂) = I + A φ̂ + B φ̂²,       θ = |φ|,
        A = sinθ/θ,  B = (1−cosθ)/θ².                      (Родригес)

se(3):  exp(ξ̂) = [[exp(φ̂), V u],
                  [0,       1  ]],
        V = I + B φ̂ + C φ̂²,   C = (θ−sinθ)/θ³.            (левый якобиан)

Присоединённые представления (порядок (u,φ)):
        Ad_T = [[R, r̂ R],            ad_ξ = [[φ̂, û],
                [0, R  ]],                    [0,  φ̂]].
Тождество, связывающее их:  Ad(exp ξ) = exp(ad ξ).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hat_so3", "vee_so3", "exp_so3", "log_so3",
    "hat_se3", "vee_se3", "exp_se3", "log_se3",
    "Ad", "ad", "inverse", "compose", "left_jacobian_so3",
]

# Порог перехода на ряды Тейлора (относительно угла θ в радианах).
_SMALL = 1e-6


# --------------------------------------------------------------------------- #
#  so(3)
# --------------------------------------------------------------------------- #
def hat_so3(w: np.ndarray) -> np.ndarray:
    r"""R^3 → 3×3 кососимметричная: ŵ v = w × v."""
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee_so3(W: np.ndarray) -> np.ndarray:
    r"""3×3 кососимметричная → R^3 (обратна hat_so3)."""
    W = np.asarray(W, dtype=float)
    return np.array([W[2, 1] - W[1, 2],
                     W[0, 2] - W[2, 0],
                     W[1, 0] - W[0, 1]]) * 0.5


def _coeffs_AB(theta: float) -> tuple[float, float]:
    r"""A = sinθ/θ, B = (1−cosθ)/θ²  с рядами при малых θ."""
    if theta < _SMALL:
        t2 = theta * theta
        A = 1.0 - t2 / 6.0 + t2 * t2 / 120.0
        B = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
    return A, B


def _coeff_C(theta: float) -> float:
    r"""C = (θ−sinθ)/θ³  с рядом при малых θ."""
    if theta < _SMALL:
        t2 = theta * theta
        return 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
    return (theta - np.sin(theta)) / (theta ** 3)


def exp_so3(w: np.ndarray) -> np.ndarray:
    r"""so(3) → SO(3):  R = I + A ŵ + B ŵ²  (формула Родригеса)."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    K = hat_so3(w)
    A, B = _coeffs_AB(theta)
    return np.eye(3) + A * K + B * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    r"""SO(3) → so(3). Устойчиво при θ→0 и θ→π (неоднозначность знака оси при θ=π)."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    # NB: vee_so3 несёт множитель 1/2 (точный обратный к hat), поэтому
    # vee_so3(R−Rᵀ) = 2 sinθ · a; отсюда лишнее деление на 2 в обеих ветках.
    if theta < _SMALL:
        # w = vee(R−Rᵀ)/(2A),  A = sinθ/θ ≈ 1
        A, _ = _coeffs_AB(theta)
        return vee_so3(R - R.T) / (2.0 * A)
    if theta < np.pi - _SMALL:
        return theta / (2.0 * np.sin(theta)) * vee_so3(R - R.T)

    # θ ≈ π: sinθ→0, ось из симметричной части (R+I)/2 ≈ a aᵀ.
    # Знак оси фиксируем по антисимметричной части (∝ +a при θ<π);
    # при ровно θ=π знак объективно неоднозначен (обе оси дают тот же R).
    M = (R + np.eye(3)) * 0.5
    k = int(np.argmax(np.diag(M)))
    axis = M[:, k] / np.sqrt(max(M[k, k], 1e-300))
    axis = axis / np.linalg.norm(axis)
    s = vee_so3(R - R.T)                       # = 2 sinθ · a
    if float(s @ axis) < 0.0:
        axis = -axis
    return theta * axis


def left_jacobian_so3(w: np.ndarray) -> np.ndarray:
    r"""V = I + B ŵ + C ŵ²  (переводит u в трансляцию: r = V u)."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    K = hat_so3(w)
    _, B = _coeffs_AB(theta)
    C = _coeff_C(theta)
    return np.eye(3) + B * K + C * (K @ K)


# --------------------------------------------------------------------------- #
#  se(3)
# --------------------------------------------------------------------------- #
def hat_se3(xi: np.ndarray) -> np.ndarray:
    r"""R^6 = [u; φ] → 4×4:  [[φ̂, u], [0, 0]]."""
    xi = np.asarray(xi, dtype=float)
    u, phi = xi[:3], xi[3:]
    X = np.zeros((4, 4))
    X[:3, :3] = hat_so3(phi)
    X[:3, 3] = u
    return X


def vee_se3(X: np.ndarray) -> np.ndarray:
    r"""4×4 → R^6 = [u; φ]  (обратна hat_se3)."""
    X = np.asarray(X, dtype=float)
    return np.concatenate([X[:3, 3], vee_so3(X[:3, :3])])


def exp_se3(xi: np.ndarray) -> np.ndarray:
    r"""se(3) → SE(3):  R = exp(φ̂),  r = V(φ) u."""
    xi = np.asarray(xi, dtype=float)
    u, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = exp_so3(phi)
    T[:3, 3] = left_jacobian_so3(phi) @ u
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    r"""SE(3) → se(3):  φ = log(R),  u = V(φ)⁻¹ r."""
    T = np.asarray(T, dtype=float)
    R, r = T[:3, :3], T[:3, 3]
    phi = log_so3(R)
    V = left_jacobian_so3(phi)
    u = np.linalg.solve(V, r)   # V всегда обратима при |φ|<2π
    return np.concatenate([u, phi])


# --------------------------------------------------------------------------- #
#  Групповые операции и присоединённые представления
# --------------------------------------------------------------------------- #
def inverse(T: np.ndarray) -> np.ndarray:
    r"""T⁻¹ = [[Rᵀ, −Rᵀ r], [0, 1]]."""
    T = np.asarray(T, dtype=float)
    R, r = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ r
    return Ti


def compose(*Ts: np.ndarray) -> np.ndarray:
    r"""Произведение элементов группы (матричное)."""
    out = np.eye(4)
    for T in Ts:
        out = out @ np.asarray(T, dtype=float)
    return out


def Ad(T: np.ndarray) -> np.ndarray:
    r"""Присоединённое представление группы, порядок (u,φ):
        Ad_T = [[R, r̂ R], [0, R]] ∈ R^{6×6}."""
    T = np.asarray(T, dtype=float)
    R, r = T[:3, :3], T[:3, 3]
    M = np.zeros((6, 6))
    M[:3, :3] = R
    M[:3, 3:] = hat_so3(r) @ R
    M[3:, 3:] = R
    return M


def ad(xi: np.ndarray) -> np.ndarray:
    r"""Присоединённое представление алгебры, порядок (u,φ):
        ad_ξ = [[φ̂, û], [0, φ̂]] ∈ R^{6×6}."""
    xi = np.asarray(xi, dtype=float)
    u, phi = xi[:3], xi[3:]
    W, U = hat_so3(phi), hat_so3(u)
    M = np.zeros((6, 6))
    M[:3, :3] = W
    M[:3, 3:] = U
    M[3:, 3:] = W
    return M
