r"""
relax.py — фрустрация и риманова релаксация оболочки (MODEL.md §5.3, веха M4).

В отличие от M2b (ANM-стиль, длина покоя = фактическое расстояние ⇒ фрустрация 0),
здесь у трискелиона СОБСТВЕННАЯ геометрия покоя, и клетка, что под неё не
подходит, ФРУСТРИРОВАНА. Энергия:

  E = ½ Σ_edges k_L (|r_i − r_j| − L)²                         (масштаб/рёбра)
    + ½ Σ_bead-pairs k_b |x_p − x_q|²                          (ориентация/кручение)

Бусины — фиксированные телесные точки на проксимальных сегментах ног со сдвигом
вдоль связывающей нормали n̂(χ); на ребре две антипараллельные ноги дают пары
бусин, которые в идеале совпадают. Геометрическая невозможность совпасть на всех
рёбрах одновременно = фрустрация (механизм Шайна: кручение ноги; χ входит через
сдвиг). Длина покоя бусинных пружин = 0 (векторная пружина, устойчиво везде) —
никакого SE(3)-log и его сингулярности при θ=π.

Релаксация — Гаусс-Ньютон/Левенберг-Марквардт на SE(3)^N: гессиан H = Σ k J Jᵀ
(тот же 12-вектор J_c, `hamiltonian.assemble`), шаг (H+λI)Δξ = −grad, ретракция
g ← g·exp(Δξ). Гессиан считаем только в истинном минимуме.

  [v1: бусины только на проксимальных сегментах (в straight-leg модели они у
   ребра); дистальные контакты Morris и калибровка (δ, k, ψ) по PDB — далее.
   Открывает T10: релакс. каждой клетки → E_min/N → отделяются ли 15 Шайна.]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import hamiltonian as ham
import lattice as lat
import routing as rt
import se3
from rigidbody import hub_transforms
from triskelion import EDGE_NM, Triskelion

# Геометрия бусин (нм) и жёсткости. Калибровка из 6SCT.
PROX_LEN = 11.4                                  # проксималь, прямой размах (6SCT)
S1, S2 = PROX_LEN / 3.0, 2.0 * PROX_LEN / 3.0    # бусины В ПРЕДЕЛАХ проксимали
DELTA = 1.5                                      # поперечный сдвиг = радиус ноги (6SCT)
K_EDGE = 1.0
K_BEAD = 1.0


@dataclass
class FrustrationModel:
    latt: object
    tris: Triskelion
    hand: int
    rot: dict
    g0: list                                     # начальные позы хабов
    contacts: list = field(default_factory=list)  # (typ,i,j,ρ_i,ρ_j,d0,k)


def build_frustration_model(latt=None, tris: Triskelion | None = None,
                            hand: int = +1, delta: float = DELTA,
                            k_edge: float = K_EDGE, k_bead: float = K_BEAD
                            ) -> FrustrationModel:
    r"""Собрать модель фрустрации (по умолчанию — додекаэдр)."""
    if latt is None:
        latt = lat.dodecahedron()
    if tris is None:
        tris = Triskelion()
    rot = rt.rotation_system(latt)
    g0 = hub_transforms(latt, rot)

    def bead(owner: int, nb: int, s: float) -> np.ndarray:
        a = rt.leg_index(owner, nb, rot)
        return s * tris.leg_axis(a) + delta * tris.binding_normal(a)

    contacts = []
    for e in latt.edges:
        i, j = int(e[0]), int(e[1])
        contacts.append(("edge", i, j, np.zeros(3), np.zeros(3), EDGE_NM, k_edge))
        for s, sp in [(S1, S2), (S2, S1)]:       # антипараллельные пары бусин
            contacts.append(("bead", i, j, bead(i, j, s), bead(j, i, sp), 0.0, k_bead))
    return FrustrationModel(latt, tris, hand, rot, g0, contacts)


def _terms(model: FrustrationModel, g):
    r"""Для каждого скалярного ограничения — (idx, J, k, residual).
    Ребро → 1 член (пружина длины); бусина → 3 (векторная пружина, d0=0)."""
    R = [gi[:3, :3] for gi in g]
    C = np.array([gi[:3, 3] for gi in g])
    for typ, i, j, ri, rj, d0, k in model.contacts:
        xp = C[i] + R[i] @ ri
        xq = C[j] + R[j] @ rj
        dvec = xp - xq
        idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        if typ == "edge":
            n = dvec / np.linalg.norm(dvec)
            J = np.concatenate([R[i].T @ n, np.cross(ri, R[i].T @ n),
                                -R[j].T @ n, -np.cross(rj, R[j].T @ n)])
            yield idx, J, k, float(np.linalg.norm(dvec) - d0)
        else:
            for d in range(3):
                ed = np.zeros(3); ed[d] = 1.0
                J = np.concatenate([R[i].T @ ed, np.cross(ri, R[i].T @ ed),
                                    -R[j].T @ ed, -np.cross(rj, R[j].T @ ed)])
                yield idx, J, k, float(dvec @ ed)


def energy(model: FrustrationModel, g) -> float:
    r"""Полная энергия фрустрации E = ½ Σ k·residual²."""
    return 0.5 * sum(k * r * r for _, _, k, r in _terms(model, g))


def gradient(model: FrustrationModel, g) -> np.ndarray:
    r"""Аналитический градиент энергии в касательном пространстве (6N)."""
    grad = np.zeros(6 * model.latt.n_vertices)
    for idx, J, k, res in _terms(model, g):
        grad[np.asarray(idx)] += k * res * J
    return grad


def gn_hessian(model: FrustrationModel, g):
    r"""Гессиан Гаусса–Ньютона H = Σ k J Jᵀ — для ШАГА релаксации (разрежённый).
    NB: при ненулевой невязке НЕ инвариантен к глобальному вращению (нет 3
    вращательных нулевых мод) — для СПЕКТРА использовать `hessian` (истинный)."""
    n_dof = 6 * model.latt.n_vertices
    return ham.assemble(n_dof, ((idx, J, k) for idx, J, k, _ in _terms(model, g)))


def _retract(g, dx):
    return [g[i] @ se3.exp_se3(dx[6 * i:6 * i + 6]) for i in range(len(g))]


def hessian(model: FrustrationModel, g, eps: float = 1e-6) -> np.ndarray:
    r"""Истинный гессиан в конфигурации g: конечная разность аналитического
    градиента по касательным DOF. В минимуме (grad≈0) это точный римановый
    гессиан — инвариантен к глобальному SE(3) ⇒ ровно 6 нулевых мод."""
    n_dof = 6 * len(g)
    H = np.zeros((n_dof, n_dof))
    for a in range(n_dof):
        d = np.zeros(n_dof); d[a] = eps
        H[:, a] = (gradient(model, _retract(g, d))
                   - gradient(model, _retract(g, -d))) / (2 * eps)
    return 0.5 * (H + H.T)


@dataclass
class RelaxResult:
    g: list
    E0: float
    E_min: float
    n_iter: int
    converged: bool


def relax(model: FrustrationModel, max_iter: int = 100,
          grad_tol: float = 1e-8) -> RelaxResult:
    r"""Релаксация Левенберга–Марквардта на SE(3)^N."""
    N = model.latt.n_vertices
    g = [gi.copy() for gi in model.g0]
    E0 = energy(model, g)
    E = E0
    lam = 1e-2
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        grad = gradient(model, g)
        H = gn_hessian(model, g).toarray()
        if np.linalg.norm(grad) < grad_tol:
            converged = True
            break
        stepped = False
        for _ in range(30):
            dx = np.linalg.solve(H + lam * np.eye(6 * N), -grad)
            gn = [g[i] @ se3.exp_se3(dx[6 * i:6 * i + 6]) for i in range(N)]
            En = energy(model, gn)
            if En < E:
                g, E = gn, En
                lam = max(lam * 0.5, 1e-9)
                stepped = True
                break
            lam *= 3.0
        if not stepped:
            converged = True
            break
    return RelaxResult(g, E0, E, it, converged)


def frustration_per_vertex(model: FrustrationModel, **kw) -> float:
    r"""Релаксированная энергия фрустрации на вершину E_min/N — ключевая величина
    для T10 (отделяются ли 15 разрешённых клеток Шайна)."""
    res = relax(model, **kw)
    return res.E_min / model.latt.n_vertices
