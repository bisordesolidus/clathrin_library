r"""
rigidbody.py — SE(3) rigid-body гессиан клатриновой оболочки (MODEL.md §5.2).

Каждый хаб-трискелион — твёрдое тело g_i=(R_i,r_i)∈SE(3), 6 DOF. Контактные
«бусины» жёстко привязаны к хабам; пружина между бусинами p (на хабе i) и q
(на хабе j) даёт вклад k·J_c J_cᵀ с 12-вектором (вывод — MODEL.md §5.2):

    δℓ_c = (δx_i^p − δx_j^q)·n̂_c,   δx_i^p = R_i(u_i + φ_i×ρ_i^p),
    J_c = [ R_iᵀ n̂ ; ρ_i^p × R_iᵀ n̂ ; −R_jᵀ n̂ ; −ρ_j^q × R_jᵀ n̂ ] ∈ ℝ¹².

Сборка — тем же `hamiltonian.assemble`, что и точечный ANM (M1), только J длиной
12 вместо 6.

Нулевые моды (телесные!). Глобальное движение g_i→G·g_i в телесном репере есть
ξ_i = Ad_{g_i^{-1}} η для η∈se(3); эти 6 мод — ядро H0 (жёсткое движение не меняет
ни одной длины контакта).

КОНТАКТНАЯ МОДЕЛЬ (v1). На каждом ребре 4 ноги (2 прокс + 2 дист из routing,
MODEL.md §4.3). Бусины ставим в области перекрытия у ребра: 2 внешних (прокс) и
2 внутренних (дист), антипараллельно — структура Morris/Smith. χ (хиральность)
входит через МИРОВУЮ связывающую нормаль ноги n̂_a=R_i·binding_normal(a).
Длины покоя = фактические расстояния (ANM-стиль ⇒ структура равновесна).

  [v1-упрощение: бусины кладём прямо в зону перекрытия и жёстко крепим к хабам,
   не моделируя изгиб ноги в колене. Изогнутая нога с дистальными контактами
   Morris (883–888↔981–984 и т.д.) и калибровка по PDB 6SCT — следующая версия.
   Для ворот M2b (SE(3)-сборка, 6 нулевых=se(3), Iₕ, хиральность) этого хватает.]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hamiltonian as ham
import lattice as lat
import routing as rt
import se3
from triskelion import Triskelion

# Геометрия контактной модели (нм). Умолчания подобраны для наглядной жёсткости.
R_LEG = 3.0          # поперечный сдвиг бусины вдоль связывающей нормали (несёт χ)
SHELL_OUT = 2.0      # радиальный сдвиг внешней (проксимальной) оболочки
SHELL_IN = 2.0       # радиальный сдвиг внутренней (дистальной) оболочки
S_OFF = 3.0          # осевой сдвиг ± вдоль ребра


@dataclass
class SE3Model:
    latt: object
    rot: dict
    tris: Triskelion
    hand: int
    g: list                       # хаб-преобразования g_i (4×4)
    beads: list                   # (owner, ρ_body, world)
    contacts: list                # (bead_p, bead_q)
    H: object                     # разрежённый 6N×6N гессиан

    @property
    def n_dof(self) -> int:
        return 6 * self.latt.n_vertices


def hub_transforms(latt, rot: dict) -> list[np.ndarray]:
    r"""g_i = [[R_i, r_i],[0,1]] ∈ SE(3): R_i — хаб-репер, r_i — координата хаба."""
    R = rt.hub_frames(latt, rot)
    C = latt.coords
    g = []
    for i in range(latt.n_vertices):
        T = np.eye(4)
        T[:3, :3] = R[i]
        T[:3, 3] = C[i]
        g.append(T)
    return g


def build_se3_model(latt=None, tris: Triskelion | None = None, hand: int = +1,
                    stiffness: float = 1.0,
                    r_leg: float = R_LEG, shell_out: float = SHELL_OUT,
                    shell_in: float = SHELL_IN, s_off: float = S_OFF) -> SE3Model:
    r"""Построить SE(3) rigid-body модель оболочки (по умолчанию — додекаэдр)."""
    if latt is None:
        latt = lat.dodecahedron()
    if tris is None:
        tris = Triskelion()
    rot = rt.rotation_system(latt)
    g = hub_transforms(latt, rot)
    R = [gi[:3, :3] for gi in g]
    C = latt.coords

    routing = rt.leg_routing(latt, rot, hand)
    distal_of = {distal: leg for leg, distal in routing.items()}   # φ⁻¹

    beads: list = []              # (owner, ρ_body, world)

    def add_bead(owner: int, world: np.ndarray) -> int:
        rho = R[owner].T @ (world - C[owner])
        beads.append((owner, rho, world))
        return len(beads) - 1

    def leg_binding_normal_world(dart) -> np.ndarray:
        owner = dart[0]
        a = rt.leg_index(owner, dart[1], rot)
        return R[owner] @ tris.binding_normal(a)

    contacts: list = []

    for e in latt.edges:
        i, j = int(e[0]), int(e[1])
        t = C[j] - C[i]; t /= np.linalg.norm(t)
        mid = 0.5 * (C[i] + C[j])
        rad = mid / np.linalg.norm(mid)
        rad = rad - (rad @ t) * t; rad /= np.linalg.norm(rad)

        p_i, p_j = (i, j), (j, i)                       # проксимальные дротики
        d_a, d_b = distal_of[(i, j)], distal_of[(j, i)]  # дистальные ноги (дротики)

        # центры бусин: прокс — внешняя оболочка (+rad), дист — внутренняя (−rad);
        # ±s_off вдоль ребра (антипараллельно); + χ-сдвиг по связывающей нормали.
        specs = [(p_i, +s_off, +shell_out), (p_j, -s_off, +shell_out),
                 (d_a, +s_off, -shell_in),  (d_b, -s_off, -shell_in)]
        bid = {}
        for dart, sax, hrad in specs:
            world = mid + sax * t + hrad * rad + r_leg * leg_binding_normal_world(dart)
            bid[dart] = add_bead(dart[0], world)

        # контактный пучок ребра: dd + 2·dp (Morris) + pp + перекрёстные (жёсткость)
        for u, v in [(d_a, d_b), (d_a, p_i), (d_b, p_j),
                     (p_i, p_j), (d_a, p_j), (d_b, p_i)]:
            contacts.append((bid[u], bid[v]))

    def terms():
        for bp, bq in contacts:
            op, rho_p, xp = beads[bp]
            oq, rho_q, xq = beads[bq]
            n = xp - xq
            n /= np.linalg.norm(n)
            Rp_n = R[op].T @ n
            Rq_n = R[oq].T @ n
            J = np.concatenate([Rp_n, np.cross(rho_p, Rp_n),
                                -Rq_n, -np.cross(rho_q, Rq_n)])
            idx = list(range(6 * op, 6 * op + 6)) + list(range(6 * oq, 6 * oq + 6))
            yield idx, J, stiffness

    H = ham.assemble(6 * latt.n_vertices, terms())
    return SE3Model(latt, rot, tris, hand, g, beads, contacts, H)


def se3_zero_modes(model_or_g) -> np.ndarray:
    r"""6 аналитических нулевых мод ξ_i = Ad(g_i^{-1})·η, η∈se(3), как (6N,6)."""
    g = model_or_g.g if isinstance(model_or_g, SE3Model) else model_or_g
    N = len(g)
    Z = np.zeros((6 * N, 6))
    for a in range(6):
        eta = np.zeros(6); eta[a] = 1.0
        for i in range(N):
            Z[6 * i:6 * i + 6, a] = se3.Ad(se3.inverse(g[i])) @ eta
    return Z
