r"""
routing.py — укладка ориентированных трискелионов на решётку (MODEL.md §4).

ПРАВИЛО ГОЛОВА-ХВОСТ = ТРАССИРОВКА ГРАНЕЙ.
Нога трискелиона i идёт проксимальным сегментом вдоль ребра i→j, затем в
вершине j поворачивает на σ-СЛЕДУЮЩЕЕ ребро и продолжается дистальным сегментом
j→k. Формально это перестановка обхода граней на вращательной системе:

    дротик (полуребро)     d = (i, j)      — проксимальный сегмент ноги i к j;
    инволюция ребра        α(d) = (j, i);
    поворот в вершине      σ(d) = (i, следующий за j сосед в CCW-порядке rot[i]);
    обход граней           φ(d) = σ(α(d));
    дистальный сегмент ноги d есть дротик φ(d).

Тогда:
  * итерации φ обходят ГРАНЬ ⇒ орбиты φ = грани (12 пятиугольников додекаэдра);
  * каждое ребро несёт ровно 4 сегмента: 2 проксимальных (антипараллельны) +
    2 дистальных (антипараллельны) — структура ребра Morris/Smith получается
    автоматически, т.к. φ — биекция дротиков;
  * ВСЕ повороты в одну сторону (uniform hand) ⟺ орбиты = грани ⟺ валидная
    решётка одной хиральности. Смешанные повороты сливают грани → ЗАПРЕЩЕНО.
  * hand=+1 и hand=−1 — два энантиомера (зеркальные, изоспектральные — T4).

Хаб-реперы R_i: ê₃ радиально наружу, ê₁ к первому соседу в rot[i]; три ноги
трискелиона (азимуты 0/120/240°) смотрят на трёх соседей.
"""

from __future__ import annotations

import numpy as np

Dart = tuple[int, int]      # направленное ребро i→j = проксимальный сегмент ноги i


# --------------------------------------------------------------------------- #
#  Касательный репер и вращательная система
# --------------------------------------------------------------------------- #
def tangent_frame(r_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Правый касательный репер (ê₁, ê₂) при ê₃ = r_hat: ê₁×ê₂ = ê₃."""
    r_hat = np.asarray(r_hat, dtype=float)
    a = np.eye(3)[int(np.argmin(np.abs(r_hat)))]
    e1 = a - (a @ r_hat) * r_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(r_hat, e1)
    return e1, e2


def rotation_system(latt) -> dict[int, tuple[int, ...]]:
    r"""σ: для каждой вершины — соседи в порядке CCW вокруг внешней нормали.
    Циклический порядок не зависит от выбора ê₁ в касательном репере."""
    C = latt.coords
    rh = latt.radial_directions()
    rot: dict[int, tuple[int, ...]] = {}
    for i in range(latt.n_vertices):
        e1, e2 = tangent_frame(rh[i])
        nbrs = list(latt.graph.neighbors(i))
        ang = [np.arctan2((C[j] - C[i]) @ e2, (C[j] - C[i]) @ e1) for j in nbrs]
        rot[i] = tuple(int(nbrs[k]) for k in np.argsort(ang))
    return rot


# --------------------------------------------------------------------------- #
#  Дротики и перестановки комбинаторной карты
# --------------------------------------------------------------------------- #
def darts(latt) -> list[Dart]:
    r"""Все дротики (полурёбра) = все ноги. Их 2E = 3N."""
    return [(i, int(j)) for i in range(latt.n_vertices)
            for j in latt.graph.neighbors(i)]


def alpha(d: Dart) -> Dart:
    r"""Инволюция ребра: α(i,j) = (j,i)."""
    return (d[1], d[0])


def _hand(hand, i: int) -> int:
    return hand[i] if isinstance(hand, dict) else hand


def sigma(d: Dart, rot: dict, hand=+1) -> Dart:
    r"""Поворот в хвостовой вершине: следующий сосед в rot (hand=+1) или
    предыдущий (hand=−1). hand может быть словарём {вершина: ±1} — для проверки
    запрета смешанных поворотов."""
    i, j = d
    nb = rot[i]
    step = _hand(hand, i)
    return (i, nb[(nb.index(j) + step) % len(nb)])


def face_permutation(d: Dart, rot: dict, hand=+1) -> Dart:
    r"""φ = σ∘α — обход граней; φ(проксимальный дротик) = дистальный дротик."""
    return sigma(alpha(d), rot, hand)


def trace_orbits(latt, rot: dict, hand=+1) -> list[list[Dart]]:
    r"""Орбиты φ. При валидной укладке = грани (циклы вершин [d[0] for d in orbit])."""
    seen: set = set()
    orbits: list[list[Dart]] = []
    for d in darts(latt):
        if d in seen:
            continue
        orbit = []
        x = d
        while x not in seen:
            seen.add(x)
            orbit.append(x)
            x = face_permutation(x, rot, hand)
        orbits.append(orbit)
    return orbits


def orbit_vertex_sets(orbits: list[list[Dart]]) -> set[frozenset]:
    return {frozenset(d[0] for d in orbit) for orbit in orbits}


def is_head_to_tail_consistent(latt, rot: dict, hand=+1) -> bool:
    r"""Орбиты φ совпадают с гранями ⟺ уникальная хиральность (валидно).
    Смешанные повороты сливают грани → False."""
    faces = {frozenset(f) for f in latt.faces}
    return orbit_vertex_sets(trace_orbits(latt, rot, hand)) == faces


# --------------------------------------------------------------------------- #
#  Маршрутизация ног и состав рёбер
# --------------------------------------------------------------------------- #
def leg_index(i: int, j: int, rot: dict) -> int:
    r"""Локальный номер ноги a∈{0,1,2}: rot[i][a] = j (нога a смотрит на соседа a)."""
    return rot[i].index(j)


def leg_routing(latt, rot: dict, hand=+1) -> dict[Dart, Dart]:
    r"""Каждой ноге (проксимальный дротик d) — её дистальный дротик φ(d)."""
    return {d: face_permutation(d, rot, hand) for d in darts(latt)}


def edge_segments(latt, rot: dict, hand=+1) -> dict[frozenset, dict]:
    r"""Состав каждого ребра: списки проксимальных и дистальных дротиков.
    В валидной замкнутой клетке — ровно 2 + 2."""
    seg = {frozenset(e): {"proximal": [], "distal": []} for e in latt.edges}
    for d, distal in leg_routing(latt, rot, hand).items():
        seg[frozenset(d)]["proximal"].append(d)          # проксимальный на ребре {d}
        seg[frozenset(distal)]["distal"].append(d)        # дистальный на ребре φ(d)
    return seg


# --------------------------------------------------------------------------- #
#  Хаб-реперы
# --------------------------------------------------------------------------- #
def hub_frames(latt, rot: dict) -> np.ndarray:
    r"""R_i ∈ SO(3), (N,3,3): столбцы (ê₁, ê₂, ê₃). ê₃ радиально наружу,
    ê₁ — касательная к первому соседу rot[i][0]; три ноги смотрят на соседей."""
    C = latt.coords
    e3 = latt.radial_directions()
    N = latt.n_vertices
    R = np.zeros((N, 3, 3))
    for i in range(N):
        n0 = rot[i][0]
        t = C[n0] - C[i]
        t = t - (t @ e3[i]) * e3[i]
        t /= np.linalg.norm(t)
        R[i, :, 0] = t
        R[i, :, 1] = np.cross(e3[i], t)
        R[i, :, 2] = e3[i]
    return R
