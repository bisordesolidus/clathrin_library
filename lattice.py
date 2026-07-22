r"""
lattice.py — решётки трискелионных узлов (MODEL.md §4).

Пока в объёме вех M1/M2: замкнутая клетка-додекаэдр (n=20, группа Iₕ) — граф,
сферическое вложение равновесных координат хабов и грани (12 пятиугольников).
Полный спиральный генератор фуллеренов и плоская решётка — веха M3.

Контейнер `Lattice` — общий для всех решёток:
  * coords : (N,3) равновесные положения хабов r⁰_i;
  * edges  : список рёбер (i<j) — «ноги» между хабами;
  * faces  : грани как циклы вершин, ориентированные НАРУЖУ (для вращательной
             системы σ_i в M2, объёма в M6, проверки Эйлера);
  * graph  : networkx.Graph.

Фуллереновая топология (Schein 2009): n трёхвалентных вершин, E = 3n/2 рёбер,
F = n/2 + 2 граней, ровно 12 пятиугольников, (n−20)/2 шестиугольников.
Для додекаэдра: V=20, E=30, F=12 (все пятиугольники), V−E+F=2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from triskelion import EDGE_NM

_PHI = (1.0 + np.sqrt(5.0)) / 2.0          # золотое сечение
_CANON_EDGE = 2.0 / _PHI                    # длина ребра канонического додекаэдра


# --------------------------------------------------------------------------- #
#  Контейнер
# --------------------------------------------------------------------------- #
@dataclass
class Lattice:
    coords: np.ndarray               # (N, 3)
    edges: list[tuple[int, int]]     # рёбра, i < j
    faces: list[list[int]]           # циклы вершин, ориентированы наружу
    graph: nx.Graph
    name: str = ""
    closed: bool = True              # замкнутая клетка (True) или плоская (False)

    # ---- размеры --------------------------------------------------------- #
    @property
    def n_vertices(self) -> int:
        return len(self.coords)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    # ---- топологические инварианты --------------------------------------- #
    def euler(self) -> int:
        r"""Эйлерова характеристика V − E + F (для сферы = 2)."""
        return self.n_vertices - self.n_edges + self.n_faces

    def degrees(self) -> np.ndarray:
        return np.array([d for _, d in sorted(self.graph.degree())])

    def is_three_regular(self) -> bool:
        return bool(np.all(self.degrees() == 3))

    def face_type_counts(self) -> dict[int, int]:
        r"""Сколько граней каждого размера: {5: 12} для додекаэдра."""
        return dict(Counter(len(f) for f in self.faces))

    # ---- геометрия ------------------------------------------------------- #
    def edge_lengths(self) -> np.ndarray:
        return np.array([np.linalg.norm(self.coords[i] - self.coords[j])
                         for i, j in self.edges])

    def circumradii(self) -> np.ndarray:
        r"""Радиусы вершин от центра масс (для замкнутой клетки — ≈ const)."""
        c = self.coords.mean(axis=0)
        return np.linalg.norm(self.coords - c, axis=1)

    def radial_directions(self) -> np.ndarray:
        r"""Единичные радиальные орты (наружу) в каждой вершине — будущие ê₃."""
        c = self.coords.mean(axis=0)
        d = self.coords - c
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def edge_face_incidence(self) -> dict[tuple[int, int], int]:
        r"""Сколько граней содержит каждое ребро (в замкнутой клетке = 2)."""
        cnt: Counter = Counter()
        for f in self.faces:
            for k in range(len(f)):
                i, j = f[k], f[(k + 1) % len(f)]
                cnt[(min(i, j), max(i, j))] += 1
        return dict(cnt)

    # ---- сводная проверка ------------------------------------------------ #
    def validate(self) -> dict:
        r"""Сводка инвариантов + булевы флаги корректности (для тестов/main)."""
        planar, _ = nx.check_planarity(self.graph)
        inc = self.edge_face_incidence()
        L = self.edge_lengths()
        return {
            "V": self.n_vertices, "E": self.n_edges, "F": self.n_faces,
            "euler": self.euler(),
            "three_regular": self.is_three_regular(),
            "face_types": self.face_type_counts(),
            "planar": planar,
            "every_edge_in_two_faces": all(v == 2 for v in inc.values())
                                       if self.closed else None,
            "edge_len_uniform": bool(np.ptp(L) < 1e-9 * L.mean()),
            "edge_len_mean": float(L.mean()),
        }


# --------------------------------------------------------------------------- #
#  Грани из выпуклой оболочки (объединение компланарных треугольников)
# --------------------------------------------------------------------------- #
def _faces_from_hull(coords: np.ndarray) -> list[list[int]]:
    r"""Грани выпуклого многогранника: группируем симплексы оболочки по
    внешней нормали, каждую грань упорядочиваем CCW вокруг внешней нормали."""
    hull = ConvexHull(coords)
    groups: dict[tuple, set] = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq[:3], 5))          # внешняя нормаль (единичная)
        groups.setdefault(key, set()).update(int(v) for v in simplex)

    faces = []
    for normal_key, verts in groups.items():
        normal = np.array(normal_key)
        faces.append(_order_face_ccw(list(verts), coords, normal))
    return faces


def _order_face_ccw(vidx: list[int], coords: np.ndarray,
                    normal: np.ndarray) -> list[int]:
    r"""Упорядочить вершины грани против часовой стрелки вокруг +normal.

    В плоском базисе (u, w=normal×u) угол растёт CCW, и u×w = normal, поэтому
    правая нормаль полученного цикла = +normal (наружу)."""
    pts = coords[vidx]
    c = pts.mean(axis=0)
    u = pts[0] - c
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)
    ang = np.array([np.arctan2((p - c) @ w, (p - c) @ u) for p in pts])
    order = np.argsort(ang)
    return [vidx[k] for k in order]


# --------------------------------------------------------------------------- #
#  Додекаэдр
# --------------------------------------------------------------------------- #
def _dodecahedron_coords() -> np.ndarray:
    r"""20 канонических вершин додекаэдра (золотое сечение), edge = 2/φ."""
    a = 1.0 / _PHI
    V = [np.array(p, dtype=float) for p in product((-1.0, 1.0), repeat=3)]  # куб (8)
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            V.append(np.array([0.0, s1 * a, s2 * _PHI]))
            V.append(np.array([s1 * a, s2 * _PHI, 0.0]))
            V.append(np.array([s1 * _PHI, 0.0, s2 * a]))
    return np.array(V)


def _edges_from_faces(faces: list[list[int]]) -> list[tuple[int, int]]:
    r"""Рёбра как последовательные пары вершин в циклах граней (без дублей)."""
    es = set()
    for f in faces:
        for k in range(len(f)):
            i, j = f[k], f[(k + 1) % len(f)]
            es.add((min(i, j), max(i, j)))
    return sorted(es)


def dodecahedron(edge: float = EDGE_NM) -> Lattice:
    r"""Замкнутая клетка-додекаэдр (n=20, Iₕ) с длиной ребра `edge` (нм).

    Идеальное вложение — равновесие по симметрии (нулевых сил нет), поэтому
    годится для валидации H0 без релаксатора (веха M2)."""
    coords = _dodecahedron_coords() * (edge / _CANON_EDGE)
    faces = _faces_from_hull(coords)
    edges = _edges_from_faces(faces)
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    G.add_edges_from(edges)
    return Lattice(coords=coords, edges=edges, faces=faces, graph=G,
                   name="dodecahedron", closed=True)
