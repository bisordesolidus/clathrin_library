r"""
embed.py — геометрическая реализация фуллерена (веха M3b).

Из дуала (триангуляции, `fullerene.py`) строим объект `Lattice`:
  1. спектральное вложение дуала на сферу (собственные векторы лапласиана —
     для симметричных клеток низшие ненулевые моды дают координатные функции);
  2. выпуклая оболочка дуала: треугольники = ВЕРШИНЫ фуллерена, смежные
     треугольники (по ребру дуала) = РЁБРА, кольцо треугольников вокруг вершины
     дуала = ГРАНЬ фуллерена;
  3. грани строим КОМБИНАТОРНО из дуала (точно, не зависит от качества вложения).

Проверка (тесты): додекаэдр из спирали ≅ координатный додекаэдр M0; точечные
группы названных клеток читаются как вырождения точечного ANM-спектра
(I_h→{1,3,4,5}, D_6h барабан→{1,2}, T_d мини-кот→{1,2,3}).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

import diagonalize as dg
import fullerene as fu
import hamiltonian as ham
from lattice import Lattice
from triskelion import EDGE_NM

# Отсечка ANM для проверки групп: 1.9·(средняя длина ребра) — рёбра + 2-я оболочка.
ANM_CUTOFF_FACTOR = 1.9


def _tangent_frame(r_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.eye(3)[int(np.argmin(np.abs(r_hat)))]
    e1 = a - (a @ r_hat) * r_hat
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(r_hat, e1)


def spectral_embed(dual: nx.Graph) -> np.ndarray:
    r"""Вложение дуала на единичную сферу: 3 низшие ненулевые моды лапласиана."""
    N = dual.number_of_nodes()
    L = nx.laplacian_matrix(dual, nodelist=range(N)).toarray().astype(float)
    _, V = np.linalg.eigh(L)
    P = V[:, 1:4].copy()
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def _dualize(dual: nx.Graph, pts: np.ndarray, edge: float):
    r"""Оболочка дуала → (coords, edges, faces) фуллерена."""
    N = dual.number_of_nodes()
    n_expected = 2 * N - 4                       # вершин фуллерена
    hull = ConvexHull(pts)
    tris = [tuple(sorted(int(v) for v in s)) for s in hull.simplices]
    if len(tris) != n_expected:
        raise ValueError(f"невыпуклое вложение: {len(tris)} треуг. вместо {n_expected}")
    tri_index = {t: i for i, t in enumerate(tris)}

    coords = np.array([pts[list(t)].mean(0) for t in tris])
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)

    # рёбра: два треугольника, делящие ребро дуала
    edge_tris: dict[tuple, list[int]] = {}
    for t in tris:
        for a, b in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            edge_tris.setdefault((a, b), []).append(tri_index[t])
    edges = sorted(tuple(sorted(v)) for v in edge_tris.values() if len(v) == 2)

    # масштаб к длине ребра
    L0 = np.mean([np.linalg.norm(coords[i] - coords[j]) for i, j in edges])
    coords *= edge / L0

    # грани: вокруг каждой вершины дуала, упорядочены CCW наружу
    faces = []
    for f in range(N):
        inc = [tri_index[t] for t in tris if f in t]
        rf = pts[f] / np.linalg.norm(pts[f])
        e1, e2 = _tangent_frame(rf)
        ang = []
        for i in inc:
            tvec = coords[i] - (coords[i] @ rf) * rf
            ang.append(np.arctan2(tvec @ e2, tvec @ e1))
        faces.append([inc[k] for k in np.argsort(ang)])

    return coords, edges, faces


def fullerene_lattice(spiral: list[int], edge: float = EDGE_NM,
                      name: str = "fullerene") -> Lattice:
    r"""Спиральный рецепт → объект `Lattice` фуллерена."""
    dual, valid = fu.spiral_to_dual(spiral)
    if not valid:
        raise ValueError("невалидный спиральный рецепт")
    coords, edges, faces = _dualize(dual, spectral_embed(dual), edge)
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    G.add_edges_from(edges)
    return Lattice(coords=coords, edges=edges, faces=faces, graph=G,
                   name=name, closed=True)


def isomer_lattice(n: int, index: int = 0, edge: float = EDGE_NM) -> Lattice:
    r"""index-й изомер C_n как `Lattice` (порядок — как в enumerate_duals)."""
    dual = fu.enumerate_duals(n)[index]
    coords, edges, faces = _dualize(dual, spectral_embed(dual), edge)
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    G.add_edges_from(edges)
    return Lattice(coords=coords, edges=edges, faces=faces, graph=G,
                   name=f"C{n}-{index}", closed=True)


def anm_signature(latt: Lattice) -> tuple[int, list[int]]:
    r"""(число нулевых мод, отсортированные КРАТНОСТИ вырождений) точечного ANM —
    сигнатура точечной группы через размерности неприводимых представлений."""
    H = ham.anm_from_lattice(latt, cutoff_factor=ANM_CUTOFF_FACTOR)
    w, _ = dg.spectrum(H)
    return dg.n_zero_modes(w), sorted(set(dg.multiplicities(w)))


def find_cage(n: int, irrep_dims: set[int], edge: float = EDGE_NM) -> Lattice:
    r"""Найти изомер C_n, чья сигнатура групп = irrep_dims (напр. {1,3,4,5} для I_h)."""
    for k in range(fu.n_isomers(n)):
        latt = isomer_lattice(n, k, edge)
        if set(anm_signature(latt)[1]) == irrep_dims:
            return latt
    raise ValueError(f"нет изомера C_{n} с сигнатурой {irrep_dims}")


# Именованные клетки (спиральные рецепты; барабан — реальный рецепт Schein 2009).
DODECAHEDRON = [5] * 12
BARREL_36_15 = fu.parse_spiral("65555556666665555556")     # D6h
