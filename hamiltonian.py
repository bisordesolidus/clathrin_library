r"""
hamiltonian.py — сборка гессиана упругой сети (MODEL.md §5).

Ядро — DOF-агностичный сборщик rank-1 вкладов:
        H = Σ_c  k_c · J_c J_cᵀ,
где J_c — «обобщённый вектор связи», разложенный по глобальным степеням свободы
контакта c. Это ровно форма из MODEL.md §5.2; при этом:
  * точечный ANM (веха M1): на пару (i,j) с ортом ребра n̂ вектор
        J = [n̂ ; −n̂]  над 6 трансл. DOF узлов i,j  ⇒  блоки ±k n̂n̂ᵀ;
  * rigid-body SE(3) (веха M2): на контакт — 12-вектор J_c над 6+6 DOF узлов
        (тот же `assemble`, длина J другая).

Каждый вклад rank-1 ⇒ H ⪰ 0 автоматически. Разрежённая сборка (scipy.sparse).

Точечный ANM (baseline). Анизотропная сеть (Atilgan 2001), гармонизованная
вокруг равновесия r⁰:
        E = ½ Σ_{|r⁰_i−r⁰_j|≤R_c} k [ n̂_ij · (δr_i − δr_j) ]²,
        n̂_ij = (r⁰_i − r⁰_j)/|r⁰_i − r⁰_j|.
Отсечка R_c обязательна: только на рёбрах сеть недоопределена по Максвеллу
(для додекаэдра 30 нулевых мод); включение диагоналей граней (2-я оболочка)
делает её жёсткой → ровно 6 нулевых мод.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

# Отсечка по умолчанию как множитель длины ребра: 2.0·a включает рёбра
# (1.0·a) и диагонали граней (φ·a ≈ 1.618·a), отсекает 3-ю оболочку (≈2.29·a).
DEFAULT_CUTOFF_FACTOR = 2.0


# --------------------------------------------------------------------------- #
#  DOF-агностичный сборщик  H = Σ k J Jᵀ
# --------------------------------------------------------------------------- #
def assemble(n_dof: int,
             terms: Iterable[tuple[np.ndarray, np.ndarray, float]]) -> csr_matrix:
    r"""Собрать разрежённый гессиан H = Σ k·J Jᵀ.

    terms — итерируемое из (idx, J, k):
        idx : (m,) целые глобальные индексы DOF, задействованных контактом;
        J   : (m,) обобщённый вектор связи на этих DOF;
        k   : жёсткость (скаляр).
    Каждый вклад добавляет k·outer(J,J) в подматрицу idx×idx."""
    rows, cols, data = [], [], []
    for idx, J, k in terms:
        idx = np.asarray(idx, dtype=np.intp)
        J = np.asarray(J, dtype=float)
        m = idx.size
        rows.append(np.repeat(idx, m))
        cols.append(np.tile(idx, m))
        data.append((k * np.outer(J, J)).ravel())
    if not rows:
        return csr_matrix((n_dof, n_dof))
    H = coo_matrix((np.concatenate(data),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_dof, n_dof)).tocsr()
    return 0.5 * (H + H.T)      # симметризация против ошибок округления


# --------------------------------------------------------------------------- #
#  Точечный ANM
# --------------------------------------------------------------------------- #
def anm_pairs(coords: np.ndarray, cutoff: float) -> np.ndarray:
    r"""Все пары узлов i<j на расстоянии ≤ cutoff. Возвращает (npair, 2)."""
    tree = cKDTree(np.asarray(coords, dtype=float))
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    return pairs.reshape(-1, 2)


def point_node_terms(
        coords: np.ndarray,
        pairs: np.ndarray,
        stiffness: float | Callable[[int, int, float], float] = 1.0,
) -> Iterator[tuple[list[int], np.ndarray, float]]:
    r"""Вклады точечного ANM: J = [n̂; −n̂] на 3-блоках DOF узлов i,j."""
    coords = np.asarray(coords, dtype=float)
    for i, j in pairs:
        i, j = int(i), int(j)
        v = coords[i] - coords[j]
        L = float(np.linalg.norm(v))
        n = v / L
        idx = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        J = np.concatenate([n, -n])
        k = stiffness(i, j, L) if callable(stiffness) else float(stiffness)
        yield idx, J, k


def point_node_hessian(
        coords: np.ndarray,
        cutoff: float,
        stiffness: float | Callable[[int, int, float], float] = 1.0,
) -> csr_matrix:
    r"""Гессиан точечного ANM 3N×3N с отсечкой (стандартный ANM)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    pairs = anm_pairs(coords, cutoff)
    return assemble(3 * n, point_node_terms(coords, pairs, stiffness))


def anm_from_lattice(latt, cutoff_factor: float = DEFAULT_CUTOFF_FACTOR,
                     stiffness=1.0) -> csr_matrix:
    r"""Точечный ANM для решётки: отсечка = cutoff_factor · (средняя длина ребра)."""
    a = float(latt.edge_lengths().mean())
    return point_node_hessian(latt.coords, cutoff_factor * a, stiffness)


def anm_full_hessian(coords: np.ndarray, pairs: np.ndarray,
                     rest: np.ndarray, stiffness: float = 1.0) -> np.ndarray:
    r"""ПОЛНЫЙ гессиан ANM с ФИКСИРОВАННЫМИ длинами покоя `rest` (плотный 3N×3N).

    E = ½ Σ k(|r_ij| − d₀)². Блок пары:
        k [ n̂n̂ᵀ + (L−d₀)/L (I − n̂n̂ᵀ) ]   (второй член — геометрическая жёсткость).
    В равновесии (L=d₀) = проектированный `point_node_hessian`; ВНЕ равновесия
    член (L−d₀) ненулевой — нужен для ангармонической поправки ТВ (§8.2)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    for idx, (i, j) in enumerate(pairs):
        i, j = int(i), int(j)
        r = coords[i] - coords[j]
        L = float(np.linalg.norm(r))
        nh = r / L
        B = stiffness * (np.outer(nh, nh)
                         + (L - rest[idx]) / L * (np.eye(3) - np.outer(nh, nh)))
        for a, b, s in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
            H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += s * B
    return H
