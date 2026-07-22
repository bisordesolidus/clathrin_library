r"""
fullerene.py — генерация топологии фуллереновых клеток (MODEL.md §4, §11 T10).

Спиральный алгоритм Фаулера–Манолопулоса. Строим ДУАЛ фуллерена — триангуляцию
сферы, где грань фуллерена (пятиугольник/шестиугольник) становится вершиной со
степенью 5/6. Сам фуллерен получается геометрической дуализацией дуала (вложение
+ выпуклая оболочка) — это модуль вложения (веха M3b).

Спираль = последовательность F размеров граней (5 или 6), ровно 12 пятёрок.
Намотка: вершину k присоединяем к предыдущей (back) и к «переднему» пробегу
границы, закрывая вершины, достигшие целевой степени. Не всякая последовательность
даёт замкнутый фуллерен (см. `valid`).

Фуллерен C_n: n вершин, F = n/2+2 граней = 12 пятиугольников + (n−20)/2
шестиугольников; дуал — триангуляция с F вершинами, E = 3F−6 рёбер.

Валидация (тесты): дуал додекаэдра ≅ икосаэдр; рецепты Шайна; счёт изомеров
совпадает с литературной переписью (C20:1 … C36:15 … C40:40).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import networkx as nx


def spiral_to_dual(spiral: list[int]) -> tuple[nx.Graph, bool]:
    r"""Спираль (список 5/6) → (граф дуала, valid).

    valid=True ⟺ намотка замкнулась и все вершины достигли целевой степени."""
    F = len(spiral)
    target = list(spiral)
    deg = [0] * F
    G = nx.Graph()
    G.add_nodes_from(range(F))

    def connect(a: int, b: int) -> None:
        if not G.has_edge(a, b):
            G.add_edge(a, b)
            deg[a] += 1
            deg[b] += 1

    if F == 0:
        return G, False
    boundary = [0]
    if F > 1:
        connect(0, 1)
        boundary = [0, 1]

    for k in range(2, F):
        if not boundary:                       # закрылось досрочно → невалидно
            return G, False
        connect(k, boundary[-1])               # к предыдущей (back)
        if deg[boundary[-1]] == target[boundary[-1]]:
            boundary.pop()
        while boundary:                        # передний пробег
            f = boundary[0]
            connect(k, f)
            if deg[f] == target[f]:
                boundary.pop(0)
            else:
                break
            if deg[k] == target[k]:            # k сам закрылся (последняя вершина)
                break
        if deg[k] < target[k]:
            boundary.append(k)

    valid = (len(boundary) == 0) and all(deg[i] == target[i] for i in range(F))
    return G, valid


def is_valid_fullerene_spiral(spiral: list[int]) -> bool:
    return spiral_to_dual(spiral)[1]


def parse_spiral(s: str) -> list[int]:
    r"""'65555556666665555556' → [6,5,5,...]."""
    return [int(c) for c in s]


def face_count(n: int) -> tuple[int, int, int]:
    r"""Для C_n: (F граней, 12 пятиугольников, h шестиугольников)."""
    if n < 20 or n % 2 or n == 22:
        raise ValueError(f"нет фуллерена C_{n}")
    F = n // 2 + 2
    return F, 12, F - 12


@lru_cache(maxsize=None)
def enumerate_duals(n: int) -> list[nx.Graph]:
    r"""Все различные (с точностью до изоморфизма) дуалы фуллеренов C_n.

    Перебор размещений 12 пятёрок среди F граней; дедупликация по хешу
    Вейсфейлера–Лемана + точная проверка изоморфизма внутри корзины.
    Кэшируется: каждое n перечисляется один раз за сессию."""
    F, _, h = face_count(n)
    buckets: dict[str, list[nx.Graph]] = {}
    reps: list[nx.Graph] = []
    for hexpos in combinations(range(F), h):
        S = set(hexpos)
        spiral = [6 if i in S else 5 for i in range(F)]
        dual, valid = spiral_to_dual(spiral)
        if not valid:
            continue
        key = nx.weisfeiler_lehman_graph_hash(dual, iterations=3)
        bucket = buckets.setdefault(key, [])
        if not any(nx.is_isomorphic(dual, g) for g in bucket):
            bucket.append(dual)
            reps.append(dual)
    return reps


def n_isomers(n: int) -> int:
    return len(enumerate_duals(n))


def is_triangulation(dual: nx.Graph) -> bool:
    r"""Дуал валидного фуллерена — триангуляция сферы: планарен, E = 3V−6."""
    if dual.number_of_edges() != 3 * dual.number_of_nodes() - 6:
        return False
    return nx.check_planarity(dual)[0]
