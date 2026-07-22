r"""
enumerate_par.py — параллельный перебор разрешённых Шайном клеток (для больших n).

Полный однопоточный перебор C(F,12) размещений деградирует при n≳42. Здесь —
multiprocessing: воркеры обрабатывают чанки размещений 12 пятиугольников,
возвращают ДЁШЕВО-СЕРИАЛИЗУЕМЫЕ спиральные строки разрешённых Шайном клеток
(не графы); главный процесс строит дуалы и дедуплицирует горстку разрешённых.

Использование:
    from enumerate_par import schein_allowed_par
    duals = schein_allowed_par(50)          # список networkx-дуалов
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

import fullerene as fu
import schein as sc


def _worker(args):
    r"""Обработать все размещения, где ПЕРВЫЙ пятиугольник в позиции p0 (без промотки:
    остальные 11 пятиугольников — в позициях p0+1..F-1). Хвост пентагонов ≥ первого."""
    n, p0 = args
    F = n // 2 + 2
    out = []
    for rest in combinations(range(p0 + 1, F), 11):
        S = {p0, *rest}
        spiral = [6 if i not in S else 5 for i in range(F)]
        dual, valid = fu.spiral_to_dual(spiral)
        if valid and sc.is_schein_allowed(dual):
            out.append(tuple(spiral))
    return out


def _n_combos(n: int) -> int:
    from math import comb
    F = n // 2 + 2
    return comb(F, F - 12)


def schein_allowed_par(n: int, n_proc: int | None = None) -> list[nx.Graph]:
    r"""Все разрешённые Шайном дуалы C_n (параллельно). Дедупликация по изоморфизму.

    Разбиение по позиции первого пятиугольника p0∈{0..F-12} — естественные чанки
    без промотки combinations (критично для больших n)."""
    import os
    from multiprocessing import Pool
    F = n // 2 + 2
    if _n_combos(n) <= 200_000 or (n_proc is not None and n_proc <= 1):
        return [d for d in fu.enumerate_duals(n) if sc.is_schein_allowed(d)]

    tasks = [(n, p0) for p0 in range(F - 11)]     # первый пентагон: 0..F-12
    n_proc = n_proc or max(1, (os.cpu_count() or 2) - 1)
    spirals: list[tuple] = []
    with Pool(n_proc) as pool:
        for part in pool.imap_unordered(_worker, tasks):
            spirals.extend(part)

    # построить дуалы и дедуплицировать (разрешённых мало)
    reps: list[nx.Graph] = []
    buckets: dict[str, list[nx.Graph]] = {}
    for sp in spirals:
        dual, _ = fu.spiral_to_dual(list(sp))
        key = nx.weisfeiler_lehman_graph_hash(dual, iterations=3)
        b = buckets.setdefault(key, [])
        if not any(nx.is_isomorphic(dual, g) for g in b):
            b.append(dual)
            reps.append(dual)
    return reps
