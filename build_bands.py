r"""
build_bands.py — полная зонная диаграмма объёмов по ВСЕМ 15 клеткам Шайна (§15, §16.4).

15 разрешённых Шайном изомеров при n≤60: 20-1, 24-1, 26-1, 28-2, 32-6, 36-14,
36-15, 38-17, 40-38, 40-39, 42-45, 44-75, 44-89, 50-271, 60-1784.

Параллельное перечисление (`enumerate_par`) для больших n. Дуалы кэшируются в
runs/bands_all/duals_C<n>.pkl — можно наращивать по n, не пересчитывая.

    python build_bands.py --n 20 24 26 28 32 36 38 40 42 44   # 13 клеток
    python build_bands.py --n 50                              # добавить C50
    python build_bands.py --n 60                              # добавить C60 (долго)
    python build_bands.py --plot                              # собрать диаграмму из кэша
"""

from __future__ import annotations

import argparse
import os
import pickle

import networkx as nx

import bandstructure as bs
import schein as sc
import visualize as viz
import volume as vol
from embed import _dualize, spectral_embed
from lattice import Lattice

OUT = "D:/clathrin_library/runs/bands_all"
ALL_N = [20, 24, 26, 28, 32, 36, 38, 40, 42, 44, 50, 60]


def _cache_path(n: int) -> str:
    return f"{OUT}/duals_C{n}.pkl"


def collect_n(n: int) -> list:
    r"""Разрешённые Шайном дуалы C_n (из кэша или параллельным перечислением)."""
    os.makedirs(OUT, exist_ok=True)
    path = _cache_path(n)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from enumerate_par import schein_allowed_par
    duals = schein_allowed_par(n)
    with open(path, "wb") as fh:
        pickle.dump([nx.node_link_data(d) for d in duals], fh)
    print(f"C{n}: {len(duals)} разрешённых → кэш")
    return [nx.node_link_data(d) for d in duals]


def cage_from_dual(dual_data, n: int, idx: int) -> Lattice:
    dual = nx.node_link_graph(dual_data)
    coords, edges, faces = _dualize(dual, spectral_embed(dual), 18.5)
    G = nx.Graph(); G.add_nodes_from(range(len(coords))); G.add_edges_from(edges)
    return Lattice(coords=coords, edges=edges, faces=faces, graph=G,
                   name=f"C{n}-{idx}", closed=True)


def build_plot(n_values, sigma_max: float = 0.3):
    cages = []
    for n in n_values:
        if not os.path.exists(_cache_path(n)):
            print(f"C{n}: нет в кэше — сначала `python build_bands.py --n {n}`")
            continue
        with open(_cache_path(n), "rb") as fh:
            for idx, dd in enumerate(pickle.load(fh)):
                cages.append(cage_from_dual(dd, n, idx))
    cages.sort(key=lambda L: vol.enclosed_volume(L.coords, L.faces))
    bstr = bs.band_structure(cages, sigma_max=sigma_max)
    fig = viz.plot_volume_bands(bstr)
    fig.set_size_inches(10, max(6, 0.7 * len(cages)))
    fig.savefig(f"{OUT}/volume_bands_all.png", dpi=130, bbox_inches="tight")

    lines = [f"# Зонная структура объёмов — {len(cages)} клеток Шайна\n",
             f"σ ∈ [0, {sigma_max}] пН/нм\n",
             "| клетка | N | V₀ (нм³) | зона [lo, hi] |", "|---|---|---|---|"]
    for b in bstr["bands"]:
        lines.append(f'| {b["name"]} | {b["N"]} | {b["V0"]:.0f} '
                     f'| [{b["lo"]:.0f}, {b["hi"]:.0f}] |')
    lines.append("\n## Запрещённые зоны / перекрытия")
    for f in bstr["forbidden"]:
        if f["overlap"]:
            lines.append(f'- {f["below"]}↔{f["above"]}: ПЕРЕКРЫВАЮТСЯ (кроссовер)')
        else:
            lines.append(f'- {f["below"]}↔{f["above"]}: запрещено '
                         f'[{f["lo"]:.0f}, {f["hi"]:.0f}] ширина {f["width"]:.0f}')
    with open(f"{OUT}/bands_all_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    ngap = sum(1 for f in bstr["forbidden"] if not f["overlap"])
    print(f"{len(cages)} клеток → {OUT}/volume_bands_all.png "
          f"({ngap} запрещённых зон, {len(bstr['forbidden'])-ngap} перекрытий)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", help="какие C_n перечислить (в кэш)")
    p.add_argument("--plot", action="store_true", help="собрать диаграмму из кэша")
    p.add_argument("--sigma-max", type=float, default=0.3)
    args = p.parse_args()
    if args.n:
        for n in args.n:
            collect_n(n)
    if args.plot or not args.n:
        have = [n for n in ALL_N if os.path.exists(_cache_path(n))]
        build_plot(have, args.sigma_max)


if __name__ == "__main__":
    main()
