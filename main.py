r"""
main.py — полный пайплайн клатриновой оболочки (MODEL.md §16).

Одна команда строит для КАЖДОЙ клетки рисунки cage / spectrum / soft_mode /
buckling_mode + report.md, и общую зонную диаграмму объёмов.

    python main.py                       # все клетки Шайна n<=50 + зонная диаграмма
    python main.py --cage barrel-36      # одна именованная клетка
    python main.py --n 20 24 28 32 36    # выбранные C_n (все разрешённые Шайном)
    python main.py --n 60                # включить C60 (ДОЛГО, ~2ч; для кластера)
    python main.py --bands-only          # только зонная диаграмма из кэша

Клетки — все РАЗРЕШЁННЫЕ ПРАВИЛОМ ШАЙНА изомеры каждого C_n (классификатор
`schein`). Дуалы кэшируются в runs/cache/ (наращиваются по n, не пересчитываются).
Ядро без графики; здесь — оркестрация и сохранение.
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import bandstructure as bs
import diagonalize as dg
import embed as em
import fullerene as fu
import hamiltonian as ham
import schein as sc
import stability as st
import visualize as viz
import volume as vol
from lattice import Lattice

# все n с разрешёнными Шайном клетками при n<=60 (C60 — по явному запросу)
SCHEIN_N_DEFAULT = [20, 24, 26, 28, 32, 36, 38, 40, 42, 44, 50, 60]
CACHE = "runs/cache"

NAMED = {
    "dodeca-20": lambda: em.fullerene_lattice([5] * 12, name="dodeca-20"),
    "barrel-36": lambda: em.fullerene_lattice(em.BARREL_36_15, name="barrel-36"),
    "minicoat-28": lambda: _rename(em.find_cage(28, {1, 2, 3}), "minicoat-28"),
}


def _rename(latt, name):
    latt.name = name
    return latt


# --------------------------------------------------------------------------- #
#  Сбор клеток (разрешённых Шайном), с кэшем дуалов
# --------------------------------------------------------------------------- #
def _cache_file(n: int) -> str:
    return f"{CACHE}/duals_C{n}.pkl"


def collect_duals(n: int) -> list:
    r"""Разрешённые Шайном дуалы C_n (из кэша или параллельным перечислением)."""
    os.makedirs(CACHE, exist_ok=True)
    path = _cache_file(n)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from enumerate_par import schein_allowed_par
    duals = schein_allowed_par(n)
    data = [nx.node_link_data(d) for d in duals]
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"C{n}: {len(duals)} разрешённых Шайном → кэш")
    return data


def lattice_from_dual(dual_data, name: str) -> Lattice:
    from embed import _dualize, spectral_embed
    dual = nx.node_link_graph(dual_data)
    coords, edges, faces = _dualize(dual, spectral_embed(dual), 18.5)
    G = nx.Graph(); G.add_nodes_from(range(len(coords))); G.add_edges_from(edges)
    return Lattice(coords=coords, edges=edges, faces=faces, graph=G,
                   name=name, closed=True)


def cages_for(n_values) -> list[Lattice]:
    r"""Все разрешённые Шайном клетки для набора n (объекты Lattice)."""
    cages = []
    for n in n_values:
        for idx, dd in enumerate(collect_duals(n)):
            cages.append(lattice_from_dual(dd, f"C{n}-{idx}"))
    return cages


# --------------------------------------------------------------------------- #
#  Полный набор рисунков + отчёт для одной клетки
# --------------------------------------------------------------------------- #
def analyze_cage(L: Lattice, out: str, schein_allowed=None) -> dict:
    r"""cage / spectrum / soft_mode / buckling_mode + report.md для клетки L."""
    os.makedirs(out, exist_ok=True)

    H = ham.anm_from_lattice(L)
    w, V = dg.spectrum(H)
    nz = dg.n_zero_modes(w)
    mults = dg.multiplicities(w)
    order = np.argsort(w)

    V0 = vol.enclosed_volume(L.coords, L.faces)
    chi = bs.volume_susceptibility(L)
    sigma_c, v_c = st.critical_tension(L)

    # рисунки (каждая клетка — все четыре)
    for fig, fn in [
        (viz.plot_cage(L), "cage.png"),
        (viz.plot_spectrum(w, n_zero=nz), "spectrum.png"),
        (viz.plot_mode(L, V[:, order[nz]]), "soft_mode.png"),
    ]:
        fig.savefig(f"{out}/{fn}", dpi=130, bbox_inches="tight")
        plt.close(fig)
    if v_c is not None:
        fig = viz.plot_mode(L, v_c)
        fig.savefig(f"{out}/buckling_mode.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(f"{out}/report.md", "w", encoding="utf-8") as fh:
        fh.write(f"""# {L.name}

- хабов N = {L.n_vertices}, рёбер = {L.n_edges}, граней = {L.face_type_counts()}
- Эйлер V−E+F = {L.euler()}
- разрешена правилом Шайна: {schein_allowed}

## Спектр H₀ (точечный ANM)
- нулевых мод (rigid-body): {nz}
- кратности вырождений (точечная группа): {sorted(set(mults))}
- мягчайшая ненулевая λ = {w[order[nz]]:.4f}

## Натяжение (Helfrich σ·A)
- объём V₀ = {V0:.1f} нм³
- восприимчивость χ_VA = {chi:.1f}   (V(σ) = V₀ − σ·χ_VA)
- критическое натяжение σ_c = {sigma_c:.4g}

## Рисунки
cage.png · spectrum.png · soft_mode.png{' · buckling_mode.png' if v_c is not None else ''}
""")
    print(f"[{L.name}] N={L.n_vertices} σ_c={sigma_c:.4g} → {out}/")
    return {"name": L.name, "N": L.n_vertices, "V0": V0, "chi": chi,
            "sigma_c": sigma_c}


# --------------------------------------------------------------------------- #
#  Зонная диаграмма по набору клеток
# --------------------------------------------------------------------------- #
def build_bands(cages, out: str, sigma_max: float):
    os.makedirs(out, exist_ok=True)
    cages = sorted(cages, key=lambda L: vol.enclosed_volume(L.coords, L.faces))
    bstr = bs.band_structure(cages, sigma_max=sigma_max)
    fig = viz.plot_volume_bands(bstr)
    fig.set_size_inches(10, max(6, 0.7 * len(cages)))
    fig.savefig(f"{out}/volume_bands.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    lines = [f"# Зонная структура объёмов — {len(cages)} клеток Шайна\n",
             f"σ ∈ [0, {sigma_max}] пН/нм\n",
             "| клетка | N | V₀ | зона [lo, hi] |", "|---|---|---|---|"]
    for b in bstr["bands"]:
        lines.append(f'| {b["name"]} | {b["N"]} | {b["V0"]:.0f} '
                     f'| [{b["lo"]:.0f}, {b["hi"]:.0f}] |')
    lines.append("\n## Запрещённые зоны / перекрытия")
    for f in bstr["forbidden"]:
        tag = "ПЕРЕКРЫВАЮТСЯ (кроссовер)" if f["overlap"] else \
              f'запрещено [{f["lo"]:.0f}, {f["hi"]:.0f}] ширина {f["width"]:.0f}'
        lines.append(f'- {f["below"]}↔{f["above"]}: {tag}')
    with open(f"{out}/bands_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    ngap = sum(1 for f in bstr["forbidden"] if not f["overlap"])
    print(f"[зоны] {len(cages)} клеток → {out}/volume_bands.png "
          f"({ngap} запрещённых зон, {len(bstr['forbidden'])-ngap} перекрытий)")


# --------------------------------------------------------------------------- #
#  Оркестрация
# --------------------------------------------------------------------------- #
def run_all(n_values, out: str, sigma_max: float):
    r"""Полный прогон: все разрешённые Шайном клетки + рисунки каждой + зоны."""
    cages = cages_for(n_values)
    for L in cages:
        analyze_cage(L, f"{out}/{L.name}", schein_allowed=True)
    build_bands(cages, out, sigma_max)


def run_named(spec: str, out: str, sigma_max: float):
    if spec in NAMED:
        L = NAMED[spec]()
    elif spec.startswith("C") and "-" in spec:
        n, k = spec[1:].split("-")
        L = lattice_from_dual(collect_duals(int(n))[int(k)], spec)
    else:
        raise SystemExit(f"неизвестная клетка: {spec} "
                         f"(доступно: {', '.join(NAMED)}, C<n>-<k>)")
    analyze_cage(L, out, schein_allowed=None)


def main():
    p = argparse.ArgumentParser(description="Пайплайн клатриновой оболочки")
    p.add_argument("--cage", help="одна клетка: dodeca-20 | barrel-36 | "
                                   "minicoat-28 | C<n>-<k>")
    p.add_argument("--n", type=int, nargs="+", default=None,
                   help="какие C_n (все разрешённые Шайном); дефолт n<=50")
    p.add_argument("--bands-only", action="store_true",
                   help="только зонная диаграмма из кэша")
    p.add_argument("--sigma-max", type=float, default=0.3)
    p.add_argument("--out", default="runs", help="папка вывода")
    args = p.parse_args()

    if args.cage:
        run_named(args.cage, f"{args.out}/{args.cage}", args.sigma_max)
    elif args.bands_only:
        n_vals = args.n or [n for n in SCHEIN_N_DEFAULT
                            if os.path.exists(_cache_file(n))]
        build_bands(cages_for(n_vals), args.out, args.sigma_max)
    else:
        run_all(args.n or SCHEIN_N_DEFAULT, args.out, args.sigma_max)


if __name__ == "__main__":
    main()
