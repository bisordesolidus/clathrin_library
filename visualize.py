r"""
visualize.py — визуализация результатов ядра (MODEL.md §16.2).

Чистые функции «объект → matplotlib Figure». Ядро ничего не знает о графике;
здесь берём его выходы и рисуем. Бэкенд Agg (headless, PNG для галереи `main.py`).

Рисунки: 3D-клетка, спектр мод, векторное поле моды, скан по натяжению,
зонная диаграмма объёмов.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                       # headless
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_cage(latt, ax=None, node_color="#2c6fbb", edge_color="#9db8d2",
              face_pentagon="#e8974a", face_hexagon="#cfe0f0"):
    r"""3D-каркас клетки: хабы (узлы) + ноги (рёбра), грани раскрашены пента/гекса."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    C = latt.coords
    polys, cols = [], []
    for f in latt.faces:
        polys.append(C[f])
        cols.append(face_pentagon if len(f) == 5 else face_hexagon)
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=edge_color,
                          linewidths=1.0, alpha=0.55)
    ax.add_collection3d(pc)
    ax.scatter(*C.T, c=node_color, s=28, depthshade=True)
    _equal_3d(ax, C)
    ax.set_title(f"{latt.name}: {latt.n_vertices} хабов, "
                 f"{latt.face_type_counts()}")
    ax.set_axis_off()
    return fig if fig is not None else ax.figure


def plot_mode(latt, mode, ax=None, scale=None, color="#d1495b"):
    r"""Векторное поле смещений выбранной моды на клетке (стрелки).
    mode — вектор 3N (точечный) или 6N (SE(3): берём трансляционные блоки)."""
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    C = latt.coords
    N = latt.n_vertices
    m = np.asarray(mode)
    disp = m.reshape(N, -1)[:, :3]              # трансляционная часть
    if scale is None:
        span = np.ptp(C, axis=0).max()
        dmax = np.linalg.norm(disp, axis=1).max() + 1e-30
        scale = 0.25 * span / dmax
    segs = [[C[i], C[i] + scale * disp[i]] for i in range(N)]
    ax.add_collection3d(Line3DCollection(segs, colors="#888", linewidths=2))
    tips = C + scale * disp
    ax.scatter(*C.T, c="#bbb", s=10)
    ax.scatter(*tips.T, c=color, s=22)
    _equal_3d(ax, np.vstack([C, tips]))
    ax.set_title(f"{latt.name}: мода смещений")
    ax.set_axis_off()
    return fig if fig is not None else ax.figure


def plot_spectrum(w, n_zero=6, ax=None, highlight=6):
    r"""Спектр собственных значений: нулевые моды отмечены, мягкие выделены."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    w = np.sort(np.asarray(w))
    idx = np.arange(len(w))
    ax.axhspan(-1e-9, 1e-9, color="#eee")
    ax.scatter(idx[:n_zero], w[:n_zero], c="#aaa", s=20, label=f"{n_zero} нулевых (rigid)")
    soft = slice(n_zero, n_zero + highlight)
    ax.scatter(idx[soft], w[soft], c="#d1495b", s=32, label=f"{highlight} мягких")
    ax.scatter(idx[n_zero + highlight:], w[n_zero + highlight:], c="#2c6fbb", s=14)
    ax.set_xlabel("индекс моды"); ax.set_ylabel("λ (жёсткость моды)")
    ax.set_title("Спектр H₀"); ax.legend(fontsize=8)
    return fig if fig is not None else ax.figure


def plot_tension_scan(sigmas, lambda_curves, sigma_c=None, ax=None):
    r"""Сдвиг низших мод λ_n(σ) под натяжением; отмечено критическое σ_c."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    sigmas = np.asarray(sigmas)
    for row in np.atleast_2d(lambda_curves):
        ax.plot(sigmas, row, lw=1.5)
    if sigma_c is not None and np.isfinite(sigma_c):
        ax.axvline(sigma_c, color="#d1495b", ls="--",
                   label=f"σ_c = {sigma_c:.3g}")
        ax.legend(fontsize=8)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("натяжение σ"); ax.set_ylabel("λ (жёсткость моды)")
    ax.set_title("Сдвиг мод под натяжением")
    return fig if fig is not None else ax.figure


def plot_volume_bands(band_struct, ax=None):
    r"""Зонная диаграмма объёмов: зоны клеток + запрещённые промежутки (§15.5)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    bands = band_struct["bands"]
    for i, b in enumerate(bands):
        ax.barh(i, b["hi"] - b["lo"], left=b["lo"], height=0.6,
                color="#2c6fbb", alpha=0.8)
        ax.plot(b["V0"], i, "o", color="#e8974a", ms=6)          # V₀ (σ=0)
        ax.text(b["hi"], i, f'  {b["name"]} (N={b["N"]})', va="center", fontsize=8)
    for f in band_struct["forbidden"]:
        if not f["overlap"]:
            ax.axvspan(f["lo"], f["hi"], color="#d1495b", alpha=0.12)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([b["name"] for b in bands])
    ax.set_xlabel("объём V (нм³)")
    ax.set_title("Зонная структура объёмов (красное = запрещённые зоны)")
    return fig if fig is not None else ax.figure


def _equal_3d(ax, pts):
    r"""Равные масштабы осей 3D вокруг облака точек."""
    c = pts.mean(axis=0)
    r = np.ptp(pts, axis=0).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
