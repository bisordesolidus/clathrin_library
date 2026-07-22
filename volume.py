r"""
volume.py — геометрические операторы оболочки: площадь и объём (MODEL.md §7.1, §15).

Площадь A(r) — для мембранного натяжения (Helfrich σ·A). Объём V(r) — для
зонной структуры объёмов (§15). Оба зависят ТОЛЬКО от положений хабов r_i
(грани из `Lattice`, ориентированы наружу).

Объём (теорема о дивергенции, веерная триангуляция каждой грани вокруг центроида):
        V = (1/6) Σ_грани Σ_i  c_f · (v_i × v_{i+1}),   c_f = центроид грани.
Площадь:
        A = Σ_грани Σ_i  ½ |(v_i − c_f) × (v_{i+1} − c_f)|.

Градиенты — численные (операторы дешёвые); для больших клеток можно заменить на
аналитические.
"""

from __future__ import annotations

import numpy as np


def surface_area(coords: np.ndarray, faces: list[list[int]]) -> float:
    r"""Площадь поверхности оболочки A(r) = Σ площадей граней (веер из центроида)."""
    coords = np.asarray(coords, dtype=float)
    A = 0.0
    for f in faces:
        pts = coords[f]
        c = pts.mean(axis=0)
        k = len(f)
        for i in range(k):
            A += 0.5 * np.linalg.norm(np.cross(pts[i] - c, pts[(i + 1) % k] - c))
    return float(A)


def enclosed_volume(coords: np.ndarray, faces: list[list[int]]) -> float:
    r"""Охваченный объём V(r) (теорема о дивергенции). Грани — наружу CCW."""
    coords = np.asarray(coords, dtype=float)
    V = 0.0
    for f in faces:
        pts = coords[f]
        c = pts.mean(axis=0)
        k = len(f)
        for i in range(k):
            V += c @ np.cross(pts[i], pts[(i + 1) % k])
    return float(V) / 6.0


def _numeric_gradient(func, coords: np.ndarray, faces, eps: float = 1e-6) -> np.ndarray:
    r"""Численный градиент скалярного оператора по всем 3N координатам (вектор 3N)."""
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    flat = coords.ravel().copy()
    g = np.zeros(n)
    for a in range(n):
        fp = flat.copy(); fp[a] += eps
        fm = flat.copy(); fm[a] -= eps
        g[a] = (func(fp.reshape(coords.shape), faces)
                - func(fm.reshape(coords.shape), faces)) / (2 * eps)
    return g


def area_gradient(coords, faces) -> np.ndarray:
    r"""∇A — АНАЛИТИЧЕСКИЙ градиент площади по положениям хабов (3N).
    Градиент площади треугольника (c,p,q): ∂/∂p = ½ n̂×(c−q) и т.д.; центроид c
    зависит от всех вершин грани (∂c/∂v = 1/k)."""
    coords = np.asarray(coords, dtype=float)
    g = np.zeros_like(coords)
    for f in faces:
        pts = coords[f]
        k = len(f)
        c = pts.mean(axis=0)
        for i in range(k):
            p, q = pts[i], pts[(i + 1) % k]
            nvec = np.cross(p - c, q - c)
            Ln = np.linalg.norm(nvec)
            if Ln < 1e-15:
                continue
            nh = nvec / Ln
            g[f[i]] += 0.5 * np.cross(nh, c - q)          # ∂/∂p
            g[f[(i + 1) % k]] += 0.5 * np.cross(nh, p - c)  # ∂/∂q
            gc = 0.5 * np.cross(nh, q - p)                  # ∂/∂c
            for v in f:
                g[v] += gc / k
    return g.ravel()


def volume_gradient(coords, faces, eps: float = 1e-6) -> np.ndarray:
    r"""∇V — градиент объёма по положениям хабов (3N)."""
    return _numeric_gradient(enclosed_volume, coords, faces, eps)


def area_hessian(coords, faces, eps: float = 1e-4) -> np.ndarray:
    r"""∇²A — гессиан площади (3N×3N), конечная разность градиента."""
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    flat = coords.ravel().copy()
    H = np.zeros((n, n))
    for a in range(n):
        fp = flat.copy(); fp[a] += eps
        fm = flat.copy(); fm[a] -= eps
        H[:, a] = (area_gradient(fp.reshape(coords.shape), faces)
                   - area_gradient(fm.reshape(coords.shape), faces)) / (2 * eps)
    return 0.5 * (H + H.T)
