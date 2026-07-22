r"""
Ворота вехи M5 (машинерия ТВ) — валидация против точной диагонализации.

T6: ошибка ТВ 1-го порядка = O(ε²), наклон лог-лог = 2.
T7: ошибка ТВ 2-го порядка = O(ε³), наклон = 3.
T8: симметричное возмущение не расщепляет вырожденные мультиплеты (вырожд. ТВ).
"""
import numpy as np
import pytest

import diagonalize as dg
import hamiltonian as ham
import lattice as lat
import perturbation as pt

RNG = np.random.default_rng(20260717)


def _sym(n):
    A = RNG.normal(size=(n, n))
    return 0.5 * (A + A.T)


def _loglog_slope(eps, err):
    m = err > 1e-14                       # отбрасываем машинный ноль
    return np.polyfit(np.log(eps[m]), np.log(err[m]), 1)[0]


# --------------------------------------------------------------------------- #
#  T6, T7 — наклоны на невырожденной случайной матрице (чистая математика)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def random_case():
    n = 24
    H0 = _sym(n) + 10 * np.eye(n)          # общие (невырожденные) с.з.
    w0, V0 = np.linalg.eigh(H0)
    Hp = _sym(n)
    return H0, Hp, w0, V0


def test_pt_first_order_slope_two(random_case):
    H0, Hp, w0, V0 = random_case
    eps = np.logspace(-4, -1.5, 8)
    err = []
    for e in eps:
        exact = pt.exact_eigenvalues(H0, Hp, e)
        approx = np.sort(pt.pt_eigenvalues(w0, V0, Hp, e, order=1))
        err.append(np.abs(exact - approx).max())
    slope = _loglog_slope(eps, np.array(err))
    assert 1.9 < slope < 2.1


def test_pt_second_order_slope_three(random_case):
    H0, Hp, w0, V0 = random_case
    eps = np.logspace(-4, -1.5, 8)
    err = []
    for e in eps:
        exact = pt.exact_eigenvalues(H0, Hp, e)
        approx = np.sort(pt.pt_eigenvalues(w0, V0, Hp, e, order=2))
        err.append(np.abs(exact - approx).max())
    slope = _loglog_slope(eps, np.array(err))
    assert 2.8 < slope < 3.2


def test_second_order_better_than_first(random_case):
    H0, Hp, w0, V0 = random_case
    e = 1e-2
    exact = pt.exact_eigenvalues(H0, Hp, e)
    e1 = np.abs(exact - np.sort(pt.pt_eigenvalues(w0, V0, Hp, e, order=1))).max()
    e2 = np.abs(exact - np.sort(pt.pt_eigenvalues(w0, V0, Hp, e, order=2))).max()
    assert e2 < e1


# --------------------------------------------------------------------------- #
#  Вырожденная ТВ: расщепление совпадает с точным
# --------------------------------------------------------------------------- #
def test_degenerate_pt_matches_exact():
    """H₀ с вырожденной парой + возмущение: λ⁽¹⁾ = расщепление (точно при ε→0)."""
    n = 10
    d = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # пара 2,2
    Q, _ = np.linalg.qr(_sym(n))
    H0 = Q @ np.diag(d) @ Q.T
    w0, V0 = np.linalg.eigh(H0)
    Hp = _sym(n)
    eps = np.logspace(-4, -2, 6)
    err = []
    for e in eps:
        exact = pt.exact_eigenvalues(H0, Hp, e)
        approx = np.sort(pt.pt_eigenvalues(w0, V0, Hp, e, order=1))
        err.append(np.abs(exact - approx).max())
    assert 1.8 < _loglog_slope(eps, np.array(err)) < 2.2   # вырожд. ТВ корректна


# --------------------------------------------------------------------------- #
#  T8 — симметричное возмущение не расщепляет мультиплеты
# --------------------------------------------------------------------------- #
def test_symmetric_perturbation_does_not_split():
    """На додекаэдре (I_h): возмущение ∝ H₀ (полностью симметрично) сдвигает
    мультиплеты целиком, λ⁽¹⁾ одинаковы внутри вырожденной группы."""
    H0 = ham.anm_from_lattice(lat.dodecahedron()).toarray()
    w0, V0 = np.linalg.eigh(H0)
    active = np.where(np.abs(w0) > 1e-6 * abs(w0).max())[0]
    Hp = H0.copy()                          # ∝ H₀: сохраняет всю симметрию
    lam1, _, _ = pt.pt_corrections(w0, V0, Hp, active=active)
    # внутри каждой вырожденной группы λ⁽¹⁾ совпадают
    for _, grp in _groups_by_value(w0, active):
        vals = lam1[grp]
        assert np.ptp(vals) < 1e-6 * (abs(vals).max() + 1)


def _groups_by_value(w, active, rtol=1e-6):
    scale = max(abs(w).max(), 1.0)
    order = np.array(active)[np.argsort(w[active])]
    out = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and w[order[j + 1]] - w[order[i]] < rtol * scale:
            j += 1
        out.append((w[order[i]], list(order[i:j + 1])))
        i = j + 1
    return out


def test_tension_correct_beats_naive():
    """§8.2: правильный H⁽¹⁾ натяжения (с ангармоникой C[H₀⁺∇A]) на порядки
    точнее наивного (∇²A) — линейный член сдвигает равновесие и входит в спектр."""
    import volume as vol
    d = lat.dodecahedron()
    coords0, fcs = d.coords, d.faces
    a = d.edge_lengths().mean()
    pairs = ham.anm_pairs(coords0, 2.0 * a)
    rest = np.array([np.linalg.norm(coords0[i] - coords0[j]) for i, j in pairs])
    H0 = ham.anm_full_hessian(coords0, pairs, rest)
    w0, V0 = np.linalg.eigh(H0)
    active = w0 > 1e-6 * w0.max()
    H1c, H1n = pt.tension_h1(coords0, fcs, pairs, rest)
    l1c = np.diag(V0.T @ H1c @ V0)
    l1n = np.diag(V0.T @ H1n @ V0)

    H0p = pt._pinv_drop_null(H0)
    def anm_grad(C):
        g = np.zeros_like(C)
        for idx, (i, j) in enumerate(pairs):
            i, j = int(i), int(j); r = C[i] - C[j]; L = np.linalg.norm(r)
            f = (L - rest[idx]) * r / L; g[i] += f; g[j] -= f
        return g
    sigma = 1e-4
    C = coords0.copy()
    for _ in range(300):
        g = anm_grad(C) + sigma * vol.area_gradient(C, fcs).reshape(C.shape)
        C = C - (H0p @ g.ravel()).reshape(C.shape)
        if np.linalg.norm(g) < 1e-11:
            break
    Hex = ham.anm_full_hessian(C, pairs, rest) + sigma * vol.area_hessian(C, fcs)
    wex = np.sort(np.linalg.eigvalsh(Hex)[active])
    err_c = np.abs(wex - np.sort((w0 + sigma * l1c)[active])).max()
    err_n = np.abs(wex - np.sort((w0 + sigma * l1n)[active])).max()
    assert err_c < err_n / 50                # правильная ТВ радикально точнее


def test_pt_on_cage_hessian_slope_two():
    """ТВ на реальном гессиане клетки (6 нулевых мод исключены) — наклон 2."""
    H0 = ham.anm_from_lattice(lat.dodecahedron()).toarray()
    w0, V0 = np.linalg.eigh(H0)
    active = np.where(np.abs(w0) > 1e-6 * abs(w0).max())[0]
    Hp = _sym(len(w0))
    eps = np.logspace(-5, -3, 6)
    err = []
    for e in eps:
        lam1, _, _ = pt.pt_corrections(w0, V0, Hp, active=active)
        approx = w0[active] + e * lam1[active]
        exact_all = pt.exact_eigenvalues(H0, Hp, e)
        # сопоставляем активные (внутренние) моды по порядку
        exact = exact_all[active]
        err.append(np.abs(np.sort(exact) - np.sort(approx)).max())
    assert 1.8 < _loglog_slope(eps, np.array(err)) < 2.2
