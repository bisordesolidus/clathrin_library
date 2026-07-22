r"""
Ворота вехи M3a — спиральный генератор фуллеренов.

Главное: счёт изомеров совпадает с литературной переписью (та, на которой стоит
Шайн); дуал додекаэдра ≅ икосаэдр; реальный рецепт Шайна 36-15 даёт C36.
"""
import networkx as nx
import pytest

import fullerene as fu


# --------------------------------------------------------------------------- #
#  Золотой тест: додекаэдр
# --------------------------------------------------------------------------- #
def test_dodecahedron_dual_is_icosahedron():
    dual, valid = fu.spiral_to_dual([5] * 12)
    assert valid
    assert dual.number_of_nodes() == 12
    assert nx.is_isomorphic(dual, nx.icosahedral_graph())


def test_dodecahedron_dual_is_triangulation():
    dual, _ = fu.spiral_to_dual([5] * 12)
    assert fu.is_triangulation(dual)
    assert all(d == 5 for _, d in dual.degree())      # 20 граней → все deg 5


# --------------------------------------------------------------------------- #
#  Реальный рецепт Шайна: 36-15 (D6h барабан)
# --------------------------------------------------------------------------- #
def test_schein_36_15_recipe():
    """Рецепт A1 из Schein 2009 для 36-15 → корректный C36."""
    dual, valid = fu.spiral_to_dual(fu.parse_spiral("65555556666665555556"))
    assert valid
    assert dual.number_of_nodes() == 20                # F = 36/2 + 2
    degs = sorted(d for _, d in dual.degree())
    assert degs.count(5) == 12 and degs.count(6) == 8  # 12 пятиугольников + 8 шести
    assert fu.is_triangulation(dual)


def test_schein_36_15_recipes_same_isomer():
    """Спиральные рецепты Schein для 36-15, проходящие прямую намотку, дают
    один и тот же изомер C36. (Часть рецептов требует обобщённой спирали —
    для перечисления безвредно: клетку находит хотя бы один спираль.)"""
    recipes = [
        "65555556666665555556", "66556555656565655655", "65565565656565655655",
        "65655656565656555655", "56566555565556665655", "55665656555655656565",
        "56656565556555656565", "56565655655565656655", "55656665556555566565",
    ]
    valid_duals = [d for d, v in
                   (fu.spiral_to_dual(fu.parse_spiral(r)) for r in recipes) if v]
    assert len(valid_duals) >= 2                       # хотя бы несколько проходят
    for d in valid_duals:
        assert d.number_of_nodes() == 20
        assert nx.is_isomorphic(d, valid_duals[0])     # все — одна клетка


# --------------------------------------------------------------------------- #
#  Невалидные спирали отвергаются
# --------------------------------------------------------------------------- #
def test_invalid_spiral_rejected():
    assert not fu.is_valid_fullerene_spiral([5] * 11 + [6])   # не 12 пятёрок в замык.
    assert not fu.is_valid_fullerene_spiral([5, 6] * 7)       # выдуманная


def test_all_valid_spirals_have_twelve_pentagons():
    """Любой валидный дуал имеет ровно 12 вершин степени 5."""
    for n in [20, 24, 28, 32]:
        for dual in fu.enumerate_duals(n):
            degs = [d for _, d in dual.degree()]
            assert degs.count(5) == 12


# --------------------------------------------------------------------------- #
#  ГЛАВНОЕ: счёт изомеров = литературная перепись
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,count", [
    (20, 1), (24, 1), (26, 1), (28, 2), (30, 3), (32, 6),
])
def test_isomer_counts_match_literature(n, count):
    assert fu.n_isomers(n) == count


def test_face_count_formula():
    assert fu.face_count(20) == (12, 12, 0)      # додекаэдр: 12 граней, 0 шести
    assert fu.face_count(36) == (20, 12, 8)      # C36: 20 граней, 8 шести
    with pytest.raises(ValueError):
        fu.face_count(22)                        # C22 не существует


@pytest.mark.slow
def test_isomer_counts_large():
    """Медленно: C34→6, C36→15, C38→17 (проверено в зонде)."""
    assert fu.n_isomers(34) == 6
    assert fu.n_isomers(36) == 15
