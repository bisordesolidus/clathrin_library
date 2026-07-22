r"""
Тесты классификатора Шайна — правило запрета «голова-хвост».

Главное: счёт разрешённых изомеров совпадает с таблицей Шайна / литературой;
разделение чистое (worst-ht по клетке строго 0 или 2, никогда 1).
"""
import pytest

import fullerene as fu
import schein as sc


def test_dihedral_ordering():
    """Двугранный угол убывает с числом пятиугольников в вершине."""
    d = sc.DIHEDRAL
    assert d[(6, 6, 6)] > d[(5, 6, 6)] > d[(5, 5, 6)] > d[(5, 5, 5)]


def test_dodecahedron_allowed():
    """Додекаэдр (20-1) разрешён; голова-хвост шагов нет."""
    dual, _ = fu.spiral_to_dual([5] * 12)
    assert sc.is_schein_allowed(dual)
    assert max(sc.head_to_tail_counts(dual)) == 0


def test_barrel_36_15_allowed():
    """Барабан 36-15 (реальный рецепт Шайна) разрешён."""
    dual, _ = fu.spiral_to_dual(fu.parse_spiral("65555556666665555556"))
    assert sc.is_schein_allowed(dual)


# --------------------------------------------------------------------------- #
#  ГЛАВНОЕ: счёт разрешённых = таблица Шайна
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,count", [(28, 1), (30, 0), (32, 1)])
def test_allowed_counts(n, count):
    a = sum(1 for d in fu.enumerate_duals(n) if sc.is_schein_allowed(d))
    assert a == count


def test_clean_separation():
    """worst-ht по клетке строго 0 (разрешена) или 2 (исключена), никогда 1."""
    for d in fu.enumerate_duals(32):
        assert max(sc.head_to_tail_counts(d)) in (0, 2)


@pytest.mark.slow
@pytest.mark.parametrize("n,count", [(34, 0), (36, 2), (38, 1), (40, 2)])
def test_allowed_counts_large(n, count):
    """Медленно (перечисление больших клеток): C34:0, C36:2, C38:1, C40:2."""
    a = sum(1 for d in fu.enumerate_duals(n) if sc.is_schein_allowed(d))
    assert a == count
