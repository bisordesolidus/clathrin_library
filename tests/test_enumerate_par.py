r"""
Тест параллельного перечислителя разрешённых Шайном клеток.

Малые n идут через fallback (без процессов) — сверяем с полным перебором + Шайном.
Параллельный путь (большие n) проверяется в интеграции build_bands.
"""
import fullerene as fu
import schein as sc
from enumerate_par import schein_allowed_par, _n_combos


def test_small_n_matches_full():
    """C28, C32: параллельный (fallback) счёт = полный перебор + Шайн."""
    for n in (28, 32):
        par = len(schein_allowed_par(n, n_proc=1))
        full = sum(1 for d in fu.enumerate_duals(n) if sc.is_schein_allowed(d))
        assert par == full


def test_counts_match_schein_table():
    """Разрешённых при малых n — как в таблице Шайна."""
    assert len(schein_allowed_par(20, n_proc=1)) == 1
    assert len(schein_allowed_par(28, n_proc=1)) == 1
    assert len(schein_allowed_par(32, n_proc=1)) == 1


def test_n_combos_formula():
    assert _n_combos(20) == 1              # C(12,0)
    assert _n_combos(36) == 125970         # C(20,12)
    assert _n_combos(44) == 2704156        # C(24,12)
