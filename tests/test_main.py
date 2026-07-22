r"""
Ворота M8 — пайплайн и визуализация (smoke-tests: рисунки строятся, файлы пишутся).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import embed as em
import hamiltonian as ham
import diagonalize as dg
import lattice as lat
import bandstructure as bs
import visualize as viz
import main


@pytest.fixture(scope="module")
def dodeca():
    return lat.dodecahedron()


# --------------------------------------------------------------------------- #
#  visualize.py — фигуры строятся без ошибок
# --------------------------------------------------------------------------- #
def test_plot_cage(dodeca):
    import matplotlib.pyplot as plt
    fig = viz.plot_cage(dodeca)
    assert fig is not None
    plt.close(fig)


def test_plot_spectrum_and_mode(dodeca):
    H = ham.anm_from_lattice(dodeca)
    w, V = dg.spectrum(H)
    import matplotlib.pyplot as plt
    f1 = viz.plot_spectrum(w); plt.close(f1)
    f2 = viz.plot_mode(dodeca, V[:, dg.zero_mode_indices(w)[0] + 6]); plt.close(f2)


def test_plot_volume_bands():
    cages = [em.fullerene_lattice([5] * 12, name="C20"), em.isomer_lattice(24, 0)]
    bstr = bs.band_structure(cages, sigma_max=0.1)
    import matplotlib.pyplot as plt
    fig = viz.plot_volume_bands(bstr); plt.close(fig)


# --------------------------------------------------------------------------- #
#  main.py — end-to-end: одна клетка пишет 4 рисунка + отчёт
# --------------------------------------------------------------------------- #
def test_analyze_cage_writes_all_figures(tmp_path):
    import os
    L = lat.dodecahedron(); L.name = "dodeca-20"
    main.analyze_cage(L, str(tmp_path / "d"), schein_allowed=True)
    for fn in ["report.md", "cage.png", "spectrum.png", "soft_mode.png",
               "buckling_mode.png"]:
        assert os.path.exists(os.path.join(str(tmp_path / "d"), fn))
    with open(os.path.join(str(tmp_path / "d"), "report.md"), encoding="utf-8") as fh:
        assert "нулевых мод (rigid-body): 6" in fh.read()


def test_run_named_variants(tmp_path):
    import os
    main.run_named("dodeca-20", str(tmp_path / "n"), sigma_max=0.1)
    assert os.path.exists(os.path.join(str(tmp_path / "n"), "cage.png"))
    with pytest.raises(SystemExit):
        main.run_named("nonsense", str(tmp_path / "x"), sigma_max=0.1)


def test_build_bands_writes_outputs(tmp_path):
    import os
    cages = [em.fullerene_lattice([5] * 12, name="C20-0"), em.isomer_lattice(24, 0)]
    main.build_bands(cages, str(tmp_path / "b"), sigma_max=0.1)
    assert os.path.exists(os.path.join(str(tmp_path / "b"), "volume_bands.png"))
    assert os.path.exists(os.path.join(str(tmp_path / "b"), "bands_report.md"))


def test_lattice_from_dual():
    import fullerene as fu, networkx as nx
    dual, _ = fu.spiral_to_dual([5] * 12)
    L = main.lattice_from_dual(nx.node_link_data(dual), "C20-0")
    assert L.n_vertices == 20
