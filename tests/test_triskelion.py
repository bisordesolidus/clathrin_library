r"""
Ворота вехи M0 для triskelion.py.

Ключевое: C₃-симметрия хаба переставляет ноги; угол пакера ψ задаёт наклон;
закрутка χ не меняет направление ноги, но несёт хиральность (n̂_a·φ̂_a = sinχ).
"""
import numpy as np
import pytest

import triskelion as tk
from triskelion import Triskelion, E1, E2, E3

RNG = np.random.default_rng(20260715)


def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
#  Реперы ног — валидные элементы SO(3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [0, 1, 2])
def test_leg_rotation_is_SO3(a):
    T = Triskelion()
    R = T.leg_rotation(a)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-13)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-13)
    # leg_frame — валидный SE(3) с нулевой трансляцией
    F = T.leg_frame(a)
    assert np.allclose(F[:3, 3], 0.0)
    assert np.allclose(F[3, :], [0, 0, 0, 1])


# --------------------------------------------------------------------------- #
#  Угол пакера
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("psi_deg", [90.0, 97.0, 110.0])
def test_pucker_angle(psi_deg):
    T = Triskelion(psi=np.deg2rad(psi_deg))
    for a in range(3):
        u = T.leg_axis(a)
        assert np.isclose(u @ E3, np.cos(np.deg2rad(psi_deg)), atol=1e-13)
        assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-13)


def test_pucker_direction_toward_membrane():
    """ψ>90° ⇒ ноги ниже экватора (−ê₃, к мембране)."""
    T = Triskelion(psi=np.deg2rad(97.0))
    for a in range(3):
        assert T.leg_axis(a) @ E3 < 0.0


def test_leg_axis_matches_closed_form():
    T = Triskelion(psi=np.deg2rad(97.0), chi=np.deg2rad(41.0))
    psi = T.psi
    for a in range(3):
        phi = 2 * np.pi * a / 3
        expect = np.array([np.sin(psi) * np.cos(phi),
                           np.sin(psi) * np.sin(phi),
                           np.cos(psi)])
        assert np.allclose(T.leg_axis(a), expect, atol=1e-13)


# --------------------------------------------------------------------------- #
#  C₃-симметрия — центральные ворота M0
# --------------------------------------------------------------------------- #
def test_C3_permutes_legs():
    """Поворот хаба на 2π/3 вокруг ê₃ переводит ногу a → a+1."""
    T = Triskelion()
    Rc = _Rz(2 * np.pi / 3)
    for a in range(3):
        assert np.allclose(Rc @ T.leg_rotation(a),
                           T.leg_rotation((a + 1) % 3), atol=1e-13)


def test_C3_axes_120_apart():
    """Оси ног попарно под одним углом (C₃); проекции на 120° по азимуту."""
    T = Triskelion(psi=np.deg2rad(97.0))
    psi = T.psi
    expect_dot = np.sin(psi) ** 2 * np.cos(2 * np.pi / 3) + np.cos(psi) ** 2
    for a in range(3):
        b = (a + 1) % 3
        assert np.isclose(T.leg_axis(a) @ T.leg_axis(b), expect_dot, atol=1e-13)
    # азимутальные проекции разнесены ровно на 120°
    az = [np.arctan2(T.leg_axis(a)[1], T.leg_axis(a)[0]) for a in range(3)]
    d = np.diff(np.unwrap(az))
    assert np.allclose(d, 2 * np.pi / 3, atol=1e-12)


# --------------------------------------------------------------------------- #
#  χ: чистая закрутка (не трогает направление) + хиральность
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [0, 1, 2])
def test_leg_axis_independent_of_chi(a):
    u1 = Triskelion(chi=np.deg2rad(10.0)).leg_axis(a)
    u2 = Triskelion(chi=np.deg2rad(80.0)).leg_axis(a)
    assert np.allclose(u1, u2, atol=1e-13)


@pytest.mark.parametrize("a", [0, 1, 2])
def test_binding_normal_perp_axis(a):
    T = Triskelion()
    assert abs(T.binding_normal(a) @ T.leg_axis(a)) < 1e-13


@pytest.mark.parametrize("chi_deg", [10.0, 30.0, 75.0, -45.0])
def test_chirality_invariant(chi_deg):
    """n̂_a · φ̂_a = sinχ для всех ног; знак = хиральность."""
    T = Triskelion(chi=np.deg2rad(chi_deg))
    for a in range(3):
        phi = 2 * np.pi * a / 3
        phi_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])  # азимутальный орт
        assert np.isclose(T.binding_normal(a) @ phi_hat,
                          np.sin(np.deg2rad(chi_deg)), atol=1e-13)


def test_mirror_flips_chirality():
    """Энантиомер (χ→−χ) обращает знак хирального инварианта."""
    T = Triskelion(chi=np.deg2rad(30.0))
    M = T.mirror()
    for a in range(3):
        phi = 2 * np.pi * a / 3
        phi_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])
        assert np.isclose(T.binding_normal(a) @ phi_hat,
                          -(M.binding_normal(a) @ phi_hat), atol=1e-13)


# --------------------------------------------------------------------------- #
#  Дуговая длина s(остаток)
# --------------------------------------------------------------------------- #
def test_arclength_calibration():
    T = Triskelion()
    # проксимальный сегмент = 11.4 нм (измерено из 6SCT)
    assert np.isclose(T.arclength(1576), 0.0, atol=1e-12)     # у хаба
    assert np.isclose(T.arclength(1198), 11.4, atol=1e-12)    # конец proximal (6SCT)
    # полная нога ≈ 50 нм у кончика TD
    assert np.isclose(T.arclength(1), 50.0, atol=1e-9)


def test_arclength_monotone_decreasing_residue():
    """s растёт при движении к кончику (уменьшении номера остатка)."""
    T = Triskelion()
    residues = [1500, 1300, 1100, 1000, 900, 700, 400, 100]
    s = [T.arclength(r) for r in residues]
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1))


def test_arclength_continuous_at_boundaries():
    """На границе сегментов s непрерывна (общий остаток даёт одно s)."""
    T = Triskelion()
    for res in [1198, 1074, 838, 330]:
        # запросим чуть выше и чуть ниже границы — значения близки
        assert np.isclose(T.arclength(res + 1e-6), T.arclength(res - 1e-6),
                          atol=1e-4)


def test_arclength_rejects_hub_and_out_of_range():
    T = Triskelion()
    with pytest.raises(ValueError):
        T.arclength(1600)      # TxD/хаб
    with pytest.raises(ValueError):
        T.arclength(0)         # вне диапазона


# --------------------------------------------------------------------------- #
#  Точки на ноге
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [0, 1, 2])
def test_site_on_centerline_is_s_times_axis(a):
    """На осевой линии ρ = s·û_a (χ не входит)."""
    T = Triskelion()
    s = 12.3
    assert np.allclose(T.site_body(a, s, (0.0, 0.0)), s * T.leg_axis(a),
                       atol=1e-13)


def test_transverse_offset_lies_along_binding_normal():
    """Поперечный сдвиг вдоль ê₁ ложится вдоль связывающей нормали n̂_a(χ)."""
    T = Triskelion()
    a, c = 1, 4.0
    off = T.site_body(a, 0.0, (c, 0.0))
    assert np.allclose(off, c * T.binding_normal(a), atol=1e-13)


def test_contact_sites_have_valid_arclength():
    """Все контактные сайты Morris попадают на distal/proximal с корректным s."""
    T = Triskelion()
    for name, meta in tk.CONTACT_SITES.items():
        p = T.contact_site_body(0, name)
        assert np.all(np.isfinite(p))
        s = T.arclength(0.5 * sum(meta["residues"]))
        assert 0.0 <= s <= 50.0
