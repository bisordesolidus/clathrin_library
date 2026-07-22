r"""
Ворота вехи M0 для se3.py.

Все ключевые тождества проверяются против независимых оракулов:
  * scipy.linalg.expm            — матричная экспонента (для exp_se3, Ad=exp(ad));
  * scipy.spatial.transform.Rotation — эталон log_so3.
"""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import se3

RNG = np.random.default_rng(20260715)


def random_xi(rot_scale=1.0, trans_scale=2.0):
    u = RNG.normal(size=3) * trans_scale
    phi = RNG.normal(size=3)
    phi = phi / np.linalg.norm(phi) * RNG.uniform(0.0, rot_scale)
    return np.concatenate([u, phi])


def random_T(rot_scale=np.pi - 0.2):
    return se3.exp_se3(random_xi(rot_scale=rot_scale))


# --------------------------------------------------------------------------- #
#  hat / vee
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(20))
def test_hat_vee_so3_roundtrip(seed):
    w = np.random.default_rng(seed).normal(size=3)
    assert np.allclose(se3.vee_so3(se3.hat_so3(w)), w, atol=1e-14)


def test_hat_so3_is_cross_product():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    assert np.allclose(se3.hat_so3(a) @ b, np.cross(a, b), atol=1e-14)


@pytest.mark.parametrize("seed", range(20))
def test_hat_vee_se3_roundtrip(seed):
    xi = random_xi()
    assert np.allclose(se3.vee_se3(se3.hat_se3(xi)), xi, atol=1e-14)


# --------------------------------------------------------------------------- #
#  exp / log — обратность (главные ворота, <1e-12)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(50))
def test_log_exp_se3_roundtrip(seed):
    """log(exp(ξ)) = ξ при |φ| < π (вне зоны неоднозначности)."""
    xi = random_xi(rot_scale=np.pi - 0.1)
    assert np.allclose(se3.log_se3(se3.exp_se3(xi)), xi, atol=1e-12)


@pytest.mark.parametrize("seed", range(50))
def test_exp_log_se3_roundtrip(seed):
    """exp(log(T)) = T для произвольного T (однозначно даже при θ→π)."""
    T = random_T(rot_scale=np.pi - 1e-3)
    assert np.allclose(se3.exp_se3(se3.log_se3(T)), T, atol=1e-12)


def test_exp_so3_matches_expm():
    for _ in range(20):
        w = random_xi()[3:]
        assert np.allclose(se3.exp_so3(w), expm(se3.hat_so3(w)), atol=1e-12)


def test_exp_se3_matches_expm():
    """exp_se3 совпадает с матричной экспонентой hat_se3."""
    for _ in range(20):
        xi = random_xi()
        assert np.allclose(se3.exp_se3(xi), expm(se3.hat_se3(xi)), atol=1e-11)


def test_log_so3_matches_scipy():
    for _ in range(30):
        R = Rotation.random(random_state=RNG).as_matrix()
        got = se3.log_so3(R)
        ref = Rotation.from_matrix(R).as_rotvec()
        # log_so3 однозначен всюду, кроме θ=π; сравниваем как повороты.
        assert np.allclose(se3.exp_so3(got), R, atol=1e-11)
        if np.linalg.norm(ref) < np.pi - 1e-2:
            assert np.allclose(got, ref, atol=1e-9)


# --------------------------------------------------------------------------- #
#  Валидность результата exp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(20))
def test_exp_se3_is_valid_group_element(seed):
    T = random_T()
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)      # ортогональность
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)     # SO(3), не O(3)
    assert np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-14)    # нижняя строка


# --------------------------------------------------------------------------- #
#  Малые углы: без 0/0, совпадение с expm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
def test_small_angle_stable(theta):
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    xi = np.concatenate([RNG.normal(size=3), axis * theta])
    T = se3.exp_se3(xi)
    assert np.all(np.isfinite(T))
    assert np.allclose(T, expm(se3.hat_se3(xi)), atol=1e-12)
    assert np.allclose(se3.log_se3(T), xi, atol=1e-11)


# --------------------------------------------------------------------------- #
#  Инверсия и композиция
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(20))
def test_inverse(seed):
    T = random_T()
    assert np.allclose(se3.compose(T, se3.inverse(T)), np.eye(4), atol=1e-12)
    assert np.allclose(se3.compose(se3.inverse(T), T), np.eye(4), atol=1e-12)


def test_compose_matches_matmul():
    A, B, C = random_T(), random_T(), random_T()
    assert np.allclose(se3.compose(A, B, C), A @ B @ C, atol=1e-12)


# --------------------------------------------------------------------------- #
#  Присоединённые представления — центральные тождества
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(30))
def test_Ad_exp_equals_exp_ad(seed):
    """Ad(exp ξ) = exp(ad ξ) — связь группового и алгебраического ad."""
    xi = random_xi(rot_scale=np.pi - 0.1)
    assert np.allclose(se3.Ad(se3.exp_se3(xi)), expm(se3.ad(xi)), atol=1e-10)


@pytest.mark.parametrize("seed", range(20))
def test_Ad_is_homomorphism(seed):
    """Ad(T1 T2) = Ad(T1) Ad(T2)."""
    T1, T2 = random_T(), random_T()
    assert np.allclose(se3.Ad(se3.compose(T1, T2)),
                       se3.Ad(T1) @ se3.Ad(T2), atol=1e-11)


@pytest.mark.parametrize("seed", range(20))
def test_Ad_conjugation_definition(seed):
    """Ad_T ξ = vee(T ξ̂ T⁻¹): присоединённое = сопряжение в группе."""
    T, xi = random_T(), random_xi()
    lhs = se3.Ad(T) @ xi
    rhs = se3.vee_se3(se3.compose(T, se3.hat_se3(xi), se3.inverse(T)))
    assert np.allclose(lhs, rhs, atol=1e-11)


@pytest.mark.parametrize("seed", range(20))
def test_ad_is_lie_bracket(seed):
    """ad_ξ η = vee([ξ̂, η̂]) — определение алгебраического ad через скобку Ли."""
    xi, eta = random_xi(), random_xi()
    X, Y = se3.hat_se3(xi), se3.hat_se3(eta)
    lhs = se3.ad(xi) @ eta
    rhs = se3.vee_se3(X @ Y - Y @ X)
    assert np.allclose(lhs, rhs, atol=1e-11)


# --------------------------------------------------------------------------- #
#  log_so3 вблизи θ = π (обе ветки: регулярная и околопи)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta", [1e-9, 1e-7, 1e-4, 0.0])
def test_log_so3_near_pi(delta):
    theta = np.pi - delta
    for _ in range(20):
        a = RNG.normal(size=3)
        a /= np.linalg.norm(a)
        R = se3.exp_so3(theta * a)
        got = se3.log_so3(R)
        # exp(log(R)) = R выполняется всегда (инвариант, не зависит от знака оси)
        assert np.allclose(se3.exp_so3(got), R, atol=1e-7)
        assert np.isclose(np.linalg.norm(got), theta, atol=1e-6)
        # при θ<π знак оси определён — должен совпасть с истинным +a
        if delta > 0:
            assert np.allclose(got, theta * a, atol=1e-3)
