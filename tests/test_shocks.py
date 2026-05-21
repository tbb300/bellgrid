import math

import pytest
import torch

from bellgrid.shocks import Lognormal, Normal


def _moment(nodes: torch.Tensor, weights: torch.Tensor, k: int) -> float:
    return torch.sum(weights * nodes**k).item()


# --- Normal shock ---------------------------------------------------------


def test_normal_default_is_standard():
    n = Normal()
    assert n.name is None
    assert n.sigma == 1.0


def test_normal_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        Normal(sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        Normal(sigma=-1.0)


def test_normal_weights_sum_to_one():
    for n_quad in (3, 5, 7, 11):
        nodes, weights = Normal().nodes_and_weights(n_quad)
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_normal_first_moment_zero_by_symmetry():
    # E[X] = 0 for X ~ N(0, 1), exact by symmetry at any n_quad
    for n_quad in (3, 5, 7):
        nodes, weights = Normal().nodes_and_weights(n_quad)
        assert _moment(nodes, weights, 1) == pytest.approx(0.0, abs=1e-12)


def test_normal_second_moment_matches_variance():
    # E[X^2] = sigma^2 for X ~ N(0, sigma^2). 3-point GH is exact for poly deg 5.
    for sigma in (1.0, 0.18, 2.5):
        nodes, weights = Normal(sigma=sigma).nodes_and_weights(3)
        assert _moment(nodes, weights, 2) == pytest.approx(sigma**2, abs=1e-12)


def test_normal_fourth_moment_matches_3sigma4():
    # E[X^4] = 3 * sigma^4 for X ~ N(0, sigma^2). Need n_quad >= 3 (deg 4 <= 2n-1).
    sigma = 1.5
    nodes, weights = Normal(sigma=sigma).nodes_and_weights(3)
    assert _moment(nodes, weights, 4) == pytest.approx(3.0 * sigma**4, abs=1e-10)


def test_normal_odd_moments_zero():
    sigma = 0.7
    nodes, weights = Normal(sigma=sigma).nodes_and_weights(5)
    for k in (1, 3, 5):
        assert _moment(nodes, weights, k) == pytest.approx(0.0, abs=1e-12)


def test_normal_nodes_scale_with_sigma():
    n0, w0 = Normal(sigma=1.0).nodes_and_weights(5)
    n1, w1 = Normal(sigma=3.0).nodes_and_weights(5)
    assert torch.allclose(n1, 3.0 * n0)
    assert torch.allclose(w1, w0)  # weights are invariant to sigma scaling


def test_normal_dtype_and_device():
    nodes, weights = Normal().nodes_and_weights(5, dtype=torch.float32)
    assert nodes.dtype == torch.float32
    assert weights.dtype == torch.float32
    assert nodes.device.type == "cpu"


def test_normal_n_quad_below_one_raises():
    with pytest.raises(ValueError, match="n_quad must be >= 1"):
        Normal().nodes_and_weights(0)


def test_normal_is_frozen():
    n = Normal(name="x")
    with pytest.raises(AttributeError):
        n.sigma = 2.0


def test_normal_sample_shape_and_dtype():
    s = Normal(sigma=0.5).sample(100, dtype=torch.float32)
    assert s.shape == (100,)
    assert s.dtype == torch.float32


def test_normal_sample_seed_reproducibility():
    n = Normal(sigma=2.0)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = n.sample(50, generator=g1)
    b = n.sample(50, generator=g2)
    assert torch.allclose(a, b)


def test_normal_sample_mean_and_std():
    """Many samples have mean ≈ 0 and std ≈ sigma."""
    n = Normal(sigma=1.5)
    s = n.sample(50_000, generator=torch.Generator().manual_seed(0))
    assert abs(s.mean().item()) < 0.05
    assert abs(s.std().item() - 1.5) < 0.05


# --- Lognormal shock ----------------------------------------------------


def test_lognormal_defaults():
    ln = Lognormal()
    assert ln.name is None
    assert ln.mu == 0.0
    assert ln.sigma == 1.0


def test_lognormal_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        Lognormal(sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        Lognormal(sigma=-0.5)


def test_lognormal_weights_sum_to_one():
    for n_quad in (3, 5, 7, 11):
        _, weights = Lognormal(mu=0.3, sigma=0.4).nodes_and_weights(n_quad)
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_lognormal_mean_matches_closed_form():
    """E[X] = exp(mu + sigma^2 / 2) for X ~ Lognormal(mu, sigma)."""
    for mu, sigma, n_quad in [(0.0, 0.2, 5), (0.05, 0.3, 7), (-0.1, 0.5, 11)]:
        nodes, weights = Lognormal(mu=mu, sigma=sigma).nodes_and_weights(n_quad)
        em = _moment(nodes, weights, 1)
        expected = math.exp(mu + 0.5 * sigma**2)
        assert em == pytest.approx(expected, rel=1e-6)


def test_lognormal_second_moment_matches_closed_form():
    """E[X^2] = exp(2*mu + 2*sigma^2)."""
    for mu, sigma, n_quad in [(0.0, 0.2, 7), (0.05, 0.3, 11)]:
        nodes, weights = Lognormal(mu=mu, sigma=sigma).nodes_and_weights(n_quad)
        em = _moment(nodes, weights, 2)
        expected = math.exp(2 * mu + 2 * sigma**2)
        assert em == pytest.approx(expected, rel=1e-5)


def test_lognormal_n_quad_below_one_raises():
    with pytest.raises(ValueError, match="n_quad must be >= 1"):
        Lognormal().nodes_and_weights(0)


def test_lognormal_is_frozen():
    ln = Lognormal(name="x")
    with pytest.raises(AttributeError):
        ln.sigma = 2.0


def test_lognormal_sample_mean_matches_closed_form():
    """Sample mean ≈ exp(mu + sigma^2/2) over many draws."""
    mu, sigma = 0.05, 0.3
    ln = Lognormal(mu=mu, sigma=sigma)
    s = ln.sample(50_000, generator=torch.Generator().manual_seed(1))
    expected = math.exp(mu + 0.5 * sigma**2)
    assert abs(s.mean().item() - expected) < 0.02
    assert (s > 0).all()  # lognormal is positive


def test_lognormal_sample_shape_and_dtype():
    s = Lognormal(mu=0.0, sigma=0.5).sample(50, dtype=torch.float32)
    assert s.shape == (50,)
    assert s.dtype == torch.float32
