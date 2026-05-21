import math

import pytest
import torch

from bellgrid.grids import RegularGrid, WarpedGrid


def test_point_count():
    g = RegularGrid(n=10)
    pts = g.points(0.0, 1.0)
    assert pts.shape == (10,)


def test_endpoints_included():
    g = RegularGrid(n=5)
    pts = g.points(-2.0, 3.0)
    assert pts[0].item() == pytest.approx(-2.0)
    assert pts[-1].item() == pytest.approx(3.0)


def test_uniform_spacing():
    g = RegularGrid(n=5)
    pts = g.points(0.0, 4.0)
    diffs = torch.diff(pts)
    assert torch.allclose(diffs, torch.full_like(diffs, 1.0))


def test_default_dtype_is_float64():
    g = RegularGrid(n=4)
    pts = g.points(0.0, 1.0)
    assert pts.dtype == torch.float64


def test_dtype_override():
    g = RegularGrid(n=4)
    pts = g.points(0.0, 1.0, dtype=torch.float32)
    assert pts.dtype == torch.float32


def test_device_override_cpu():
    g = RegularGrid(n=4)
    pts = g.points(0.0, 1.0, device="cpu")
    assert pts.device.type == "cpu"


def test_n_below_two_raises():
    with pytest.raises(ValueError, match="n >= 2"):
        RegularGrid(n=1)
    with pytest.raises(ValueError, match="n >= 2"):
        RegularGrid(n=0)


def test_invalid_range_raises():
    g = RegularGrid(n=4)
    with pytest.raises(ValueError, match="high > low"):
        g.points(1.0, 0.0)
    with pytest.raises(ValueError, match="high > low"):
        g.points(1.0, 1.0)


def test_grid_is_frozen():
    g = RegularGrid(n=4)
    with pytest.raises(AttributeError):
        g.n = 8


def test_regular_grid_accepts_and_ignores_warp_kwarg():
    """The warp= kwarg is for signature uniformity with WarpedGrid; unused here."""
    g = RegularGrid(n=5)
    pts_no_warp = g.points(0.0, 1.0)
    pts_with_warp = g.points(0.0, 1.0, warp="asinh")  # should be identical
    assert torch.allclose(pts_no_warp, pts_with_warp)


# --- WarpedGrid -----------------------------------------------------------


def test_warped_asinh_endpoints():
    g = WarpedGrid(n=10, warp="asinh")
    pts = g.points(0.0, 10.0)
    assert pts.shape == (10,)
    assert pts[0].item() == pytest.approx(0.0, abs=1e-12)
    assert pts[-1].item() == pytest.approx(10.0)


def test_warped_asinh_concentrates_near_zero():
    """Spacing should grow from low end to high end under asinh."""
    g = WarpedGrid(n=10, warp="asinh")
    pts = g.points(0.0, 100.0)
    diffs = torch.diff(pts)
    assert diffs[0].item() < diffs[-1].item()


def test_warped_asinh_uniform_in_warped_space():
    """asinh(points) should be uniformly spaced."""
    g = WarpedGrid(n=11, warp="asinh")
    pts = g.points(0.0, 50.0)
    warped = torch.asinh(pts)
    diffs = torch.diff(warped)
    assert torch.allclose(diffs, torch.full_like(diffs, diffs[0].item()), atol=1e-12)


def test_warped_log_endpoints():
    g = WarpedGrid(n=10, warp="log")
    pts = g.points(1.0, 100.0)
    assert pts[0].item() == pytest.approx(1.0)
    assert pts[-1].item() == pytest.approx(100.0)


def test_warped_log_uniform_in_log_space():
    g = WarpedGrid(n=11, warp="log")
    pts = g.points(1.0, 1024.0)
    log_pts = torch.log(pts)
    diffs = torch.diff(log_pts)
    expected_step = math.log(1024.0) / 10  # 10 intervals
    assert torch.allclose(
        diffs, torch.full_like(diffs, expected_step), atol=1e-12
    )


def test_warped_log_requires_positive_low():
    g = WarpedGrid(n=5, warp="log")
    with pytest.raises(ValueError, match="log warp requires low > 0"):
        g.points(0.0, 10.0)
    with pytest.raises(ValueError, match="log warp requires low > 0"):
        g.points(-1.0, 10.0)


def test_warped_inherits_warp_from_caller():
    """When WarpedGrid.warp is None, the warp= kwarg supplies it (state's warp)."""
    inherited = WarpedGrid(n=10).points(0.0, 10.0, warp="asinh")
    explicit = WarpedGrid(n=10, warp="asinh").points(0.0, 10.0)
    assert torch.allclose(inherited, explicit)


def test_warped_self_overrides_inherited():
    """WarpedGrid.warp wins over the warp= kwarg if both are set."""
    g = WarpedGrid(n=10, warp="asinh")
    pts = g.points(1.0, 100.0, warp="log")  # log should be ignored
    expected = WarpedGrid(n=10, warp="asinh").points(1.0, 100.0)
    assert torch.allclose(pts, expected)


def test_warped_no_warp_raises():
    g = WarpedGrid(n=10)
    with pytest.raises(ValueError, match="requires a warp"):
        g.points(0.0, 10.0)


def test_warped_n_below_two_raises():
    with pytest.raises(ValueError, match="n >= 2"):
        WarpedGrid(n=1, warp="asinh")


def test_warped_unknown_string_raises():
    g = WarpedGrid(n=5, warp="bogus")
    with pytest.raises(ValueError, match="unknown warp"):
        g.points(0.0, 10.0)


def test_warped_callable_not_implemented():
    g = WarpedGrid(n=5, warp=lambda x: x)
    with pytest.raises(NotImplementedError):
        g.points(0.0, 10.0)


def test_warped_invalid_range_raises():
    g = WarpedGrid(n=5, warp="asinh")
    with pytest.raises(ValueError, match="high > low"):
        g.points(5.0, 5.0)


def test_warped_grid_frozen():
    g = WarpedGrid(n=10, warp="asinh")
    with pytest.raises(AttributeError):
        g.n = 20


def test_warped_dtype_and_device():
    g = WarpedGrid(n=5, warp="asinh")
    pts = g.points(0.0, 10.0, dtype=torch.float32)
    assert pts.dtype == torch.float32
    assert pts.device.type == "cpu"


# --- transform_for_interp ------------------------------------------------


def test_regular_transform_for_interp_is_identity():
    g = RegularGrid(n=5)
    x = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float64)
    out = g.transform_for_interp(x)
    assert torch.allclose(out, x)
    # warp kwarg is accepted and ignored
    assert torch.allclose(g.transform_for_interp(x, warp="asinh"), x)


def test_warped_transform_asinh():
    g = WarpedGrid(n=5, warp="asinh")
    x = torch.tensor([0.0, 1.0, 10.0], dtype=torch.float64)
    out = g.transform_for_interp(x)
    assert torch.allclose(out, torch.asinh(x))


def test_warped_transform_log():
    g = WarpedGrid(n=5, warp="log")
    x = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
    out = g.transform_for_interp(x)
    assert torch.allclose(out, torch.log(x))


def test_warped_transform_inherits_warp():
    g = WarpedGrid(n=5)  # no warp set
    x = torch.tensor([1.0, 10.0], dtype=torch.float64)
    out = g.transform_for_interp(x, warp="log")
    assert torch.allclose(out, torch.log(x))


def test_warped_transform_log_clamps_nonpositive():
    """Log transform should clamp non-positive inputs to tiny positive."""
    g = WarpedGrid(n=5, warp="log")
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    out = g.transform_for_interp(x)
    # all finite (no -inf or NaN from log of 0 or negative)
    assert torch.isfinite(out).all()
    # the positive value transforms to log(1) = 0
    assert out[2].item() == pytest.approx(0.0)


def test_warped_transform_pts_are_uniform_under_own_warp():
    """The transform of the grid points themselves is uniformly spaced."""
    g = WarpedGrid(n=11, warp="asinh")
    pts = g.points(0.0, 50.0)
    u = g.transform_for_interp(pts)
    diffs = torch.diff(u)
    assert torch.allclose(diffs, torch.full_like(diffs, diffs[0].item()), atol=1e-12)


def test_warped_interp_is_exact_for_function_linear_in_warped_coord():
    """With a log warp, a function linear in ``log(w)`` is recovered exactly
    by multilinear interp in warped space — this is the payoff of doing
    interpolation in the transformed coordinate."""
    from bellgrid.interpolation import multilinear

    g = WarpedGrid(n=10, warp="log")
    s_pts = g.points(0.5, 50.0)            # log-spaced wealth
    u_pts = g.transform_for_interp(s_pts)  # uniform in log space

    # f(w) = 3 + 2*log(w) — exactly linear in u
    values = 3.0 + 2.0 * u_pts

    query = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    u_query = g.transform_for_interp(query)
    result = multilinear([u_pts], values, [u_query])
    expected = 3.0 + 2.0 * torch.log(query)

    assert torch.allclose(result, expected, atol=1e-12)
