import pytest
import torch

from bellgrid.grids import RegularGrid


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
