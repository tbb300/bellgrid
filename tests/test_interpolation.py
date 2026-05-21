import pytest
import torch

from bellgrid.interpolation import multilinear


def test_linear_function_exact():
    grid = torch.linspace(0.0, 10.0, 11, dtype=torch.float64)
    values = 2.0 * grid + 3.0
    query = torch.tensor([1.5, 4.7, 8.2], dtype=torch.float64)
    expected = 2.0 * query + 3.0
    assert torch.allclose(multilinear(grid, values, query), expected)


def test_at_grid_points_returns_grid_values():
    grid = torch.linspace(0.0, 4.0, 5, dtype=torch.float64)
    values = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    assert torch.allclose(multilinear(grid, values, grid), values)


def test_below_grid_clamps_to_first_value():
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
    query = torch.tensor([-5.0, -1.0, -1e-9], dtype=torch.float64)
    out = multilinear(grid, values, query)
    assert torch.allclose(out, torch.full_like(out, 10.0))


def test_above_grid_clamps_to_last_value():
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
    query = torch.tensor([2.0, 100.0, 1.0 + 1e-9], dtype=torch.float64)
    out = multilinear(grid, values, query)
    assert torch.allclose(out, torch.full_like(out, 50.0))


def test_midpoint_is_average():
    grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([4.0, 10.0], dtype=torch.float64)
    out = multilinear(grid, values, torch.tensor([0.5], dtype=torch.float64))
    assert out.item() == pytest.approx(7.0)


def test_quadratic_approximation_error_decays_with_grid():
    query = torch.tensor([0.123, 0.456, 0.789], dtype=torch.float64)
    expected = query**2

    def max_err(n):
        g = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        return (multilinear(g, g**2, query) - expected).abs().max().item()

    err_coarse = max_err(11)
    err_fine = max_err(21)
    assert err_fine < err_coarse / 3.0


def test_query_shape_is_preserved():
    grid = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    values = 3.0 * grid
    query = torch.rand(5, 7, 3, dtype=torch.float64)
    out = multilinear(grid, values, query)
    assert out.shape == query.shape


def test_list_form_1d_matches_bare_form():
    """The list-of-axes API and the 1-D shortcut give the same result."""
    grid = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    values = grid**3
    query = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)

    bare = multilinear(grid, values, query)
    listed = multilinear([grid], values, [query])
    assert torch.allclose(bare, listed)


def test_tuple_form_also_works():
    grid = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    values = grid**3
    query = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)

    out = multilinear((grid,), values, (query,))
    assert out.shape == query.shape


def test_mismatched_shapes_raise():
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    values_bad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="does not match values.shape"):
        multilinear(grid, values_bad, torch.tensor([0.5], dtype=torch.float64))


def test_axis_not_1d_raises():
    grid = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="axes .* must match values.ndim"):
        multilinear([grid], grid, [torch.tensor([0.0], dtype=torch.float64)])


def test_axis_too_short_raises():
    grid = torch.tensor([0.0], dtype=torch.float64)
    values = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 2 points"):
        multilinear(grid, values, torch.tensor([0.5], dtype=torch.float64))


def test_axes_values_dim_mismatch_raises():
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    values_2d = torch.zeros(5, 5, dtype=torch.float64)
    query = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="must match values.ndim"):
        multilinear([grid], values_2d, [query])


def test_axes_queries_count_mismatch_raises():
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    values = torch.zeros(5, dtype=torch.float64)
    q1 = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="must match queries"):
        multilinear([grid], values, [q1, q1])


def test_2d_bilinear_function_exact():
    """f(x, y) = a*x + b*y + c — bilinear, recovered exactly."""
    gx = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    gy = torch.linspace(0.0, 2.0, 7, dtype=torch.float64)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    values = 3.0 * X + 2.0 * Y + 1.0
    qx = torch.tensor([0.13, 0.47, 0.82], dtype=torch.float64)
    qy = torch.tensor([0.31, 1.05, 1.79], dtype=torch.float64)
    expected = 3.0 * qx + 2.0 * qy + 1.0
    out = multilinear([gx, gy], values, [qx, qy])
    assert torch.allclose(out, expected, atol=1e-12)


def test_2d_at_grid_points():
    gx = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    gy = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    values = X**2 + Y**3
    # Query at the (i, j) = (1, 2) grid point
    qx = torch.tensor([gx[1].item()], dtype=torch.float64)
    qy = torch.tensor([gy[2].item()], dtype=torch.float64)
    out = multilinear([gx, gy], values, [qx, qy])
    assert out.item() == pytest.approx(values[1, 2].item())


def test_2d_center_of_unit_cell():
    """Bilinear at the center of a cell averages the four corners."""
    gx = torch.tensor([0.0, 1.0], dtype=torch.float64)
    gy = torch.tensor([0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64)
    qx = torch.tensor([0.5], dtype=torch.float64)
    qy = torch.tensor([0.5], dtype=torch.float64)
    out = multilinear([gx, gy], values, [qx, qy])
    assert out.item() == pytest.approx(25.0)


def test_2d_below_grid_clamps_to_corner():
    gx = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    gy = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    values = X + Y
    qx = torch.tensor([-1.0], dtype=torch.float64)
    qy = torch.tensor([-2.0], dtype=torch.float64)
    out = multilinear([gx, gy], values, [qx, qy])
    assert out.item() == pytest.approx(values[0, 0].item())


def test_3d_trilinear_function_exact():
    gx = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    gy = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    gz = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing="ij")
    values = 1.0 + 2.0 * X + 3.0 * Y + 4.0 * Z  # affine, exact in 3-D
    qx = torch.tensor([0.27], dtype=torch.float64)
    qy = torch.tensor([0.51], dtype=torch.float64)
    qz = torch.tensor([0.83], dtype=torch.float64)
    expected = 1.0 + 2.0 * qx + 3.0 * qy + 4.0 * qz
    out = multilinear([gx, gy, gz], values, [qx, qy, qz])
    assert torch.allclose(out, expected, atol=1e-12)


def test_2d_query_shape_preserved():
    gx = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    gy = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    values = X * Y
    qx = torch.rand(3, 4, dtype=torch.float64)
    qy = torch.rand(3, 4, dtype=torch.float64)
    out = multilinear([gx, gy], values, [qx, qy])
    assert out.shape == qx.shape


# --- mixed continuous + discrete axes ----------------------------------


def test_mixed_cont_disc_exact_recovery():
    """V(c, d) = 2c + d. Continuous interp + exact discrete gather."""
    grid_c = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    grid_d = torch.arange(3, dtype=torch.long)
    C, D = torch.meshgrid(grid_c, grid_d.to(torch.float64), indexing="ij")
    V = 2.0 * C + D
    qc = torch.tensor([0.5, 0.25, 0.75], dtype=torch.float64)
    qd = torch.tensor([1, 0, 2], dtype=torch.long)
    out = multilinear([grid_c, grid_d], V, [qc, qd])
    expected = 2.0 * qc + qd.to(torch.float64)
    assert torch.allclose(out, expected, atol=1e-12)


def test_mixed_at_grid_point():
    grid_c = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    grid_d = torch.arange(3, dtype=torch.long)
    V = torch.rand(5, 3, dtype=torch.float64)
    # Query at exact grid points
    qc = torch.tensor([grid_c[1].item()], dtype=torch.float64)
    qd = torch.tensor([2], dtype=torch.long)
    out = multilinear([grid_c, grid_d], V, [qc, qd])
    assert out.item() == pytest.approx(V[1, 2].item())


def test_all_discrete_gather():
    """All-discrete query is just an exact gather."""
    grid_d1 = torch.arange(3, dtype=torch.long)
    grid_d2 = torch.arange(4, dtype=torch.long)
    V = torch.rand(3, 4, dtype=torch.float64)
    qd1 = torch.tensor([1, 2], dtype=torch.long)
    qd2 = torch.tensor([3, 0], dtype=torch.long)
    out = multilinear([grid_d1, grid_d2], V, [qd1, qd2])
    assert torch.allclose(
        out,
        torch.tensor([V[1, 3].item(), V[2, 0].item()], dtype=torch.float64),
    )


def test_discrete_dim_with_size_one_axis():
    """Single-category discrete axis: must accept 1-point axis."""
    grid_c = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    grid_d = torch.arange(1, dtype=torch.long)
    V = torch.linspace(0, 1, 5, dtype=torch.float64).unsqueeze(-1)  # shape (5, 1)
    qc = torch.tensor([0.5], dtype=torch.float64)
    qd = torch.tensor([0], dtype=torch.long)
    out = multilinear([grid_c, grid_d], V, [qc, qd])
    assert out.item() == pytest.approx(0.5)
