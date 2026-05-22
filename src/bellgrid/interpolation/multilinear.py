"""Multilinear interpolation on a tensor-product grid. GPU-vectorized.

Supports any number of axes ``K``, each independently either:

- **continuous** — the axis is a sorted-ascending 1-D tensor of grid
  coordinates and the query is a float tensor; standard linear interp
  with edge-clamping at the boundaries.
- **discrete**  — the axis describes a finite integer-indexed dimension
  (typically ``torch.arange(n)``) and the query is a ``torch.long``
  tensor of indices into that dimension; no interpolation, exact gather.

Axis kind is detected from ``queries[k].dtype``: floating-point → continuous,
integer → discrete. So a typical mixed call is

    multilinear(
        axes=[wealth_grid, regime_arange],
        values=V,                          # shape (n_wealth, n_regime)
        queries=[next_wealth, next_regime],  # next_wealth float, next_regime long
    )

The hot path is JIT-compiled via ``torch.compile`` when the query batch
crosses a threshold (the LQG-scale K=2 / 12M-query case runs ~10x faster
compiled than eager).
"""

import torch


def _multilinear_core(
    axes: list[torch.Tensor],
    values: torch.Tensor,
    queries: list[torch.Tensor],
) -> torch.Tensor:
    """Hot path. Detects continuous-vs-discrete per axis from query dtype.

    Implementation precomputes everything that's constant across the
    ``2^K_c`` corner sum (the lo/hi stride offsets per continuous axis,
    the lo/hi weight per continuous axis, and the combined discrete-axis
    offset summed once across all discrete axes). The corner loop then
    just picks lo-vs-hi per axis by the corner bits. This restructure
    saves ~25-36% in eager mode for K_c >= 2 by avoiding redundant
    tensor allocations inside the loop; under ``torch.compile`` the
    savings are caught by CSE and the compiled artifact is unchanged.
    """
    K = len(axes)
    is_cont = [queries[k].dtype.is_floating_point for k in range(K)]

    # Row-major strides (Python ints)
    dims = values.shape
    strides = [0] * K
    strides[-1] = 1
    for k in range(K - 2, -1, -1):
        strides[k] = strides[k + 1] * dims[k + 1]

    cont_axes = [k for k in range(K) if is_cont[k]]
    disc_axes = [k for k in range(K) if not is_cont[k]]
    V_flat = values.reshape(-1)

    # Per continuous axis: precompute lo/hi stride-multiplied offsets and
    # lo/hi weights so the corner loop picks them by bit without recomputing.
    cont_lo_off: list[torch.Tensor] = []
    cont_hi_off: list[torch.Tensor] = []
    cont_w_lo: list[torch.Tensor] = []
    cont_w_hi: list[torch.Tensor] = []
    for k in cont_axes:
        axis = axes[k]
        query = queries[k]
        s = strides[k]
        idx = torch.searchsorted(axis, query, right=False)
        idx = torch.clamp(idx, 1, axis.numel() - 1)
        left = idx - 1
        x_left = axis[left]
        x_right = axis[idx]
        w = torch.clamp((query - x_left) / (x_right - x_left), 0.0, 1.0)
        cont_lo_off.append(left * s)
        cont_hi_off.append(idx * s)
        cont_w_hi.append(w)
        cont_w_lo.append(1.0 - w)

    # Combine all discrete-axis contributions once (was previously computed
    # 2^K_c times inside the corner loop).
    disc_offset: torch.Tensor | None = None
    for k in disc_axes:
        c = queries[k] * strides[k]
        disc_offset = c if disc_offset is None else disc_offset + c

    # Sum over 2^K_c corners.
    result = None
    n_corners = 2 ** len(cont_axes)
    for corner in range(n_corners):
        flat_idx = disc_offset
        weight = None
        for cont_pos in range(len(cont_axes)):
            bit = (corner >> cont_pos) & 1
            off = cont_hi_off[cont_pos] if bit else cont_lo_off[cont_pos]
            w = cont_w_hi[cont_pos] if bit else cont_w_lo[cont_pos]
            flat_idx = off if flat_idx is None else flat_idx + off
            weight = w if weight is None else weight * w
        gathered = V_flat[flat_idx]
        term = gathered if weight is None else weight * gathered
        result = term if result is None else result + term

    return result


# JIT-compile the hot path. The compiled artifact gives ~10x on big query
# batches (LQG K=2 with ~12M queries: 667 ms → 67 ms, ~5.7 ns/query) at
# the cost of a few seconds of first-time compile overhead per K. We only
# route through it once the query batch is large enough that the compile
# cost is amortised over the work.
_multilinear_core_compiled = torch.compile(_multilinear_core, dynamic=True)
_COMPILE_QUERY_THRESHOLD = 100_000


def multilinear(
    axes: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    values: torch.Tensor,
    queries: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Mixed continuous/discrete multilinear interpolation.

    Parameters
    ----------
    axes : 1-D tensor (``K=1`` shortcut) or list/tuple of K 1-D tensors.
        For continuous axes: sorted ascending coordinates.
        For discrete axes: typically ``torch.arange(n)`` (long dtype).
    values : tensor of shape ``(n_0, ..., n_{K-1})``.
    queries : tensor (``K=1`` shortcut) or list/tuple of K tensors that share
        a common shape. Axis kind is detected by query dtype: floating →
        continuous (interpolated), integer → discrete (exact gather).

    Returns
    -------
    Tensor with the common query shape.
    """
    if isinstance(axes, torch.Tensor):
        axes = [axes]
    else:
        axes = list(axes)
    if isinstance(queries, torch.Tensor):
        queries = [queries]
    else:
        queries = list(queries)

    K = len(axes)
    if K != values.ndim:
        raise ValueError(
            f"axes ({K}) must match values.ndim ({values.ndim})"
        )
    if K != len(queries):
        raise ValueError(f"axes ({K}) must match queries ({len(queries)})")
    if K == 0:
        raise ValueError("multilinear requires at least one axis")

    for k in range(K):
        axis = axes[k]
        if axis.ndim != 1:
            raise ValueError(
                f"axis {k} must be 1-D, got shape {tuple(axis.shape)}"
            )
        if axis.numel() < 1:
            raise ValueError(
                f"axis {k} must have at least 1 point, got {axis.numel()}"
            )
        if values.shape[k] != axis.numel():
            raise ValueError(
                f"axis {k} length ({axis.numel()}) does not match "
                f"values.shape[{k}] ({values.shape[k]})"
            )
        # Continuous axes still need at least 2 points for bracketing.
        if queries[k].dtype.is_floating_point and axis.numel() < 2:
            raise ValueError(
                f"continuous axis {k} must have at least 2 points"
            )

    if queries[0].numel() >= _COMPILE_QUERY_THRESHOLD:
        return _multilinear_core_compiled(axes, values, queries)
    return _multilinear_core(axes, values, queries)
