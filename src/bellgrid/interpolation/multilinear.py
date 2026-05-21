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
    """Hot path. Detects continuous-vs-discrete per axis from query dtype."""
    K = len(axes)
    is_continuous = [queries[k].dtype.is_floating_point for k in range(K)]

    # Per-axis bracket info. For continuous: idx_lo + weight, with hi = lo + 1.
    # For discrete: idx = the long query directly, no weight.
    idx_base_list: list[torch.Tensor] = []
    weight_list: list = []  # weight or None
    for k in range(K):
        if is_continuous[k]:
            axis = axes[k]
            query = queries[k]
            idx = torch.searchsorted(axis, query, right=False)
            idx = torch.clamp(idx, 1, axis.numel() - 1)
            left = idx - 1
            x_left = axis[left]
            x_right = axis[idx]
            w = (query - x_left) / (x_right - x_left)
            w = torch.clamp(w, 0.0, 1.0)
            idx_base_list.append(left)
            weight_list.append(w)
        else:
            idx_base_list.append(queries[k])
            weight_list.append(None)

    dims = values.shape
    strides = [0] * K
    strides[-1] = 1
    for k in range(K - 2, -1, -1):
        strides[k] = strides[k + 1] * dims[k + 1]

    V_flat = values.reshape(-1)

    # Only continuous axes contribute corners; discrete axes contribute a
    # single fixed index for every corner.
    cont_axes = [k for k in range(K) if is_continuous[k]]

    result = None
    for corner in range(2 ** len(cont_axes)):
        flat_idx = None
        weight = None
        for cont_pos, k in enumerate(cont_axes):
            bit = (corner >> cont_pos) & 1
            idx_k = idx_base_list[k] + bit
            w_k = weight_list[k] if bit else (1.0 - weight_list[k])
            contribution = idx_k * strides[k]
            flat_idx = contribution if flat_idx is None else flat_idx + contribution
            weight = w_k if weight is None else weight * w_k
        # Add discrete-axis contributions (single index, no weight factor).
        for k in range(K):
            if not is_continuous[k]:
                contribution = idx_base_list[k] * strides[k]
                flat_idx = contribution if flat_idx is None else flat_idx + contribution

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
