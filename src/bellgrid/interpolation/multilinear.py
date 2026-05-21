"""Multilinear interpolation on a tensor-product grid. GPU-vectorized.

Supports any number of axes ``K``. Each axis is independently sorted; the
result is the standard ``2^K``-corner blend with linear weights in each
dimension. Queries outside the grid are clamped to the nearest edge value
along each axis.

For ``K=1`` you can pass bare tensors as a shortcut.

The hot path is JIT-compiled via ``torch.compile`` — for the LQG-scale
K=2 case (~12M queries) this is ~10x faster than the pure-eager kernel
because the compiler fuses the 2^K corner gathers and the surrounding
arithmetic into a few kernels instead of allocating and discarding
intermediate tensors per corner.
"""

import torch


def _multilinear_core(
    axes: list[torch.Tensor],
    values: torch.Tensor,
    queries: list[torch.Tensor],
) -> torch.Tensor:
    """Hot path: per-axis bracketing + 2^K corner weighted sum.

    Assumes inputs have already been normalized to lists and validated.
    Compiled via ``torch.compile`` at module load.
    """
    K = len(axes)

    idx_lo_list: list[torch.Tensor] = []
    weight_list: list[torch.Tensor] = []
    for k in range(K):
        axis = axes[k]
        query = queries[k]
        idx = torch.searchsorted(axis, query, right=False)
        idx = torch.clamp(idx, 1, axis.numel() - 1)
        left = idx - 1
        right = idx
        x_left = axis[left]
        x_right = axis[right]
        w = (query - x_left) / (x_right - x_left)
        w = torch.clamp(w, 0.0, 1.0)
        idx_lo_list.append(left)
        weight_list.append(w)

    dims = values.shape
    strides = [0] * K
    strides[-1] = 1
    for k in range(K - 2, -1, -1):
        strides[k] = strides[k + 1] * dims[k + 1]

    V_flat = values.reshape(-1)

    result = None
    for corner in range(2 ** K):
        flat_idx = None
        weight = None
        for k in range(K):
            bit = (corner >> k) & 1
            idx_k = idx_lo_list[k] + bit
            w_k = weight_list[k] if bit else (1.0 - weight_list[k])
            contribution = idx_k * strides[k]
            flat_idx = contribution if flat_idx is None else flat_idx + contribution
            weight = w_k if weight is None else weight * w_k
        gathered = V_flat[flat_idx]
        term = weight * gathered
        result = term if result is None else result + term

    return result


# JIT-compile the hot path. The compiled artifact gives ~10x on big query
# batches (the LQG K=2 case with ~12M queries goes from ~670 ms → ~67 ms)
# at the cost of a few seconds of first-time compile overhead per K. We
# only route through it once the query batch is large enough that the
# compile cost is amortised over the work (small tests / single-point
# policy queries get the eager kernel directly).
_multilinear_core_compiled = torch.compile(_multilinear_core, dynamic=True)
_COMPILE_QUERY_THRESHOLD = 100_000


def multilinear(
    axes: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    values: torch.Tensor,
    queries: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Multilinear interpolation with edge-clamping for out-of-bounds queries.

    Parameters
    ----------
    axes : 1-D tensor (``K=1`` shortcut) or list/tuple of K 1-D tensors. The
        k-th axis is the coordinate vector for the k-th dimension of
        ``values`` and must be sorted ascending.
    values : tensor of shape ``(n_0, n_1, ..., n_{K-1})``.
    queries : tensor (``K=1`` shortcut) or list/tuple of K tensors that share
        a common shape. ``queries[k]`` gives the coordinate along axis k.

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
        raise ValueError(
            f"axes ({K}) must match queries ({len(queries)})"
        )
    if K == 0:
        raise ValueError("multilinear requires at least one axis")

    for k in range(K):
        axis = axes[k]
        if axis.ndim != 1:
            raise ValueError(
                f"axis {k} must be 1-D, got shape {tuple(axis.shape)}"
            )
        if axis.numel() < 2:
            raise ValueError(
                f"axis {k} must have at least 2 points, got {axis.numel()}"
            )
        if values.shape[k] != axis.numel():
            raise ValueError(
                f"axis {k} length ({axis.numel()}) does not match "
                f"values.shape[{k}] ({values.shape[k]})"
            )

    if queries[0].numel() >= _COMPILE_QUERY_THRESHOLD:
        return _multilinear_core_compiled(axes, values, queries)
    return _multilinear_core(axes, values, queries)
