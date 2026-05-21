"""Multilinear interpolation on a tensor-product grid. GPU-vectorized.

Currently 1-D only. The API is shaped for the multi-D extension: pass
`axes` as a list of K 1-D tensors and `queries` as a list of K
common-shaped tensors. For 1-D you can pass bare tensors as a shortcut.
"""

import torch


def multilinear(
    axes: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    values: torch.Tensor,
    queries: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Multilinear interpolation with edge-clamping for out-of-bounds queries.

    Parameters
    ----------
    axes : 1-D tensor (1-D shortcut) or list/tuple of K 1-D tensors. The
        k-th axis is the coordinate vector for the k-th dimension of
        ``values`` and must be sorted ascending.
    values : tensor of shape ``(n_0, n_1, ..., n_{K-1})``.
    queries : tensor (1-D shortcut) or list/tuple of K tensors that share
        a common shape. ``queries[k]`` gives the coordinate along axis k.

    Returns
    -------
    Tensor with the common query shape. Queries outside the grid are
    clamped to the nearest edge value along each axis.

    Notes
    -----
    Only ``K == 1`` is implemented today. The signature is the multi-D
    one so the solver and tests don't change shape when ``K > 1`` lands.
    """
    if isinstance(axes, torch.Tensor):
        axes = [axes]
    else:
        axes = list(axes)

    if isinstance(queries, torch.Tensor):
        queries = [queries]
    else:
        queries = list(queries)

    if len(axes) != values.ndim:
        raise ValueError(
            f"axes ({len(axes)}) must match values.ndim ({values.ndim})"
        )
    if len(axes) != len(queries):
        raise ValueError(
            f"axes ({len(axes)}) must match queries ({len(queries)})"
        )

    if len(axes) != 1:
        raise NotImplementedError(
            f"multilinear only supports K=1 for now; got K={len(axes)}"
        )

    return _interp_1d(axes[0], values, queries[0])


def _interp_1d(
    grid: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    if grid.ndim != 1:
        raise ValueError(f"axis must be 1-D, got shape {tuple(grid.shape)}")
    if values.shape != grid.shape:
        raise ValueError(
            f"values must have the same shape as the axis; got values "
            f"{tuple(values.shape)}, axis {tuple(grid.shape)}"
        )
    if grid.numel() < 2:
        raise ValueError(f"axis must have at least 2 points, got {grid.numel()}")

    # searchsorted gives the index where query would be inserted.
    # Clamp to [1, N-1] so left = idx - 1 and right = idx are both in-range.
    idx = torch.searchsorted(grid, query, right=False)
    idx = torch.clamp(idx, 1, grid.numel() - 1)
    left = idx - 1
    right = idx

    x_left = grid[left]
    x_right = grid[right]
    y_left = values[left]
    y_right = values[right]

    # Clamping the blend weight to [0, 1] yields edge-value extrapolation
    # for out-of-bounds queries.
    weight = (query - x_left) / (x_right - x_left)
    weight = torch.clamp(weight, 0.0, 1.0)

    return y_left + weight * (y_right - y_left)
