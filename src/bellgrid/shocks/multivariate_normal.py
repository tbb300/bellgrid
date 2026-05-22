"""Multivariate normal innovation shock with Cholesky-rotated Gauss-Hermite quadrature.

``X ~ MVN(mean, cov)`` over K dimensions, with one user-visible shock name
per dimension. Quadrature is the K-dimensional tensor product of standard-
normal GH nodes, rotated by the Cholesky factor of ``cov``. Total node
count is ``n_quad ** K`` so this is best for K ≤ 3 or so; larger K wants
a sparse-grid or Monte Carlo approach.

For consistency with the rest of the shock interface, both
``nodes_and_weights`` and ``sample`` return a ``{name: tensor}`` dict
rather than a single tensor — the solver and ``simulate`` dispatch on
dict-vs-tensor at the call site.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MultivariateNormal:
    """Multivariate normal innovation: ``X ~ N(mean, cov)`` over K dimensions.

    Parameters
    ----------
    names : tuple of K strings
        One name per dimension. Each name surfaces in ``shock[name]`` in
        user-supplied ``transition`` / ``reward`` callables.
    mean : array-like of length K, default zeros.
    cov : (K, K) symmetric positive-definite array-like, default identity.
    """

    names: tuple
    mean: object = None
    cov: object = None

    def __post_init__(self):
        if not isinstance(self.names, tuple):
            object.__setattr__(self, "names", tuple(self.names))
        if len(self.names) < 1:
            raise ValueError("MultivariateNormal requires at least one name")
        if any(n is None for n in self.names):
            raise ValueError("MultivariateNormal names cannot be None")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"MultivariateNormal names must be unique: {self.names}")
        K = len(self.names)
        if self.mean is None:
            object.__setattr__(self, "mean", np.zeros(K, dtype=np.float64))
        else:
            mean_arr = np.asarray(self.mean, dtype=np.float64)
            if mean_arr.shape != (K,):
                raise ValueError(
                    f"mean must have shape ({K},); got {mean_arr.shape}"
                )
            object.__setattr__(self, "mean", mean_arr)
        if self.cov is None:
            object.__setattr__(self, "cov", np.eye(K, dtype=np.float64))
        else:
            cov_arr = np.asarray(self.cov, dtype=np.float64)
            if cov_arr.shape != (K, K):
                raise ValueError(
                    f"cov must have shape ({K}, {K}); got {cov_arr.shape}"
                )
            if not np.allclose(cov_arr, cov_arr.T, atol=1e-10):
                raise ValueError("cov must be symmetric")
            object.__setattr__(self, "cov", cov_arr)

    @property
    def K(self) -> int:
        return len(self.names)

    def _cholesky(self) -> np.ndarray:
        try:
            return np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"cov is not positive definite: {e}") from None

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[dict, torch.Tensor]:
        """Tensor-product Gauss-Hermite quadrature, rotated by Chol(cov).

        Returns
        -------
        nodes : dict {name: (n_quad ** K,) tensor}
            Per-dimension MVN realizations at each joint quadrature node.
        weights : (n_quad ** K,) tensor summing to 1.
        """
        if n_quad < 1:
            raise ValueError(f"n_quad must be >= 1, got {n_quad}")

        K = self.K
        L = self._cholesky()  # lower-triangular K × K

        raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(n_quad)
        weights_1d = raw_weights / math.sqrt(2.0 * math.pi)

        # Cartesian product of 1-D nodes (K axes, indexing="ij")
        z_mesh = np.meshgrid(*([raw_nodes] * K), indexing="ij")
        w_mesh = np.meshgrid(*([weights_1d] * K), indexing="ij")
        z_flat = np.stack([m.flatten() for m in z_mesh], axis=-1)   # (N_q, K)
        w_per_axis = np.stack([m.flatten() for m in w_mesh], axis=-1)  # (N_q, K)
        weights = np.prod(w_per_axis, axis=-1)                       # (N_q,)

        # X = mean + L @ Z  (transform standard-normal Z to MVN(mean, cov))
        x = self.mean + z_flat @ L.T  # (N_q, K)

        nodes_dict = {
            name: torch.as_tensor(x[:, i], dtype=dtype, device=device)
            for i, name in enumerate(self.names)
        }
        return nodes_dict, torch.as_tensor(weights, dtype=dtype, device=device)

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> dict:
        """Draw n MVN samples, returned as a ``{name: (n,) tensor}`` dict."""
        K = self.K
        L = torch.as_tensor(self._cholesky(), dtype=dtype, device=device)
        mean_t = torch.as_tensor(self.mean, dtype=dtype, device=device)
        z = torch.randn(n, K, generator=generator, dtype=dtype, device=device)
        x = mean_t + z @ L.T
        return {name: x[:, i].contiguous() for i, name in enumerate(self.names)}
