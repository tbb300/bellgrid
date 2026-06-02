"""Small MLP building blocks for the actor and critic.

Both are plain feed-forward nets over the featurised state (see
``RLSetup.featurize``). The critic outputs a scalar value; the actor outputs one
raw real per continuous action, squashed into the (possibly state-dependent)
action bounds by the solver — kept out of the net so the net stays a pure
function of the state features.
"""

import torch
from torch import nn

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


class MLP(nn.Module):
    """Feed-forward net. ``nn.Linear`` acts on the last dim, so any leading
    batch shape (``[B]`` or ``[B, n_q]``) passes through unchanged."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple, activation: str):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}"
            )
        act = _ACTIVATIONS[activation]
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def squash_to_bounds(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Map an unbounded real to ``[lo, hi]`` via a sigmoid. Differentiable, so
    the actor's pathwise gradient flows through the bound transform."""
    return lo + (hi - lo) * torch.sigmoid(raw)


class NormalizedCritic(nn.Module):
    """Critic that predicts the value in a standardized space and unnormalizes.

    Value functions can span hundreds of units (Merton's ``A + B·log w`` does),
    which makes a raw-MSE regression poorly conditioned. We fit the inner MLP to
    ``(target − mean) / std`` and expose the un-normalized value, so the rest of
    the solver (continuation-value lookups, the value callable) always sees the
    true value. ``mean``/``std`` are set once per period from a target sample (a
    PopArt-style trick, fixed per period rather than running).

    The inner MLP may have ``Q`` outputs — the **quantile atoms** of the return
    distribution (``Q=1`` is a plain scalar critic). All methods keep the trailing
    ``[..., Q]`` axis; truncation/averaging across atoms happens in
    ``_TruncatedEnsemble``.
    """

    def __init__(self, mlp: MLP):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("std", torch.ones((), dtype=torch.float64))

    def set_norm(self, mean: float, std: float) -> None:
        self.mean.fill_(float(mean))
        self.std.fill_(max(float(std), 1e-8))

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.std

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Inner MLP atoms in standardized space ``[..., Q]`` (what training
        regresses)."""
        return self.mlp(x)

    def atoms(self, x: torch.Tensor) -> torch.Tensor:
        """Un-normalized quantile atoms ``[..., Q]``."""
        return self.mean + self.std * self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.atoms(x)
