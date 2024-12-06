""" MLP that approximates spherical interpolation of 3 rotations. """

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float


class SphericalInterpolationMLP(nn.Module):
    """MLP that approximates spherical interpolation of 3 rotations. """

    def __init__(self) -> None:
        """Initialize the MLP. """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3*4 + 3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )

    def forward(
        self,
        barycentric_weights: Float[torch.Tensor, "n_gaussians 3"],
        rotations: Float[torch.Tensor, "n_gaussians 3 4"],
    ) -> Float[torch.Tensor, "n_gaussians 4"]:
        """
        Args:
            barycentric_weights (Tensor): Barycentric weights for each gaussian,
                shape (n_gaussians, 3).
            rotations (Tensor): Rotations for each gaussian, shape (n_gaussians, 3, 4).
        """
        rotations = rearrange(rotations, "n_gaussians a b -> n_gaussians (a b)")
        x = torch.cat([barycentric_weights, rotations], dim=-1)
        x = self.mlp.forward(x)
        return nn.functional.normalize(x, p=2, dim=-1)
