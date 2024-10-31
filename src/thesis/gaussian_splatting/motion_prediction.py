""" Predicts the per gaussian motion from the latent features."""

import torch
import torch.nn as nn
from jaxtyping import Float


class MotionPrediction(nn.Module):
    """ Predicts the per gaussian motion from the latent features."""

    def __init__(
        self,
        per_gaussian_latent_dim: int,
        hidden_dim: int = 64,
        scaling: float = 1e-3,
    ) -> None:
        """
        Args:
            per_gaussian_latent_dim (int): The dimension of the per-gaussian latents.
            hidden_dim (int): The hidden dimension of the MLP.
            scaling (float): The scaling factor.
        """
        super().__init__()
        self.scaling = scaling
        self.latent_mlp = nn.Sequential(
            nn.Linear(per_gaussian_latent_dim + 3 + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(
        self,
        means: Float[torch.Tensor, "n_gaussians 3"],
        orientations: Float[torch.Tensor, "n_gaussians 4"],
        per_gaussian_latents: Float[torch.Tensor, "n_gaussians latent_dim"],
    ) -> tuple[Float[torch.Tensor, "n_gaussians 3"], Float[torch.Tensor, "n_gaussians 4"]]:
        """
        Args:
            means (Float[torch.Tensor, 'n_gaussians 3']): The means of the gaussians.
            orientations (Float[torch.Tensor, 'n_gaussians 4']): The orientations of the gaussians.
            per_gaussian_latents (Float[torch.Tensor, 'n_gaussians latent_dim']): The per-gaussian latents.

        Returns:
            tuple[Float[torch.Tensor, 'n_gaussians 3'], Float[torch.Tensor, 'n_gaussians 4']]: The predicted means and orientations.
        """

        input = torch.cat([means, orientations, per_gaussian_latents], dim=1)
        output = self.latent_mlp(input)
        means_delta, orientations_delta = output.split([3, 4], dim=1)
        means = means + means_delta * self.scaling
        orientations = orientations + orientations_delta * self.scaling
        return means, orientations
