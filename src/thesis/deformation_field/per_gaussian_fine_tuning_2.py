""" Per Gaussian fine tuning 2."""

import torch
import torch.nn as nn
from jaxtyping import Float

from thesis.utils import quaternion_multiplication


class RotationAndScaleAdjustments(nn.Module):
    """
    Computes the rotation and scale adjustments for the gaussians. Should ideally approximate
    SLERP.
    """

    def __init__(
        self,
        gaussian_latent_dim: int,
        adjust_rotations: bool = True,
        adjust_scale: bool = True,
    ) -> None:
        """
        Args:
            adjust_rotations (bool): Whether to adjust the rotations.
            adjust_scale (bool): Whether to adjust the scale.
        """
        super().__init__()
        self.adjust_rotations = adjust_rotations
        self.adjust_scale = adjust_scale

        # means (3) + quats (4) translation (3) + rotation (12) + barycentric_weights (3)
        # + scale (3) + latent
        input_dim = 3 + 4 + 3 + 12 + 3 + 3 + gaussian_latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 7),
        )

    def forward(
        self,
        means: Float[torch.Tensor, "n_gaussians 3"],
        quats: Float[torch.Tensor, "n_gaussians 4"],
        translations: Float[torch.Tensor, "n_gaussians 3"],
        rotations: Float[torch.Tensor, "n_gaussians 3 4"],
        barycentric_weights: Float[torch.Tensor, "n_gaussians 3"],
        scales: Float[torch.Tensor, "n_gaussians 3"],
        per_gaussian_latent: Float[torch.Tensor, "n_gaussians per_gaussian_latent_dim"],
    ) -> tuple[Float[torch.Tensor, "n_gaussians 4"], Float[torch.Tensor, "n_gaussians 3"]]:
        """
        Args:
            means (Float[torch.Tensor, "n_gaussians 3"]): The position of the gaussians.
            quats (Float[torch.Tensor, "n_gaussians 4"]): The orientations of the gaussians.
            translations (Float[torch.Tensor, "n_gaussians 3"]): The translations of the gaussians.
            rotations (Float[torch.Tensor, "n_gaussians 4"]): The rotations of the gaussians.
        Returns:
            Float[torch.Tensor, "n_gaussians 4"]: The adjusted rotations.
        """

        n_gaussians = translations.shape[0]
        closest_rotation = rotations[:, 0]
        rotations = rotations.reshape(n_gaussians, -1)
        input_tensor = torch.cat([
            means, quats, translations, rotations, barycentric_weights, scales, per_gaussian_latent
        ],
                                 dim=-1)
        output_tensor = self.mlp(input_tensor)
        rotations = quaternion_multiplication(output_tensor[:, :4], closest_rotation)
        if self.adjust_scale:
            scales = output_tensor[:, 4:] * 1e-3 + scales
        return rotations, scales
