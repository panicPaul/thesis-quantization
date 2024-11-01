""" Small MLP that fine-tunes the position and orientation of the gaussians. """

import torch
import torch.nn as nn
from jaxtyping import Float


class PerGaussianFineTuning(nn.Module):
    """ Small MLP that fine-tunes the position and orientation of the gaussians. """

    def __init__(
        self,
        use_audio: bool,
        use_motion_history: bool,
        window_size: int,
        audio_latent_dim: int,
        per_gaussian_latent_dim: int,
        translation_scaling: float = 1e-4,
        rotation_scaling: float = 1e-2,
    ) -> None:
        """
        Args:
            use_audio (bool): Whether to use audio features.
            use_motion_history (bool): Whether to use motion history features.
            window_size (int): The size of the motion history window.
            audio_latent_dim (int): The dimension of the audio latent representation.
            per_gaussian_latent_dim (int): The dimension of the per gaussian latent representation.
            translation_scaling (float): The scaling factor for the translation.
            rotation_scaling (float): The scaling factor for the rotation.
        """
        super().__init__()
        self.use_audio = use_audio
        self.use_motion_history = use_motion_history
        self.translation_scaling = translation_scaling
        self.rotation_scaling = rotation_scaling

        input_dim = per_gaussian_latent_dim
        if use_audio:
            input_dim += audio_latent_dim
        if use_motion_history:
            input_dim += window_size * 3

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 7),
        )

    def forward(
        self,
        per_gaussian_latent: Float[torch.Tensor, "n_gaussians per_gaussian_latent_dim"],
        audio_latent: Float[torch.Tensor, "n_gaussians audio_latent_dim"] | None = None,
        motion_history: Float[torch.Tensor, "window_size n_gaussian 3"] | None = None,
    ) -> tuple[Float[torch.Tensor, "n_gaussians 4"], Float[torch.Tensor, "n_gaussians 3"]]:
        """
        Args:
            per_gaussian_latent (torch.Tensor): The latent representation of the gaussians,
                shape (n_gaussians, per_gaussian_latent_dim).
            audio_latent (torch.Tensor): The latent representation of the audio,
                shape (n_gaussians, audio_latent_dim).
            motion_history (torch.Tensor): The motion history features,
                shape (window_size, n_gaussians, 3).

        Returns:
            tuple: The rotation and translation of the gaussians.
        """

        x = per_gaussian_latent
        if self.use_audio:
            x = torch.cat([x, audio_latent], dim=-1)
        if self.use_motion_history:
            motion_history = motion_history.permute(1, 0, 2)
            motion_history = motion_history.reshape(per_gaussian_latent.shape[0], -1)
            x = torch.cat([x, motion_history], dim=-1)

        x = self.mlp(x)
        translation = x[:, :3] * self.translation_scaling
        rotation = x[:, 3:] * self.rotation_scaling
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)

        return rotation, translation
