""" Directly predict the latents conditioned on the prior. """

from typing import Literal

import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Float

from thesis.data_management import UnbatchedFlameParams
from thesis.deformation_field.audio_latents import AudioLatents


class DirectPrediction(nn.Module):
    """
    We directly predict the latents based on the canonical position and per-gaussian
    latent features
    """

    def __init__(
        self,
        window_size: int,
        per_gaussian_latent_dim: int,
        hidden_dim: int = 64,
        levels: list[int] = [8, 8],
        audio_latent_dim: int = 8,
        use_audio_latents: bool = False,
        use_per_gaussian_latents: bool = False,
        use_flame_params: bool = False,
        flame_projection_dim: int = 16,
    ) -> None:
        """
        Args:
            hidden_dim (int): The hidden dimension of the MLP.
            use_audio_latents (bool): Whether to use audio latents.
            levels (list[int]): The levels for the FSQ.
            audio_latent_dim (int): The dimension of the audio latents.
            window_size (int): The window size.
            per_gaussian_latent_dim (int): The dimension of the per-gaussian latents.
            flame_projection_dim (int): The dimension of the flame projection.
        """
        super().__init__()
        self.use_audio_latents = use_audio_latents
        self.use_per_gaussian_latents = use_per_gaussian_latents
        self.use_flame_params = use_flame_params

        input_dim = 3
        if use_flame_params:
            self.input_projection = nn.Linear(103, flame_projection_dim)
            self.merge_projection = nn.Linear(flame_projection_dim * window_size,
                                              flame_projection_dim)
            input_dim += flame_projection_dim

        if use_audio_latents:
            self.audio_latent_encoder = AudioLatents(
                levels=levels, audio_latent_dim=audio_latent_dim, window_size=window_size)
            input_dim += audio_latent_dim

        if use_per_gaussian_latents:
            input_dim += per_gaussian_latent_dim

        self.latent_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, per_gaussian_latent_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        means: Float[torch.Tensor, 'n_gaussians 3'],
        per_gaussian_latents: Float[torch.Tensor, 'n_gaussians per_gaussian_latent_dim']
        | None = None,
        flame_params: UnbatchedFlameParams | None = None,
        audio_features: Float[torch.Tensor, 'window_size 1024'] | None = None,
    ) -> Float[torch.Tensor, 'n_gaussians per_gaussian_latent_dim']:
        """
        Args:
            means (torch.Tensor): The positions of the Gaussians. Shape: (n_gaussians, 3)
            per_gaussian_latents (torch.Tensor): The per-gaussian latents. Shape:
                (n_gaussians, per_gaussian_latent_dim)
            flame_params (UnbatchedFlameParams): The canonical flame parameters.
            audio_features (torch.Tensor): The audio features. Shape: (window_size, 1024)

        Returns:
            torch.Tensor: The predicted per-gaussian latents. Shape:
            (n_gaussians, per_gaussian_latent_dim)
        """
        x = means
        if self.use_flame_params:
            flame_params = torch.cat([flame_params.expr, flame_params.jaw],
                                     dim=-1)  # (window, 103)
            flame_params = self.input_projection(flame_params)  # (window, 16)
            flame_params = nn.functional.silu(flame_params)
            flame_params = flame_params.flatten().unsqueeze(0)  # (1, window*16)
            flame_params = self.merge_projection(flame_params)  # (1, 16)
            flame_params = nn.functional.silu(flame_params)
            flame_params = repeat(
                flame_params, 'b d -> (b n) d', n=means.shape[0])  # (n_gaussians, 16)
            x = torch.cat([x, flame_params], dim=-1)

        if self.use_audio_latents:
            audio_latents = self.audio_latent_encoder(audio_features)  # (audio_latent_dim, )
            audio_latents = audio_latents.unsqueeze(0)  # (1, audio_latent_dim)
            audio_latents = repeat(audio_latents, 'b d -> (b n) d', n=means.shape[0])
            x = torch.cat([x, audio_latents], dim=-1)

        if self.use_per_gaussian_latents:
            x = torch.cat([x, per_gaussian_latents], dim=-1)

        x = self.latent_mlp(x)  # (n_gaussians, per_gaussian_latent_dim)
        return x
