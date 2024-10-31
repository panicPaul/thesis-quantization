""" Predict latents based on audio features. """

import torch
import torch.nn as nn
from jaxtyping import Float
from vector_quantize_pytorch import FSQ


class AudioLatents(nn.Module):
    """ Extreme bottleneck model to learn the audio latents. """

    def __init__(
        self,
        levels: list[int] = [5, 5],
        audio_latent_dim: int = 8,
        window_size: int = 9,
    ) -> None:
        """
        Args:
            audio_latent_dim (int): The dimension of the audio latents.
        """
        super().__init__()
        self.audio_latent_dim = audio_latent_dim
        self.initial_projection = nn.Linear(1024, 128)

        self.audio_latent_encoder = nn.Sequential(
            nn.Linear(128 * window_size, 128),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fsq = FSQ(levels=levels)
        self.audio_latent_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, audio_latent_dim),
        )

    def forward(
        self,
        audio_features: Float[torch.Tensor, 'window_size 1024'],
    ) -> Float[torch.Tensor, 'audio_latent_dim']:
        """
        Args:
            audio_features (torch.Tensor): The audio features. Shape: (window_size, 1024)

        Returns:
            torch.Tensor: The audio latents. Shape: (audio_latent_dim, )
        """
        audio_features = audio_features.unsqueeze(0)
        x = self.initial_projection.forward(audio_features)
        x = x.flatten().unsqueeze(0)
        x = self.audio_latent_encoder.forward(x)
        x = self.fsq(x)  # TODO: probably need to reshape it here
        x = self.audio_latent_decoder.forward(x)
        return x
