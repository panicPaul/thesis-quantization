""" Use the FLAME model to generate a deformation field from latents. """

from typing import Literal

import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Float, Int

from thesis.data_management import UnbatchedFlameParams
from thesis.flame import FlameHead


class FlameLatents(nn.Module):
    """ Learnable FLAME vertex latents. """

    def __init__(
        self,
        flame_latent_dim: int,
        window_size: int,
        canonical_flame_params: UnbatchedFlameParams,
        receptive_field: int = 9,
    ) -> None:
        """
        Args:
            latent_dim (int): The dimension of the latent space.
            window_size (int): The window size.
            canonical_flame_params (UnbatchedFlameParams): The canonical FLAME parameters.
            receptive_field (int): The receptive field, i.e. the k-nn parameter.
        """
        super().__init__()
        self.flame_head = FlameHead()
        self.flame_head.to(canonical_flame_params.expr.device)
        n_vertices = 5143
        self.latents = nn.Embedding(n_vertices, flame_latent_dim)
        self.window_size = window_size
        self.receptive_field = receptive_field

        canonical_vertex_positions = self.flame_head.forward(canonical_flame_params)
        canonical_vertex_positions = repeat(
            canonical_vertex_positions,
            "1 n_vertices 3 -> window_size n_vertices 3",
            window_size=window_size)
        self.register_buffer(
            "canonical_vertex_positions",
            canonical_vertex_positions,
        )

        # caching parameters
        self.cached_recomputation_distance = None
        self.cached_position = None

    def knn(
        self,
        means: Float[torch.Tensor, 'n_gaussians 3'],
    ) -> Int[torch.Tensor, 'n_gaussians k']:
        """
        Compute the k+1-nearest neighbors of the input. If the cache exists, we
        first check if the means have moved more than the distance to the next
        nearest neighbor. If not, we return the cached neighbors.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The k-nearest neighbors.
        """
        if self.cached_position is not None and self.cached_position.shape[0] != means.shape[0]:
            # Recompute index after densification
            self.cached_position = None
            self.cached_recomputation_distance = None
        raise NotImplementedError

    def forward(
        self, means: Float[torch.Tensor, 'n_gaussians 3'], flame_params: UnbatchedFlameParams
    ) -> tuple[
            Float[torch.Tensor, 'n_gaussians k motions'],
            Float[torch.Tensor, 'n_gaussians k latents'],
            Float[torch.Tensor, 'n_gaussians k 1'],
    ]:
        """
        Args:
            means (torch.Tensor): The means of the Gaussians. Shape: (n_gaussians, 3)
            flame_params (UnbatchedFlameParams): The FLAME parameters. Shape: (n_gaussians, 3)

        Returns:
            tuple: the motion for the k-nearest neighbors over the window, the
                latents for the k-nearest neighbors and the distances to the k-nearest neighbors.
        """
        # TODO: maybe distance to the vertices as well? would be a learnable convolution then
        vertex_positions = self.flame_head.forward(flame_params)
        vertex_motion = vertex_positions - self.canonical_vertex_positions
        knn_indices = self.knn(means)
        # TODO: gather the motion and latents, compute the distances
        raise NotImplementedError


class LatentProcessing(nn.Module):
    """ Processes the latents to get a new per-gaussian latent. """

    def __init__(
        self,
        flame_latent_dim: int,
        window_size: int,
        hidden_dim: int,
        use_audio_latents: bool,
        audio_latent_dim: int,
        per_gaussian_latent_dim: int,
        canonical_flame_params: UnbatchedFlameParams,
        receptive_field: int = 9,
        gating_function: Literal['glu', 'swiglu'] = 'swiglu',
    ) -> None:
        """
        Args:

        """
        super().__init__()
        self.gating_function = gating_function
        self.use_audio_latents = use_audio_latents
        if use_audio_latents:
            self.audio_latent_encoder = AudioLatents()

        self.flame_latent_encoder = FlameLatents(
            latent_dim=flame_latent_dim,
            window_size=window_size,
            receptive_field=receptive_field,
            canonical_flame_params=canonical_flame_params,
        )
        gating_dim = 3*window_size + flame_latent_dim + 1
        self.gating_mlp = nn.Sequential(
            nn.Linear(per_gaussian_latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gating_dim),
        )
        processing_dim = (
            gating_dim + per_gaussian_latent_dim if not use_audio_latents else gating_dim
            + per_gaussian_latent_dim + audio_latent_dim)
        self.latent_processing_mlp = nn.Sequential(
            nn.Linear(processing_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, per_gaussian_latent_dim),
        )

    def forward(
        self,
        means: Float[torch.Tensor, 'n_gaussians 3'],
        per_gaussian_latents: Float[torch.Tensor, 'n_gaussians per_gaussian_latent_dim'],
        flame_params: UnbatchedFlameParams,
        audio_features: Float[torch.Tensor, 'window_size 1024'] | None = None,
    ) -> Float[torch.Tensor, 'n_gaussians per_gaussian_latent_dim']:
        """
        Args:
            means (torch.Tensor): The means of the Gaussians. Shape: (n_gaussians, 3)
            per_gaussian_latents (torch.Tensor): The per-gaussian latents. Shape:
                (n_gaussians, per_gaussian_latent_dim)
            flame_params (UnbatchedFlameParams): The FLAME parameters. Shape: (n_gaussians, 3)
            audio_features (torch.Tensor): The audio features. Shape: (window_size, 1024)

        Returns:
            torch.Tensor: The per-gaussian latent. Shape: (n_gaussians, per_gaussian_latent_dim)
        """

        motions, latents, distances = self.flame_latent_encoder.forward(means, flame_params)
        # motions: (n_gaussians, k, 3)
        neighbor_vector = torch.cat((motions, latents, distances), dim=-1)
        gating_weights = self.gating_mlp.forward(per_gaussian_latents)
        gating_weights = gating_weights.unsqueeze(1).expand_as(neighbor_vector)
        match self.gating_function:
            case 'glu':
                gated_vector = neighbor_vector * nn.functional.sigmoid(gating_weights)
            case 'swiglu':
                gated_vector = neighbor_vector * nn.functional.silu(gating_weights)
        aggregated_latents = torch.sum(gated_vector, dim=1)

        if self.use_audio_latents:
            audio_latents = self.audio_latent_encoder.forward(audio_features)
            aggregated_latents = torch.cat((aggregated_latents, audio_latents), dim=-1)
        aggregated_latents = torch.cat((aggregated_latents, per_gaussian_latents), dim=-1)

        return self.latent_processing_mlp.forward(aggregated_latents)
