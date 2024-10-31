""" Flame deformation field generation. """

import torch
import torch.nn as nn
from jaxtyping import Float

from thesis.constants import CANONICAL_FLAME_PARAMS
from thesis.data_management.data_classes import UnbatchedFlameParams
from thesis.deformation_field.barycentric_weighting import (
    apply_barycentric_weights,
    compute_barycentric_weights,
)
from thesis.deformation_field.flame_knn import FlameKNN
from thesis.flame import FlameHead


class FlameDeformation(nn.Module):
    """ Queries the closes flame facet for motion vectors. """

    def __init__(
        self,
        window_size: int,
        latent_dim: int,
        audio_latent_dim: int = 8,
        hidden_dim: int = 32,
        canonical_params: UnbatchedFlameParams | tuple = CANONICAL_FLAME_PARAMS,
        use_audio_latents: bool = False,
        use_per_gaussian_latents: bool = False,
        use_flame_vertex_latents: bool = False,
        move_with_flame_directly: bool = True,
    ) -> None:
        """
        Args:
            window_size (int): The number of nearest neighbors.
            latent_dim (int): The dimension of the latent space.
            audio_latent_dim (int): The dimension of the audio latents.
            hidden_dim (int): The dimension of the hidden layers.
            canonical_params (UnbatchedFlameParams | tuple): The canonical flame parameters.
            use_audio_latents (bool): Whether to use audio latents.
            use_per_gaussian_latents (bool): Whether to use per-gaussian latents.
            use_flame_vertex_latents (bool): Whether to use flame vertex latents.
            move_with_flame_directly (bool): Whether to move the gaussians with the flame directly.
        """
        super().__init__()
        self.window_size = window_size
        self.use_audio_latents = use_audio_latents
        self.use_per_gaussian_latents = use_per_gaussian_latents
        self.use_flame_vertex_latents = use_flame_vertex_latents
        self.move_with_flame_directly = move_with_flame_directly

        self.flame_head = FlameHead()
        self.flame_knn = FlameKNN(k=3, canonical_params=canonical_params)
        canonical_params = UnbatchedFlameParams(*canonical_params)
        canonical_vertices = self.flame_head.forward(canonical_params).squeeze(
            0)  # (n_vertices, 3)
        self.register_buffer("canonical_vertices", canonical_vertices)

        decoder_input_dim = 0
        # flame vertex motions
        self.motion_encoder = nn.Sequential(
            nn.Linear(window_size * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        decoder_input_dim += hidden_dim
        # per gaussian position and orientation
        self.position_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),  # position + quaternion
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        decoder_input_dim += hidden_dim

        # audio latents
        if use_audio_latents:
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            decoder_input_dim += hidden_dim
        # per gaussian latents
        if use_per_gaussian_latents:
            self.per_gaussian_encoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            decoder_input_dim += hidden_dim
        # flame vertex latents
        if use_flame_vertex_latents:
            self.embedding = nn.Embedding(5143, latent_dim)
            self.flame_vertex_encoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            decoder_input_dim += hidden_dim

        # decoders
        self.motion_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )
        self.latent_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        means: Float[torch.Tensor, "n_gaussians 3"],
        quats: Float[torch.Tensor, "n_gaussians 4"],
        features: Float[torch.Tensor, "n_gaussians latent_dim"],
        flame_params: UnbatchedFlameParams,
        audio_latents: Float[torch.Tensor, "window_size audio_latent_dim"] | None = None,
        refresh_cache: bool = False,
    ) -> tuple[
            Float[torch.Tensor, "n_gaussians 3"],
            Float[torch.Tensor, "n_gaussians 4"],
            Float[torch.Tensor, "n_gaussians latent_dim"],
            float,
    ]:
        """
        Args:
            means (Float[torch.Tensor, 'n_gaussians 3']): The means of the gaussians.
            quats (Float[torch.Tensor, 'n_gaussians 4']): The orientations of the gaussians.
            features (Float[torch.Tensor, 'n_gaussians latent_dim']): The per-gaussian latents.
            flame_params (UnbatchedFlameParams): The flame parameters.
            audio_latents (Float[torch.Tensor, 'window_size 1024']): The audio latents.

        Returns:
            tuple: The updated means, quaternions and features. Also returns the cache hit rate.
        """
        # get the closest flame vertices and the vertex positions
        nn_indices, cache_hit_rate = self.flame_knn.forward(
            means,
            refresh_cache=True,  # TODO: doesn't seem to work as expected
        )  # (n_gaussians, k)
        vertex_motions = self.flame_head.forward(flame_params)  # (window_size, n_vertices, 3)
        current_vertex_motions = vertex_motions[self.window_size // 2]
        vertex_motions = torch.permute(vertex_motions, (1, 0, 2)).reshape(
            -1, self.window_size * 3)  # (n_vertices, window_size*3)

        # get the neighbor information of shape (n_gaussians, k, ...)
        neighbor_motion_vectors = self.flame_knn.gather(nn_indices, vertex_motions)
        current_vertex_motions = self.flame_knn.gather(nn_indices, current_vertex_motions)
        neighbor_positions = self.flame_knn.gather(nn_indices, self.canonical_vertices)
        if self.use_flame_vertex_latents:
            neighbor_latents = self.embedding(nn_indices.reshape(-1)).reshape(
                nn_indices.shape[0], nn_indices.shape[1], -1)

        # apply the barycentric weighting (n_gaussians, k, ...) -> (n_gaussians, ...)
        barycentric_weights = compute_barycentric_weights(means, neighbor_positions)
        neighbor_motion_vectors = apply_barycentric_weights(barycentric_weights,
                                                            neighbor_motion_vectors)
        current_vertex_motions = apply_barycentric_weights(barycentric_weights,
                                                           current_vertex_motions)
        if self.use_flame_vertex_latents:
            neighbor_latents = apply_barycentric_weights(barycentric_weights, neighbor_latents)

        # encode the motion vectors
        x = self.motion_encoder.forward(neighbor_motion_vectors)

        # encode the position and orientation
        position_encodings = self.position_encoder.forward(torch.cat((means, quats), dim=-1))
        x = torch.cat((x, position_encodings), dim=-1)

        # audio latents
        if self.use_audio_latents:
            audio_encodings = self.audio_encoder.forward(audio_latents)
            x = torch.cat((x, audio_encodings), dim=-1)
        # per gaussian latents
        if self.use_per_gaussian_latents:
            audio_encodings = self.per_gaussian_encoder.forward(features)
            x = torch.cat((x, audio_encodings), dim=-1)
        # flame vertex latents
        if self.use_flame_vertex_latents:
            vertex_latents = self.flame_vertex_encoder.forward(neighbor_latents)
            x = torch.cat((x, vertex_latents), dim=-1)

        # decode the motion vectors
        motion_vectors = self.motion_decoder.forward(x)
        # if self.move_with_flame_directly:
        #     means = means + current_vertex_motions
        means = means + motion_vectors[:, :3] * 1e-3
        quats = quats + motion_vectors[:, 3:] * 1e-2

        # decode the latents
        features = self.latent_decoder.forward(x)

        return means, quats, features, cache_hit_rate
