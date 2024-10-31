""" Barycentric weighting."""

import torch
from jaxtyping import Float


def compute_barycentric_weights(
    query_positions: Float[torch.Tensor, "num_queries 3"],
    neighbor_positions: Float[torch.Tensor, "num_vertices 3 3"],
) -> Float[torch.Tensor, "num_queries 3"]:
    """
    Calculate the barycentric weights for a set of query points and their nearest
    vertices.

    Args:
        query_positions (Float[torch.Tensor, "num_queries 3]): The query positions.
        vertex_positions (Float[torch.Tensor, "num_vertices 3]): The vertex positions.
        nn_indices (Int[torch.Tensor, "num_queries 3]): The nearest neighbor indices.

    Returns:
        (Float[torch.Tensor, "num_queries 3]): The barycentric weights.
    """

    # Calculate barycentric weights
    nn_vertices = neighbor_positions
    v0 = nn_vertices[:, 1] - nn_vertices[:, 0]
    v1 = nn_vertices[:, 2] - nn_vertices[:, 0]
    v2 = query_positions - nn_vertices[:, 0]

    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)

    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    u = 1.0 - v - w

    barycentric_weights = torch.stack([u, v, w], dim=-1)
    # In case the point is outside the triangle, clamp the weights to [0, 1]
    barycentric_weights = torch.clamp(barycentric_weights, min=0.0, max=1.0)
    barycentric_weights = barycentric_weights / torch.sum(
        barycentric_weights, dim=-1, keepdim=True)

    return barycentric_weights


def apply_barycentric_weights(
    barycentric_weights: Float[torch.Tensor, "num_queries 3"],
    neighbor_features: Float[torch.Tensor, "num_queries 3 ..."],
) -> Float[torch.Tensor, "num_queries ..."]:
    """
    Apply the barycentric weights to the neighbor features.

    Args:
        barycentric_weights (Float[torch.Tensor, "num_queries 3]): The barycentric weights.
        neighbor_features (Float[torch.Tensor, "num_queries 3 ...]): The neighbor features.

    Returns:
        (Float[torch.Tensor, "num_queries ...]): The weighted features.
    """
    feature_dims = neighbor_features.ndim - 2
    for _ in range(feature_dims):
        barycentric_weights = barycentric_weights.unsqueeze(-1)
    return torch.sum(barycentric_weights * neighbor_features, dim=1)
