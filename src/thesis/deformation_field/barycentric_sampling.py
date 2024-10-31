""" Barycentric sampling."""

import torch
from jaxtyping import Float, Int


def barycentric_weights(
    query_points: Float[torch.Tensor, "num_queries 3"],
    vertices: Float[torch.Tensor, "num_vertices 3"],
    nn_indices: Int[torch.Tensor, "num_queries 3"],
) -> Float[torch.Tensor, "num_queries 3"]:
    """
    Calculate the barycentric weights for a set of query points and their nearest
    neighbors.
    """

    # Get the three nearest neighbors.
    nn_vertices = vertices[nn_indices]

    # Calculate barycentric weights
    v0 = nn_vertices[:, 1] - nn_vertices[:, 0]
    v1 = nn_vertices[:, 2] - nn_vertices[:, 0]
    v2 = query_points - nn_vertices[:, 0]

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


def compute_barycentric_weights(
    query_points: Float[torch.Tensor, "batch num_queries 3"],
    vertices: Float[torch.Tensor, "batch num_vertices 3"],
    nn_indices: Int[torch.Tensor, "batch num_queries 3"],
) -> Float[torch.Tensor, "batch num_queries 3"]:
    """
    Calculate the barycentric weights for a set of query points and their nearest
    neighbors.
    """

    return torch.vmap(barycentric_weights)(query_points, vertices, nn_indices)


def gather_value(
    indices: Int[torch.Tensor, "num_queries 3"],
    values: Float[torch.Tensor, "num_vertices ..."],
    barycentric_weights: Float[torch.Tensor, "num_queries 3"],
) -> Float[torch.Tensor, "num_queries ..."]:
    """
    Gather values from a tensor using indices. Do NOT use this function with rotations
    as it will not interpolate them correctly, use `unbatched_gather_rotation` instead.

    Args:
        indices (Int[torch.Tensor, "num_queries 3]): The indices.
        values (Float[torch.Tensor, "num_vertices ...]): The values.
        barycentric_weights (Float[torch.Tensor, "num_queries 3]): The barycentric
            weights.

    Returns:
        (Float[torch.Tensor, "num_queries ...]): The gathered values.
    """

    # Gather the values using the indices
    gathered_values = values[indices]  # Shape: (num_queries, 3, ...)

    # Apply barycentric weights
    weighted_values = gathered_values * barycentric_weights.unsqueeze(-1)

    # Sum along the barycentric dimension
    return torch.sum(weighted_values, dim=1)
