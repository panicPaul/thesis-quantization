""" Barycentric weighting."""

import torch
from jaxtyping import Float


def project_points_to_triangle(
    query_positions: Float[torch.Tensor, "num_queries 3"],
    neighbor_positions: Float[torch.Tensor, "num_queries 3 3"],
) -> Float[torch.Tensor, "num_queries 3"]:
    """
    Project a set of query points to the nearest triangle.
    Args:
        query_positions (Float[torch.Tensor, "num_queries 3]): The query positions.
        neighbor_positions (Float[torch.Tensor, "num_vertices 3 3]): The vertex positions.
    Returns:
        (Float[torch.Tensor, "num_queries 3]): The projected positions.
    """
    # Calculate the normal of the triangle
    v0 = neighbor_positions[:, 1] - neighbor_positions[:, 0]
    v1 = neighbor_positions[:, 2] - neighbor_positions[:, 0]
    normal = torch.cross(v0, v1, dim=-1)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    # Project query points onto the triangle plane
    v = query_positions - neighbor_positions[:, 0]
    dist = torch.sum(v * normal, dim=-1, keepdim=True)
    projected = query_positions - dist*normal

    # Now check if the projected point is inside the triangle
    # using barycentric coordinates
    edge0 = neighbor_positions[:, 1] - neighbor_positions[:, 0]
    edge1 = neighbor_positions[:, 2] - neighbor_positions[:, 0]
    v2 = projected - neighbor_positions[:, 0]

    # Compute dot products
    d00 = torch.sum(edge0 * edge0, dim=-1)
    d01 = torch.sum(edge0 * edge1, dim=-1)
    d11 = torch.sum(edge1 * edge1, dim=-1)
    d20 = torch.sum(v2 * edge0, dim=-1)
    d21 = torch.sum(v2 * edge1, dim=-1)

    # Compute barycentric coordinates
    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    u = 1.0 - v - w

    # Check if point is inside triangle
    is_inside = (u >= 0) & (v >= 0) & (w >= 0)

    # If point is outside, project onto nearest edge or vertex
    def project_to_segment(point, segment_start, segment_end):
        segment = segment_end - segment_start
        t = torch.sum(
            (point-segment_start) * segment, dim=-1) / torch.sum(
                segment * segment, dim=-1)
        t = torch.clamp(t, 0.0, 1.0)
        return segment_start + t.unsqueeze(-1) * segment

    edge_projections = []
    # Project onto each edge
    edges = [(neighbor_positions[:, 0], neighbor_positions[:, 1]),
             (neighbor_positions[:, 1], neighbor_positions[:, 2]),
             (neighbor_positions[:, 2], neighbor_positions[:, 0])]

    for start, end in edges:
        proj = project_to_segment(projected, start, end)
        dist = torch.norm(proj - projected, dim=-1)
        edge_projections.append((dist, proj))

    # Find nearest edge projection
    distances = torch.stack([d for d, _ in edge_projections], dim=-1)
    min_edge_idx = torch.argmin(distances, dim=-1)

    # Select the nearest edge projection for points outside the triangle
    outside_proj = torch.stack([p for _, p in edge_projections], dim=1)
    batch_idx = torch.arange(len(min_edge_idx))
    nearest_edge_proj = outside_proj[batch_idx, min_edge_idx]

    # Return inside points unchanged, outside points projected to nearest edge
    return torch.where(is_inside.unsqueeze(-1), projected, nearest_edge_proj)


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
    query_positions = project_points_to_triangle(query_positions, neighbor_positions)
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
