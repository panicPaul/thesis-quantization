"""
Extracts the SE(3) transformation from a mesh using the Kabsch-Umeyama algorithm.
"""

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from thesis.flame import FlameHead
from thesis.utils import rotation_matrix_to_quaternion

# =================================== Mesh utils =======================================


def faces_to_neighbors(
    mesh_faces: Int[torch.Tensor, "faces 3"],
) -> tuple[
        Int[torch.Tensor, "vertices max_neighbors"],
        Bool[torch.Tensor, "vertices max_neighbors"],
]:
    """
    Create a list of neighboring vertices for each vertex in the mesh.

    Args:
        mesh_faces (torch.Tensor): The mesh faces.

    Returns:
        A tuple containing:
        - Int[torch.Tensor, "vertices max_neighbors"]: The neighboring vertices.
        - Bool[torch.Tensor, "vertices max_neighbors"]: The mask for valid neighbors.
    """
    device = mesh_faces.device
    num_vertices = int(mesh_faces.max() + 1)

    # Create a list of edges
    edges = torch.cat([mesh_faces[:, [0, 1]], mesh_faces[:, [1, 2]], mesh_faces[:, [2, 0]]], dim=0)
    edges = torch.unique(torch.sort(edges, dim=1)[0], dim=0)

    # Create a dictionary to store neighbors for each vertex
    neighbors_dict = {i: set() for i in range(num_vertices)}

    # Populate the dictionary
    for edge in edges:
        v1, v2 = edge.tolist()
        neighbors_dict[v1].add(v2)
        neighbors_dict[v2].add(v1)

    # Create a list of neighbors for each vertex
    max_neighbors = max(len(neighbors) for neighbors in neighbors_dict.values())
    neighbors = torch.zeros((num_vertices, max_neighbors), dtype=torch.int64, device=device)
    mask = torch.zeros((num_vertices, max_neighbors), dtype=torch.bool, device=device)

    # Populate the tensor
    for i, neighbors_set in neighbors_dict.items():
        neighbors[i, :len(neighbors_set)] = torch.tensor(list(neighbors_set))
        mask[i, :len(neighbors_set)] = True

    return neighbors, mask


# ============================= Kabsch-Umeyama Algorithm ===============================


def unbatched_kabsch_umeyama(
    p: Float[torch.Tensor, "vertices 3"],
    q: Float[torch.Tensor, "vertices 3"],
    center_p: Float[torch.Tensor, "3"],
    center_q: Float[torch.Tensor, "3"],
) -> Float[torch.Tensor, "3 3"]:
    """
    Compute the optimal rotation between two point clouds using the Kabsch Umeyama
    algorithm.
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Args:
        p (torch.Tensor): The first point cloud.
        q (torch.Tensor): The second point cloud.

    Returns:
        The optimal rotation matrix.
    """

    p_centered = p - center_p
    q_centered = q - center_q

    covariance_matrix = p_centered.T @ q_centered

    u, _, v = torch.svd(covariance_matrix)

    # check for reflections
    det = torch.det(u @ v.T)
    d = torch.ones(3, device=p.device)
    d = torch.where(torch.abs(det - 1.0) > 1e-6, torch.cat([d[:2], det.unsqueeze(0)]), d)
    d = torch.diag(d)

    rotation_matrix = v @ d @ u.T

    return rotation_matrix


# =================================== SE(3) Extraction ================================


def _gather_neighbors(
    vertices: Float[torch.Tensor, "vertices 3"],
    neighbors: Int[torch.Tensor, "max_neighbors"],
    neighbors_mask: Bool[torch.Tensor, "max_neighbors"],
) -> Float[torch.Tensor, "max_neighbors 3"]:
    """
    Gather the neighboring vertices for a single vertex.

    Args:
        vertices (torch.Tensor): The vertices.
        neighbors (torch.Tensor): The neighboring vertices.

    Returns:
        The neighboring vertices.
    """
    return vertices[neighbors] * neighbors_mask.unsqueeze(-1)


def extract_se3_from_mesh(
    canonical_vertices: Float[torch.Tensor, "time vertices 3"],
    deformed_vertices: Float[torch.Tensor, "time vertices 3"],
    neighbors: Int[torch.Tensor, "vertices max_neighbors"],
    neighbors_mask: Bool[torch.Tensor, "vertices max_neighbors"],
) -> tuple[
        Float[torch.Tensor, "time vertices 4"],
        Float[torch.Tensor, "time vertices 3"],
]:
    """
    Extract the SE(3) transformation from a mesh deformation using the Kabsch-Umeyama
    algorithm. The SE(3) transformation is represented as a translation and a rotation
    for each vertex.

    NOTE: unlike conventional SE(3) transformation, we assume that we first
          apply the translation and then the rotation.

    Args:
        canonical_vertices (Float[torch.Tensor, 'time vertices 3']): The canonical
            vertices.
        deformed_vertices (Float[torch.Tensor, 'time vertices 3']): The deformed
            vertices.
        neighbors (Int[torch.Tensor, 'vertices max_neighbors']): The neighboring
            vertices.
        neighbors_mask (Bool[torch.Tensor, 'vertices max_neighbors']): The mask for
            valid neighbors.

    Returns:
        A tuple containing:
        - Float[torch.Tensor, 'time time vertices 4']: The rotation for each vertex.
        - Float[torch.Tensor, 'time time vertices 3']: The translation for each vertex.
    """

    # Vectorize gather_neighbors over time and time dimensions
    gather_neighbors_vertices = torch.vmap(
        _gather_neighbors, in_dims=(None, 0, 0))  # add vertex dimension

    gather_neighbors_batched = torch.vmap(
        gather_neighbors_vertices, in_dims=(0, None, None))  # add batch dimensions

    # Gather neighbors for canonical and deformed vertices
    n_neighborless_vertices = canonical_vertices.shape[1] - neighbors.shape[0]
    canonical_point_cloud = gather_neighbors_batched(canonical_vertices, neighbors, neighbors_mask)
    deformed_point_cloud = gather_neighbors_batched(deformed_vertices, neighbors, neighbors_mask)

    # Compute the rotation and translation for each vertex
    if n_neighborless_vertices > 0:
        rotation = torch.vmap(torch.vmap(unbatched_kabsch_umeyama,))(  # batch  # vertices
            canonical_point_cloud,
            deformed_point_cloud,
            canonical_vertices[:, :-n_neighborless_vertices],
            deformed_vertices[:, :-n_neighborless_vertices],
        )
        # pad the remaining vertices with identity
        identity_rotation = torch.eye(
            3, device=rotation.device).unsqueeze(0).unsqueeze(0)  # (time, vertices, 3, 3)
        identity_rotation = identity_rotation.expand(rotation.shape[0], n_neighborless_vertices, 3,
                                                     3)
        rotation = torch.cat([rotation, identity_rotation], dim=1)
    else:
        rotation = torch.vmap(torch.vmap(unbatched_kabsch_umeyama,))(  # batch  # vertices
            canonical_point_cloud,
            deformed_point_cloud,
            canonical_vertices,
            deformed_vertices,
        )
        rotation = rotation.contiguous()

    rotation = rotation_matrix_to_quaternion(rotation)
    translation = deformed_vertices - canonical_vertices
    assert rotation.shape[:2] == translation.shape[:2]
    return rotation, translation


# ================================ Flame SE(3) extraction ===============================


class FlameMeshSE3Extraction(nn.Module):

    def __init__(
        self,
        flame_head_model: FlameHead,
    ) -> None:
        """
        Args:
            flame_head_model (FlameHead): The flame head model.

        """
        super().__init__()
        mesh_faces = flame_head_model.faces
        neighbors, neighbors_mask = faces_to_neighbors(mesh_faces)
        self.register_buffer("neighbors", neighbors)
        self.register_buffer("neighbors_mask", neighbors_mask)

    def forward(
        self,
        canonical_vertices: Float[torch.Tensor, "batch vertices 3"],
        deformed_vertices: Float[torch.Tensor, "batch vertices 3"],
    ) -> tuple[
            Float[torch.Tensor, "batch vertices 4"],
            Float[torch.Tensor, "batch vertices 3"],
    ]:
        """
        Args:
            canonical_vertices (Float[torch.Tensor, 'batch vertices 3']): The canonical
                vertices.
            deformed_vertices (Float[torch.Tensor, 'batch vertices 3']): The deformed
                vertices.

        Returns:
            A tuple containing
            - Float[torch.Tensor, 'batch vertices 4']: The rotation for each
                vertex.
            - Float[torch.Tensor, 'batch vertices 3']: The translation for each
                vertex.
        """

        rotation, translation = extract_se3_from_mesh(
            canonical_vertices=canonical_vertices,
            deformed_vertices=deformed_vertices,
            neighbors=self.neighbors,
            neighbors_mask=self.neighbors_mask,
        )

        return rotation, translation
