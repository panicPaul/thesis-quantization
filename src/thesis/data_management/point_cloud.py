""" Utility file to load canonical point cloud."""

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float

from thesis.constants import (
    CANONICAL_PCD,
    DEFAULT_SE3_ROTATION,
    DEFAULT_SE3_TRANSLATION,
)
from thesis.utils import apply_se3_to_point


def load_point_cloud(
    filename: str = CANONICAL_PCD
) -> tuple[Float[torch.Tensor, "N 3"], Float[torch.Tensor, "N 3"]]:
    """Load the canonical point cloud."""

    pcd = o3d.io.read_point_cloud(filename)
    points = torch.tensor(np.asarray(pcd.points)).float()
    inverse_rotation = DEFAULT_SE3_ROTATION.T
    inverse_translation = -DEFAULT_SE3_ROTATION.T @ DEFAULT_SE3_TRANSLATION
    points = apply_se3_to_point(inverse_rotation, inverse_translation, points)

    colors = torch.tensor(np.asarray(pcd.colors)).float()

    return points, colors
