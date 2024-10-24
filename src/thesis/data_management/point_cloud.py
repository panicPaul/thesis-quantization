""" Utility file to load canonical point cloud."""

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float

from thesis.constants import CANONICAL_PCD


def load_point_cloud(
    filename: str = CANONICAL_PCD
) -> tuple[Float[torch.Tensor, "N 3"], Float[torch.Tensor, "N 3"]]:
    """Load the canonical point cloud."""

    pcd = o3d.io.read_point_cloud(filename)
    points = torch.tensor(np.asarray(pcd.points)).float()
    colors = torch.tensor(np.asarray(pcd.colors)).float()
    return points, colors
