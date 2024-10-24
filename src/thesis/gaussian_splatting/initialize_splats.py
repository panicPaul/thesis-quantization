""" Initialize splats for the Gaussian splatting algorithm. """

import torch
import torch.nn as nn
from torch_geometric.nn import knn

from thesis.data_management import GaussianSplats


def random_initialization(
    num_splats: int,
    scene_scale: float = 0.2,
    initial_opacity: float = 0.1,
    feature_dim: int | None = None,
    colors_sh_degree: int = 3,
    initialize_colors: bool = False,
) -> GaussianSplats:
    """
    Randomly initialize splats.

    Args:
        num_splats (int): Number of splats.
        scene_scale (float): Scene scale.
        initial_opacity (float): Initial opacity.
        feature_dim (int | None): Feature dimension.
        initialize_colors (bool): Whether to initialize colors. Not necessary when
            using the view-dependent color module.


    Returns:
        GaussianSplats: Randomly initialized splats.
    """

    means = nn.Parameter(scene_scale * (torch.rand((num_splats, 3)) * 2 - 1))
    quats = nn.Parameter(torch.rand((num_splats, 4)))
    opacities = nn.Parameter(torch.logit(torch.full((num_splats,), initial_opacity)))

    # Compute scales based on the average distance to the 3-nearest neighbors
    sender, receiver = knn(means, means, k=4)
    sender = sender.reshape(-1, 4)[:, 1:].reshape(-1)  # Remove self
    receiver = receiver.reshape(-1, 4)[:, 1:].reshape(-1)
    dist = torch.norm(means[sender] - means[receiver], dim=-1)
    dist_avg = dist.mean(dim=-1)
    scales = nn.Parameter(torch.log(dist_avg * scene_scale).unsqueeze(-1).repeat(1, 3))

    splats = GaussianSplats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
    )

    if feature_dim is not None:
        splats["features"] = nn.Parameter(torch.randn((num_splats, feature_dim)))
    if initialize_colors:
        splats["colors"] = nn.Parameter(torch.randn((num_splats, (colors_sh_degree + 1)**2), 3))

    return splats
