""" Initialize splats for the Gaussian splatting algorithm. """

import torch
import torch.nn as nn

from thesis.data_management import (
    GaussianSplats,
    UnbatchedFlameParams,
    load_point_cloud,
)
from thesis.flame import FlameHead, FlameHeadWithInnerMouth

# from torch_geometric.nn import knn


def knn(x, y, k):
    """
    Compute k-nearest neighbors.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Input tensor.
        k (int): Number of nearest neighbors.

    Returns:
        torch.Tensor: Sender indices.
        torch.Tensor: Receiver indices.
    """
    x_norm = (x**2).sum(dim=-1, keepdim=True)
    y_norm = (y**2).sum(dim=-1, keepdim=True)
    dist = x_norm + y_norm.transpose(-2, -1) - 2.0 * x @ y.transpose(-2, -1)
    _, indices = dist.topk(k=k, dim=-1, largest=False)
    sender = torch.arange(x.shape[0], device=x.device).view(-1, 1).repeat(1, k).view(-1)
    receiver = indices.view(-1)
    return sender, receiver


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb-0.5) / C0


def random_initialization(
    num_splats: int,
    scene_scale: float = 0.2,
    initial_opacity: float = 0.1,
    feature_dim: int | None = None,
    colors_sh_degree: int = 3,
    initialize_spherical_harmonics: bool = False,
) -> GaussianSplats:
    """
    Randomly initialize splats.

    Args:
        num_splats (int): Number of splats.
        scene_scale (float): Scene scale.
        initial_opacity (float): Initial opacity.
        feature_dim (int | None): Feature dimension.
        initialize_spherical_harmonics (bool): Whether to initialize spherical
            harmonics.

    Returns:
        GaussianSplats: Randomly initialized splats.
    """

    means = nn.Parameter(scene_scale * (torch.rand((num_splats, 3)) * 2 - 1))
    quats = nn.Parameter(torch.rand((num_splats, 4)))
    opacities = nn.Parameter(torch.logit(torch.full((num_splats,), initial_opacity)))
    static_offsets = nn.Parameter(torch.zeros((num_splats, 3)))

    # Compute scales based on the average distance to the 3-nearest neighbors
    sender, receiver = knn(means, means, k=4)
    sender = sender.reshape(-1, 4)[:, 1:].reshape(-1)  # Remove self
    receiver = receiver.reshape(-1, 4)[:, 1:].reshape(-1)
    dist = torch.norm(means[sender] - means[receiver], dim=-1)
    dist_avg = dist.mean(dim=-1)
    scales = nn.Parameter(torch.log(dist_avg * scene_scale).unsqueeze(-1).repeat(num_splats, 3))

    splats = GaussianSplats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        static_offsets=static_offsets)

    if feature_dim is not None:
        splats["features"] = nn.Parameter(torch.randn((num_splats, feature_dim)))
    if initialize_spherical_harmonics:
        spherical_harmonics = torch.zeros((num_splats, (colors_sh_degree + 1)**2, 3))
        colors = torch.randn((num_splats, 3))
        sh0 = rgb_to_sh(colors)
        spherical_harmonics[:, 0, :] = sh0
        splats["sh0"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, :1, :]))
        splats["shN"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, 1:, :]))
    else:
        colors = torch.randn((num_splats, 3)).clamp(0, 1)
        colors = torch.logit(colors)
        splats['colors'] = nn.Parameter(colors)

    return splats


def point_cloud_initialization(
    num_splats: int,
    scene_scale: float = 0.2,
    initial_opacity: float = 0.1,
    feature_dim: int | None = None,
    colors_sh_degree: int = 3,
    initialize_spherical_harmonics: bool = False,
) -> GaussianSplats:
    """
    Initialize splats based on a point cloud.

    Args:
        num_splats (int): Number of splats.
        scene_scale (float): Scene scale.
        initial_opacity (float): Initial opacity.
        feature_dim (int | None): Feature dimension.
        initialize_spherical_harmonics (bool): Whether to initialize spherical
            harmonics.
    """
    means, colors = load_point_cloud()
    random_indices = torch.randint(0, len(means), (num_splats,))
    means = means[random_indices]

    means = nn.Parameter(means)
    quats = nn.Parameter(torch.rand((num_splats, 4)))
    opacities = nn.Parameter(torch.logit(torch.full((num_splats,), initial_opacity)))
    static_offsets = nn.Parameter(torch.zeros((num_splats, 3)))

    # Compute scales based on the average distance to the 3-nearest neighbors
    sender, receiver = knn(means, means, k=4)
    sender = sender.reshape(-1, 4)[:, 1:].reshape(-1)  # Remove self
    receiver = receiver.reshape(-1, 4)[:, 1:].reshape(-1)
    dist = torch.norm(means[sender] - means[receiver], dim=-1)
    dist_avg = dist.mean(dim=-1)
    scales = nn.Parameter(torch.log(dist_avg * scene_scale).unsqueeze(-1).repeat(num_splats, 3))

    splats = GaussianSplats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        static_offsets=static_offsets,
    )

    if feature_dim is not None:
        splats["features"] = nn.Parameter(torch.randn((num_splats, feature_dim)))
    if initialize_spherical_harmonics:
        spherical_harmonics = torch.zeros((num_splats, (colors_sh_degree + 1)**2, 3))
        sh0 = rgb_to_sh(colors)
        spherical_harmonics[:, 0, :] = sh0
        splats["sh0"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, :1, :]))
        splats["shN"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, 1:, :]))
    else:
        colors = torch.logit(colors[random_indices])
        splats['colors'] = nn.Parameter(colors)

    return splats


def flame_initialization(
    flame_params: UnbatchedFlameParams,
    flame_head: FlameHead,
    scene_scale: float = 0.2,
    initial_opacity: float = 0.1,
    feature_dim: int | None = None,
    colors_sh_degree: int = 3,
    initialize_spherical_harmonics: bool = False,
) -> GaussianSplats:
    """
    Initialize splats based on flame parameters

    Args:
        flame_params (UnbatchedFlameParams): Flame parameters.
        scene_scale (float): Scene scale.
        initial_opacity (float): Initial opacity.
        feature_dim (int | None): Feature dimension.
        colors_sh_degree (int): Degree of the spherical harmonics basis.
        initialize_spherical_harmonics (bool): Whether to initialize spherical
            harmonics.


    Returns:
        GaussianSplats: Flame parameters initialized splats.
    """

    flame_head.to(flame_params.expr.device)
    vertices = flame_head.forward(flame_params)
    means = vertices.squeeze(0)
    assert means.ndim == 2, 'Means should have shape (num_vertices, 3)'

    num_splats = means.shape[0]

    means = nn.Parameter(means)
    quats = nn.Parameter(torch.rand((num_splats, 4)))
    opacities = nn.Parameter(torch.logit(torch.full((num_splats,), initial_opacity)))
    static_offsets = nn.Parameter(torch.zeros((num_splats, 3)))

    # Compute scales based on the average distance to the 3-nearest neighbors
    sender, receiver = knn(means, means, k=4)
    sender = sender.reshape(-1, 4)[:, 1:].reshape(-1)  # Remove self
    receiver = receiver.reshape(-1, 4)[:, 1:].reshape(-1)
    dist = torch.norm(means[sender] - means[receiver], dim=-1)
    dist_avg = dist.mean(dim=-1)
    scales = nn.Parameter(torch.log(dist_avg * scene_scale).unsqueeze(-1).repeat(num_splats, 3))

    splats = GaussianSplats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        static_offsets=static_offsets,
    )

    if feature_dim is not None:
        splats["features"] = nn.Parameter(torch.randn((num_splats, feature_dim)))
    if initialize_spherical_harmonics:
        spherical_harmonics = torch.zeros((num_splats, (colors_sh_degree + 1)**2, 3))
        colors = torch.randn((num_splats, 3))
        sh0 = rgb_to_sh(colors)
        spherical_harmonics[:, 0, :] = sh0
        splats["sh0"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, :1, :]))
        splats["shN"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, 1:, :]))
    else:
        colors = torch.randn((num_splats, 3)).clamp(0, 1)
        colors = torch.logit(colors)
        splats['colors'] = nn.Parameter(colors)

    return splats


def inside_mouth_flame_initialization(
    flame_params: UnbatchedFlameParams,
    scene_scale: float = 0.2,
    initial_opacity: float = 0.1,
    feature_dim: int | None = None,
    colors_sh_degree: int = 3,
    initialize_spherical_harmonics: bool = False,
) -> GaussianSplats:
    """
    Initialize inside mouth region splats based on flame parameters

    Args:
        flame_params (UnbatchedFlameParams): Flame parameters.
        scene_scale (float): Scene scale.
        initial_opacity (float): Initial opacity.
        feature_dim (int | None): Feature dimension.
        colors_sh_degree (int): Degree of the spherical harmonics basis.
        initialize_spherical_harmonics (bool): Whether to initialize spherical
            harmonics.


    Returns:
        GaussianSplats: Flame parameters initialized splats.
    """

    flame_head = FlameHeadWithInnerMouth()
    flame_head.to(flame_params.expr.device)
    vertices = flame_head.forward(flame_params)
    inside_mouth_indices = flame_head.inner_mouth_indices
    eyeball_indices = flame_head.mask.v.eyeballs
    indices = torch.cat([inside_mouth_indices, eyeball_indices])
    vertices = vertices[:, indices]

    means = vertices.squeeze(0)
    assert means.ndim == 2, 'Means should have shape (num_vertices, 3)'

    num_splats = means.shape[0]

    means = nn.Parameter(means)
    quats = nn.Parameter(torch.rand((num_splats, 4)))
    opacities = nn.Parameter(torch.logit(torch.full((num_splats,), initial_opacity)))
    static_offsets = nn.Parameter(torch.zeros((num_splats, 3)))

    # Compute scales based on the average distance to the 3-nearest neighbors
    sender, receiver = knn(means, means, k=4)
    sender = sender.reshape(-1, 4)[:, 1:].reshape(-1)  # Remove self
    receiver = receiver.reshape(-1, 4)[:, 1:].reshape(-1)
    dist = torch.norm(means[sender] - means[receiver], dim=-1)
    dist_avg = dist.mean(dim=-1)
    scales = nn.Parameter(torch.log(dist_avg * scene_scale).unsqueeze(-1).repeat(num_splats, 3))

    splats = GaussianSplats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        static_offsets=static_offsets,
    )

    if feature_dim is not None:
        splats["features"] = nn.Parameter(torch.randn((num_splats, feature_dim)))
    if initialize_spherical_harmonics:
        spherical_harmonics = torch.zeros((num_splats, (colors_sh_degree + 1)**2, 3))
        colors = torch.randn((num_splats, 3))
        sh0 = rgb_to_sh(colors)
        spherical_harmonics[:, 0, :] = sh0
        splats["sh0"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, :1, :]))
        splats["shN"] = nn.Parameter(rgb_to_sh(spherical_harmonics[:, 1:, :]))
    else:
        colors = torch.randn((num_splats, 3)).clamp(0, 1)
        colors = torch.logit(colors)
        splats['colors'] = nn.Parameter(colors)

    return splats


def pre_trained_initialization(checkpoint: str) -> GaussianSplats:
    """
    Initialize splats from a pre-trained model.

    Args:
        checkpoint (str): Checkpoint path.

    Returns:
        GaussianSplats: Pre-trained splats.
    """
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    splats = {}
    for key, value in checkpoint['state_dict'].items():
        if 'splat' in key:
            k = key.split('.')[1]
            splats[k] = nn.Parameter(value)
    return GaussianSplats(**splats)
