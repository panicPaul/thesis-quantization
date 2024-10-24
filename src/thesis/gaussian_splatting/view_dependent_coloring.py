""" View dependent color MLP. Modified version from GSplat. """

import torch
from jaxtyping import Float
from torch import nn


class ViewDependentColorMLP(nn.Module):
    """Simplified version of the GSplat appearance optimization module."""

    def __init__(
        self,
        feature_dim: int,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ) -> None:
        """
        Args:
            feature_dim (int): Dimension of the input features.
            sh_degree (int): Degree of the spherical harmonics basis.
            mlp_width (int): Width of the MLP.
            mlp_depth (int): Depth of the MLP.
        """

        super().__init__()
        self.sh_degree = sh_degree
        layers = []
        layers.append(torch.nn.Linear(feature_dim + (sh_degree + 1)**2, mlp_width))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self,
        features: Float[torch.Tensor, "n_splats feature_dim"],
        means: Float[torch.Tensor, "n_splats 3"],
        cam_2_world: Float[torch.Tensor, "cam 4 4"],
        cur_sh_degree: int,
    ) -> Float[torch.Tensor, "C N 3"]:
        """Adjust appearance based on embeddings.

        Args:
            features (Tensor): per gaussian features, shape: `(cam, n_splats, feature_dim)`.
            means (Tensor): per gaussian means, shape: `(cam, n_splats, 3)`.
            cam_2_world (Tensor): camera to world transformation, shape: `(cam, 4, 4)`.
            cur_sh_degree (int): current SH degree.


        Returns:
            (Tensor): updated colors. Shape: (cam, n_splats, 3).
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C = cam_2_world.shape[0]
        N = means.shape[0]

        # Compute view directions
        view_directions = means[None, :, :] - cam_2_world[:, None, :3, 3]

        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        view_directions = nn.functional.normalize(view_directions, dim=-1)  # [C, N, 3]
        num_bases_to_use = (cur_sh_degree + 1)**2
        num_bases = (self.sh_degree + 1)**2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, view_directions)
        # Get colors
        h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors
