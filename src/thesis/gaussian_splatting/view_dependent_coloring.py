""" View dependent color MLP. Modified version from GSplat. """

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import nn


def inverse_sigmoid(
        x: Float[torch.Tensor, "cam n_splats 3"]) -> Float[torch.Tensor, "cam n_splats 3"]:
    """Inverse sigmoid function."""
    x = torch.clamp(x, 1e-7, 1 - 1e-7)
    return torch.log(x / (1-x))


class ViewDependentColorMLP(nn.Module):
    """Simplified version of the GSplat appearance optimization module."""

    def __init__(
        self,
        feature_dim: int,
        num_cameras: int,
        embed_dim: int = 16,
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
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(num_cameras, embed_dim)
        layers = []
        layers.append(torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1)**2, mlp_width))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self,
        features: Float[torch.Tensor, "n_splats feature_dim"],
        camera_ids: Int[torch.Tensor, "cam"] | None,
        means: Float[torch.Tensor, "n_splats 3"],
        colors: Float[torch.Tensor, "n_splats 3"],
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
        # Camera embeddings
        if camera_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(camera_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = means[None, :, :] - cam_2_world[:, None, :3, 3]
        dirs = nn.functional.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (cur_sh_degree + 1)**2
        num_bases = (self.sh_degree + 1)**2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        # colors = nn.functional.sigmoid(self.color_head(h)) + colors
        color_adjustment = self.color_head(h)
        colors = colors + inverse_sigmoid(color_adjustment)
        return nn.functional.sigmoid(colors)


class LearnableColorCorrection(nn.Module):
    """ Learnable color correction matrix for each camera. """

    def __init__(self, num_cameras: int) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.color_correction = nn.Parameter(torch.eye(3, 3).repeat(num_cameras, 1, 1))

    def forward(
        self,
        camera_indices: Int[torch.Tensor, "cam"],
        image: Float[torch.Tensor, "cam h w 3"],
    ) -> Float[torch.Tensor, "cam h w 3"]:
        """
        Apply the color correction matrix to the image.

        Args:
            camera_indices: Camera indices. Has shape (cam,).
            image: Image tensor. Has shape (cam, height, width, 3).
        """

        correction_matrices = self.color_correction[camera_indices]
        cam, height, width, _ = image.shape
        reshaped_image = rearrange(image, "cam h w c -> cam (h w) c")
        # Apply color correction: (cam, h*w, 3) @ (cam, 3, 3) -> (cam, h*w, 3)
        corrected_image = torch.bmm(reshaped_image, correction_matrices)
        corrected_image = rearrange(
            corrected_image,
            "cam (h w) c -> cam h w c",
            cam=cam,
            h=height,
            w=width,
        )

        return corrected_image


class LearnableShader(nn.Module):
    """ Learnable shader / illumination field, depending on the head position"""

    def __init__(self, feature_dim: int) -> None:
        """
        Args:
            feature_dim (int): Dimension of the input features.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 3 + 4 + 3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        spatial_position: Float[torch.Tensor, "n_splats 3"],
        orientation: Float[torch.Tensor, "n_splats 4"],
        features: Float[torch.Tensor, "n_splats feature_dim"],
        colors: Float[torch.Tensor, "cam n_splats 3"],
    ) -> Float[torch.Tensor, "cam n_splats 3"]:
        """
        Args:
            spatial_position (Tensor): per gaussian spatial position after deformation,
                shape: `(n_splats, 3)`.
            orientation (Tensor): per gaussian orientation after deformation,
                shape: `(n_splats, 4)`.
            features (Tensor): per gaussian features, shape: `(n_splats, feature_dim)`.
            colors (Tensor): per gaussian colors, shape: `(cam, n_splats, 3)`.

        Returns:
            (Tensor): updated colors. Shape: (n_splats, 3).
        """

        spatial_position = spatial_position.unsqueeze(0).expand(colors.shape[0], -1, -1)
        orientation = orientation.unsqueeze(0).expand(colors.shape[0], -1, -1)
        features = features.unsqueeze(0).expand(colors.shape[0], -1, -1)
        h = torch.cat([spatial_position, orientation, features, colors], dim=-1)
        shading = self.mlp.forward(h)
        colors = inverse_sigmoid(colors) + shading
        return nn.functional.sigmoid(colors)
