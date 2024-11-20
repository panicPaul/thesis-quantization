""" Deformation field that fine tunes the per gaussian means and quaternions. """

import torch
from einops import repeat
from jaxtyping import Float
from torch import nn

from thesis.data_management.data_classes import UnbatchedFlameParams


class _Squasher(nn.Module):
    """ Squashes the windowed input to a single vector. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
    ) -> None:
        """
        Args:
            input_size (int): The input size.
            hidden_size (int): The hidden size.
            output_size (int): The output size.
        """
        super().__init__()
        cnn_layers = []
        cnn_layers.append(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1))
        cnn_layers.append(nn.SiLU())
        for _ in range(n_layers - 2):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1))
            cnn_layers.append(nn.SiLU())
        cnn_layers.append(
            nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, padding=1))
        cnn_layers.append(nn.SiLU())
        cnn_layers.append(nn.AdaptiveMaxPool1d(1))
        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, x: Float[torch.Tensor, 'window in_dim']) -> Float[torch.Tensor, 'out_dim']:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (window_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (output_size)
        """
        x = x.unsqueeze(0).transpose(1, 2)  # (batch_size, in_dim, window_size)
        x = self.cnn(x)
        return x.squeeze(0).squeeze(-1)  # (out_dim)


class PerGaussianDeformations(nn.Module):

    def __init__(
        self,
        window_size: int,
        use_audio_features: bool,
        use_flame_params: bool,
        mlp_layers: int,
        mlp_hidden_size: int,
        per_gaussian_latent_dim: int = 32,
        use_audio_code_book: bool = False,
    ) -> None:
        """
        Args:
            window_size (int): The window size.
            use_audio_features (bool): Whether to use audio features.
            use_flame_params (bool): Whether to use flame parameters.
            mlp_layers (int): The number of layers in the mlp.
            mlp_hidden_size (int): The hidden size of the mlp.
            use_audio_code_book (bool): Whether to use audio code book. Defaults to False.
        """
        super().__init__()
        self.window_size = window_size
        self.use_audio_features = use_audio_features
        self.use_audio_code_book = use_audio_code_book
        self.use_flame_params = use_flame_params

        # input
        input_size = 14 + per_gaussian_latent_dim  # 14 for 6dof position and adjustment
        if use_audio_features and not use_audio_code_book:
            input_size += 32
            self.audio_squasher = _Squasher(
                input_size=1024,
                hidden_size=256,
                output_size=32,
                n_layers=3,
            )
        if use_audio_features and use_audio_code_book:
            input_size += 32
            self.audio_squasher = _Squasher(
                input_size=1024,
                hidden_size=256,
                output_size=256,
                n_layers=3,
            )
            self.code_book_mlp = nn.Sequential(
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 32),
            )
            # TODO: quantize the audio features
            raise NotImplementedError("Code book not implemented yet.")
        if use_flame_params:
            input_size += 32
            self.flame_squasher = _Squasher(
                input_size=413,  # 300 shape, 100 expr, 3 neck, 3 jaw, 6 eye, 1 scale
                hidden_size=128,
                output_size=32,
                n_layers=3,
            )

        # mlp
        mlp_layer_list = []
        mlp_layer_list.append(nn.Linear(input_size, mlp_hidden_size))
        mlp_layer_list.append(nn.SiLU())
        for _ in range(mlp_layers - 2):
            mlp_layer_list.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            mlp_layer_list.append(nn.SiLU())
        mlp_layer_list.append(nn.Linear(mlp_hidden_size, 7))
        self.mlp = nn.Sequential(*mlp_layer_list)

    def forward(
        self,
        splats: nn.ParameterDict,
        rigged_rotation: Float[torch.Tensor, "n_gaussians 4"],
        rigged_translation: Float[torch.Tensor, "n_gaussians 3"],
        audio_features: Float[torch.Tensor, "window_size 1024"] | None = None,
        flame_params: UnbatchedFlameParams | None = None,
    ) -> tuple[Float[torch.Tensor, 'n_gaussians 4'], Float[torch.Tensor, 'n_gaussians 3']]:
        """
        Computes the final adjustments to the per gaussian means and quaternions. Returns the
        *adjustments* not the final means and quaternions.

        Args:
            splats (dict): The splats.
            rigged_rotation (torch.Tensor): The rigged rotations. Has shape `(n_gaussians, 4)`.
            rigged_translation (torch.Tensor): The rigged translations. Has shape
                `(n_gaussians, 3)`.
            audio_features (torch.Tensor): The audio features. Has shape `(window_size, 1024)`.
            flame_params (UnbatchedFlameParams): The flame parameters.

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): The rotation adjustments. Has shape `(n_gaussians, 4)`.
                - (*torch.Tensor*): The translation adjustments. Has shape `(n_gaussians, 3)`.
        """

        means = splats['means']
        quats = splats['quats']
        features = splats['features']
        input_list = [means, quats, features, rigged_rotation, rigged_translation]

        if self.use_audio_features and not self.use_audio_code_book:
            audio_features = self.audio_squasher(audio_features)  # (32)
            audio_features = repeat(
                audio_features, "f -> n_gaussians f", n_gaussians=means.shape[0])
            input_list.append(audio_features)

        if self.use_audio_features and self.use_audio_code_book:
            raise NotImplementedError("Code book not implemented yet.")

        if self.use_flame_params:
            flame_features = torch.cat([
                flame_params.shape,
                flame_params.expr,
                flame_params.neck,
                flame_params.jaw,
                flame_params.eye,
                flame_params.scale,
            ],
                                       dim=-1)
            flame_features = self.flame_squasher(flame_features)
            flame_features = repeat(
                flame_features, "f -> n_gaussians f", n_gaussians=means.shape[0])
            input_list.append(flame_features)

        input_tensor = torch.concatenate(input_list, dim=-1)
        adjustments = self.mlp.forward(input_tensor)
        rotation_adjustments = adjustments[..., :4]
        rotation_adjustments = nn.functional.normalize(rotation_adjustments, p=2, dim=-1)
        translation_adjustments = adjustments[..., 4:] * 1e-4

        return rotation_adjustments, translation_adjustments
