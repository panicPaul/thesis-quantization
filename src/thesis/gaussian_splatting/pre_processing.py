""" Pre-processing for the gaussian splatting algorithm. """

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig
from torch.optim import AdamW

from thesis.config import DynamicGaussianSplattingSettings
from thesis.constants import (
    CANONICAL_FLAME_PARAMS,
    CANONICAL_FLAME_PARAMS_OTHER_GUY,
    TRAIN_CAMS,
)
from thesis.data_management.data_classes import (
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)
from thesis.deformation_field.barycentric_weighting import (
    apply_barycentric_weights,
    compute_barycentric_weights,
)
from thesis.deformation_field.flame_knn import FlameKNN
from thesis.deformation_field.mesh_se3_extraction import FlameMeshSE3Extraction
from thesis.flame import FlameHead, FlameHeadVanilla, FlameHeadWithInnerMouth
from thesis.gaussian_splatting.per_gaussian_coloring import PerGaussianColoring
from thesis.gaussian_splatting.per_gaussian_deformation import PerGaussianDeformations
from thesis.gaussian_splatting.view_dependent_coloring import (
    LearnableShader,
    ViewDependentColorMLP,
)
from thesis.utils import (
    apply_se3_to_orientation,
    apply_se3_to_point,
    quaternion_multiplication,
)


class RiggedPreProcessor(nn.Module):
    """ Any function that happens before the rasterization but after the rigging. """

    def __init__(
        self,
        gaussian_splatting_settings: DynamicGaussianSplattingSettings,
        learning_rates: DictConfig,
    ) -> None:
        """
        Args:
            gaussian_splatting_settings (GaussianSplattingSettings): The gaussian splatting
                settings.
            learning_rates (DictConfig): The learning rates.
        """

        super().__init__()

        self.gaussian_splatting_settings = gaussian_splatting_settings
        self.learning_rates = learning_rates

        # Load canonical flame params
        if not gaussian_splatting_settings.use_other_guy:
            canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS)
        else:
            canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS_OTHER_GUY)
        self.register_buffer("canonical_flame_shape", canonical_flame_params.shape)
        self.register_buffer("canonical_flame_expr", canonical_flame_params.expr)
        self.register_buffer("canonical_flame_neck", canonical_flame_params.neck)
        self.register_buffer("canonical_flame_jaw", canonical_flame_params.jaw)
        self.register_buffer("canonical_flame_eye", canonical_flame_params.eye)
        self.register_buffer("canonical_flame_scale", canonical_flame_params.scale)
        self.cuda()

        match gaussian_splatting_settings.flame_head_type.lower():
            case 'with_inner_mouth':
                self.flame_head = FlameHeadWithInnerMouth()
                n_vertices = 5443
            case 'with_teeth':
                self.flame_head = FlameHead()
                n_vertices = 5143
            case 'vanilla':
                self.flame_head = FlameHeadVanilla()
                n_vertices = 5023
            case _:
                raise ValueError("Unknown flame head type.")
        self.flame_knn = FlameKNN(
            k=3, canonical_params=self.canonical_flame_params, flame_head=self.flame_head)
        self.flame_mesh_extractor = FlameMeshSE3Extraction(self.flame_head)

        if gaussian_splatting_settings.use_view_dependent_color_mlp:
            self.view_dependent_color_mlp = ViewDependentColorMLP(
                feature_dim=gaussian_splatting_settings.feature_dim,
                sh_degree=gaussian_splatting_settings.sh_degree,
                num_cameras=len(TRAIN_CAMS),
            )

        if gaussian_splatting_settings.per_gaussian_motion_adjustment:
            self.per_gaussian_deformations = PerGaussianDeformations(
                window_size=gaussian_splatting_settings.prior_window_size,
                use_audio_features=gaussian_splatting_settings
                .per_gaussian_motion_adjustment_use_audio,
                use_flame_params=gaussian_splatting_settings
                .per_gaussian_motion_adjustment_use_flame,
                use_rigging_params=gaussian_splatting_settings
                .per_gaussian_motion_adjustment_use_rigging,
                mlp_layers=4,
                mlp_hidden_size=128,
                per_gaussian_latent_dim=gaussian_splatting_settings.feature_dim,
                n_vertices=n_vertices,
            )

        if gaussian_splatting_settings.per_gaussian_coloring_adjustment:
            self.per_gaussian_color_adjustment = PerGaussianColoring(
                window_size=gaussian_splatting_settings.prior_window_size,
                use_audio_features=gaussian_splatting_settings
                .per_gaussian_coloring_adjustment_use_audio,
                use_flame_params=gaussian_splatting_settings
                .per_gaussian_coloring_adjustment_use_flame,
                use_rigging_params=gaussian_splatting_settings
                .per_gaussian_coloring_adjustment_use_rigging,
                mlp_layers=4,
                mlp_hidden_size=128,
                per_gaussian_latent_dim=gaussian_splatting_settings.feature_dim,
                n_vertices=n_vertices,
            )

        if gaussian_splatting_settings.learnable_shader:
            self.learnable_shader = LearnableShader(
                feature_dim=gaussian_splatting_settings.feature_dim)

    @property
    def canonical_flame_params(self) -> UnbatchedFlameParams:
        """ Returns the canonical flame parameters. """
        return UnbatchedFlameParams(
            shape=self.canonical_flame_shape,
            expr=self.canonical_flame_expr,
            neck=self.canonical_flame_neck,
            jaw=self.canonical_flame_jaw,
            eye=self.canonical_flame_eye,
            scale=self.canonical_flame_scale,
        )

    def setup_optimizer(self) -> AdamW | None:
        """ Sets up the optimizer for the pre-processor. """
        batch_size = self.gaussian_splatting_settings.camera_batch_size
        batch_scaling = math.sqrt(batch_size)
        params = []
        if hasattr(self, "view_dependent_color_mlp"):
            params.append({
                "params": self.view_dependent_color_mlp.color_head.parameters(),
                'lr': self.learning_rates.color_mlp_lr * batch_scaling,
                'weight_decay': 1e-2,  # standard weight decay
            })
            params.append({
                "params": self.view_dependent_color_mlp.embeds.parameters(),
                'lr': self.learning_rates.color_mlp_lr * batch_scaling * 10.0,
                'weight_decay': 1e-6,  # as gsplat uses
            })
        if hasattr(self, "per_gaussian_deformations"):
            params.append({
                "params": self.per_gaussian_deformations.parameters(),
                'lr': self.learning_rates.per_gaussian_deformations_lr * batch_scaling,
                'weight_decay': 1e-2,  # standard weight decay
            })
        if hasattr(self, "per_gaussian_color_adjustment"):
            params.append({
                "params": self.per_gaussian_color_adjustment.parameters(),
                'lr': self.learning_rates.per_gaussian_color_adjustment_lr * batch_scaling,
                'weight_decay': 1e-2,  # standard weight decay
            })
        if hasattr(self, "learnable_shader"):
            params.append({
                "params": self.learnable_shader.parameters(),
                'lr': self.learning_rates.learnable_shader_lr * batch_scaling,
                'weight_decay': 1e-2,  # standard weight decay
            })
        if len(params) == 0:
            return None
        else:
            return AdamW(params)

    def forward(
        self,
        splats: nn.ParameterDict | dict[str, torch.Tensor],
        se3_transform: UnbatchedSE3Transform,
        rigging_params: Float[torch.Tensor, "n_vertices 3"],
        cam_2_world: Float[torch.Tensor, "cam 4 4"],
        world_2_cam: Float[torch.Tensor, "cam 4 4"],
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
        cur_sh_degree: int | None = None,
        flame_params: UnbatchedFlameParams | None = None,
        audio_features: Float[torch.Tensor, "window_size 1024"] | None = None,
        windowed_rigging_params: Float[torch.Tensor, "window n_vertices 3"] | None = None,
        infos: dict | None = None,
    ) -> tuple[
            Float[torch.Tensor, "n_gaussians 3"],
            Float[torch.Tensor, "n_gaussians 4"],
            Float[torch.Tensor, "n_gaussians 3"],
            Float[torch.Tensor, "n_gaussians"],
            Float[torch.Tensor, "cam n_gaussians 3"],
            dict,
    ]:
        """
        Pre-processing step for the rasterization.

        Args:
            splats (nn.ParameterDict | dict[str, torch.Tensor]): The splats.
            se3_transform (UnbatchedSE3Transform): The SE3 transformation.
            rigging_params (torch.Tensor): The rigging parameters. Shape: `(n_vertices, 3)`.
            cam_2_world (torch.Tensor): The camera to world transformation. Shape: `(cam, 4, 4)`.
            world_2_cam (torch.Tensor): The world to camera transformation. Shape: `(cam, 4, 4)`.
            camera_indices (torch.Tensor): The camera indices. Shape: `(cam,)`.
            cur_sh_degree (int): The current SH degree.
            flame_params (UnbatchedFlameParams): The flame parameters.
            audio_features (torch.Tensor): The audio features. Shape: `(window_size, 1024)`.
            windowed_rigging_params (torch.Tensor): The windowed rigging parameters. Shape:
                `(window, n_vertices, 3)`.
            infos (dict): Additional information.


        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): Means, shape: `(n_gaussians, 3)`.
                - (*torch.Tensor*): Quaternions, shape: `(n_gaussians, 4)`.
                - (*torch.Tensor*): Scales, shape: `(n_gaussians, 3)`.
                - (*torch.Tensor*): Opacities, shape: `(n_gaussians, 1)`.
                - (*torch.Tensor*): Colors, shape: `(n_gaussians, 3)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """

        # ---> Set-up
        means = splats["means"]
        quats = splats["quats"]
        features = splats["features"]
        if 'colors' in splats:
            colors = splats['colors']
        if infos is None:
            infos = {}
        if camera_indices is None:
            camera_indices = torch.zeros(1, dtype=torch.int64, device=means.device)

        # ---> rigged deformation field
        if self.gaussian_splatting_settings.flame_deformation_field:
            indices, _ = self.flame_knn.forward(means)
            canonical_vertices = self.flame_head.forward(self.canonical_flame_params)
            deformed_vertices = rigging_params.unsqueeze(0)
            vertex_rotations, vertex_translations = self.flame_mesh_extractor.forward(
                canonical_vertices, deformed_vertices)
            vertex_rotations = vertex_rotations.permute(1, 0, 2)  # (n_vertices, window_size, 4)
            gaussian_rotations = self.flame_knn.gather(
                indices, vertex_rotations)  # (n_gaussians, k, window_size, 4)
            # NOTE: I do not want to have to deal with spherical interpolation and this should be
            #       close enough
            # gaussian_rotations = gaussian_rotations[:, 0]  # (n_gaussians, window_size, 4)
            # gaussian_rotations = gaussian_rotations.permute(1, 0,
            #                                                 2)  # (window_size, n_gaussians, 4)

            # (n_gaussians, k, window_size, 4) -> (window_size, n_gaussians, k, 4)
            gaussian_rotations = gaussian_rotations.permute(2, 0, 1, 3)

            vertex_translations = vertex_translations.permute(1, 0,
                                                              2)  # (n_vertices, window_size, 3)
            gaussian_translations = self.flame_knn.gather(
                indices, vertex_translations)  # (n_gaussians, k, window_size, 3)
            nn_positions = self.flame_knn.gather(indices,
                                                 canonical_vertices[0])  # (n_gaussians, 3)
            barycentric_weights = compute_barycentric_weights(means,
                                                              nn_positions)  # (n_gaussians, 3)
            gaussian_translations = apply_barycentric_weights(
                barycentric_weights, gaussian_translations)  # (n_gaussians, window_size, 3)
            gaussian_translations = gaussian_translations.permute(
                1, 0, 2)  # (window_size, n_gaussians, 3)

            # remove window size
            gaussian_rotations = gaussian_rotations.squeeze(0)
            gaussian_translations = gaussian_translations.squeeze(0)
            barycentric_weights = barycentric_weights.squeeze(0)

        else:
            gaussian_rotations = torch.zeros_like(quats)
            gaussian_rotations[:, 0] = 1.0
            gaussian_rotations = gaussian_rotations.unsqueeze(1)
            gaussian_translations = torch.zeros_like(means)

        # apply the deformation field
        means = means + gaussian_translations
        quats = nn.functional.normalize(quats, p=2, dim=-1)
        quats = quaternion_multiplication(gaussian_rotations[:, 0], quats)

        # ---> Per Gaussian fine tuning
        if hasattr(self, "per_gaussian_deformations"):
            rotation_adjustments, translation_adjustments = self.per_gaussian_deformations.forward(
                splats=splats,
                rigged_rotation=gaussian_rotations[:, 0],
                rigged_translation=gaussian_translations,
                audio_features=audio_features,
                flame_params=flame_params,
                rigging_params=windowed_rigging_params,
            )
            means = means + translation_adjustments
            quats = quaternion_multiplication(rotation_adjustments, quats)
            infos['per_gaussian_movement'] = translation_adjustments.norm(dim=-1).mean()

        if hasattr(self, "per_gaussian_color_adjustment"):
            color_adjustments = self.per_gaussian_color_adjustment.forward(
                splats=splats,
                rigged_rotation=gaussian_rotations[:, 0],
                rigged_translation=gaussian_translations,
                audio_features=audio_features,
                flame_params=flame_params,
                rigging_params=windowed_rigging_params,
            )
            colors = colors + color_adjustments
            infos['per_gaussian_color_adjustment'] = color_adjustments.norm(dim=-1).mean()
            colors = nn.functional.sigmoid(colors)

        # ---> SE(3) transformation
        rotation = se3_transform.rotation
        translation = se3_transform.translation
        means = apply_se3_to_point(rotation, translation, means)
        quats = nn.functional.normalize(quats, p=2, dim=-1)
        quats = apply_se3_to_orientation(rotation, quats)

        # ---> Coloring
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.view_dependent_color_mlp.forward(
                features=features,
                camera_ids=camera_indices,
                means=means,
                colors=colors,
                cam_2_world=cam_2_world,
                cur_sh_degree=cur_sh_degree
                if cur_sh_degree is not None else self.gaussian_splatting_settings.sh_degree,
            )
            # are being sigmoided here
        else:
            colors = colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])

        # ---> Shading
        if hasattr(self, "learnable_shader"):
            colors = self.learnable_shader.forward(
                spatial_position=means,
                orientation=quats,
                features=features,
                colors=colors,
            )

        return means, quats, scales, opacities, colors, infos
