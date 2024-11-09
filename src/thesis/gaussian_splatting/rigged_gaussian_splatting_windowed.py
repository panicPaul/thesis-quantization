""" Gaussian Splatting fitted on a single time step, then fine tuned with flame rigs only """

import argparse
import math
import socket
import time
from functools import partial
from typing import Literal

import lightning as pl
import matplotlib.pyplot as plt
import nerfview
import numpy as np
import torch
import viser
from einops import rearrange, repeat
from gsplat import DefaultStrategy, MCMCStrategy, rasterization, rasterization_2dgs
from jaxtyping import Float, Int, UInt8
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, AdamW  # , SparseAdam
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from thesis.code_talker.models.lib.quantizer import VectorQuantizer
from thesis.config import GaussianSplattingSettings, load_config
from thesis.constants import (
    CANONICAL_FLAME_PARAMS,
    CANONICAL_SEQUENCE_FRAME,
    DEFAULT_SE3_ROTATION,
    DEFAULT_SE3_TRANSLATION,
    TEST_CAMS,
    TRAIN_CAMS,
    TRAIN_SEQUENCES,
)
from thesis.data_management import (
    SequenceManager,
    SequentialMultiSequenceDataset,
    SingleSequenceDataset,
    UnbatchedFlameParams,
)
from thesis.data_management.data_classes import SingleFrameData, UnbatchedSE3Transform
from thesis.deformation_field.barycentric_weighting import (
    apply_barycentric_weights,
    compute_barycentric_weights,
)
from thesis.deformation_field.flame_knn import FlameKNN
from thesis.deformation_field.mesh_se3_extraction import FlameMeshSE3Extraction
from thesis.flame import FlameHeadWithInnerMouth
from thesis.gaussian_splatting.camera_color_correction import LearnableColorCorrection
from thesis.gaussian_splatting.initialize_splats import (
    flame_initialization,
    point_cloud_initialization,
    pre_trained_initialization,
    random_initialization,
)
from thesis.gaussian_splatting.screen_space_denoising import ScreenSpaceDenoising
from thesis.gaussian_splatting.view_dependent_coloring import ViewDependentColorMLP
from thesis.utils import (
    apply_se3_to_orientation,
    apply_se3_to_point,
    assign_segmentation_class,
    quaternion_multiplication,
)


# TODO: experiment with SIREN here?
class AudioFineTuning(nn.Module):
    """ Audio fine tuning module. """

    def __init__(
        self,
        audio_feature_dim: int = 1024,
        audio_embedding_dim: int = 32,
        audio_codebook_size: int = 64,
        audio_beta_commitment_loss: float = 0.25,
        flame_param_size: int = 113,
        n_vertices: int = 5443,
    ) -> None:
        """ """
        super().__init__()

        # audio
        self.audio_squasher = nn.Sequential(
            nn.Conv1d(audio_feature_dim, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, audio_embedding_dim, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )  # (batch, 1024, window_size) -> (batch, 128, 1)
        self.audio_quantizer = VectorQuantizer(
            n_e=audio_codebook_size,
            e_dim=audio_embedding_dim,
            beta=audio_beta_commitment_loss,
        )

        # flame params
        self.flame_squasher = nn.Sequential(
            nn.Conv1d(flame_param_size, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, n_vertices * 3),
        )

    def forward(
        self,
        audio_features: Float[torch.Tensor, "window_size audio_feature_dim"],
        flame_params: UnbatchedFlameParams,
    ) -> tuple[Float[torch.Tensor, "n_vertices 3"], Float[torch.Tensor, ""]]:
        """ """
        flame_params = torch.concatenate([
            flame_params.expr,
            flame_params.jaw,
            flame_params.neck,
            flame_params.scale,
            flame_params.shape,
        ])  # (window_size, flame_param_size)
        flame_params = rearrange(flame_params.unsqueeze(0), 'b ws f -> b f ws')

        audio_features = self.audio_squasher(audio_features.unsqueeze(0))  # (1, 32, 1)
        audio_features = rearrange(audio_features, 'b f ws -> b ws f')  # (1, 1, 32)
        audio_features, emb_loss, _ = self.audio_quantizer.forward(audio_features)  # (1, 32, 1)
        audio_features = audio_features.squeeze(-1)  # (1, 32)

        flame_params = self.flame_squasher.forward(flame_params)  # (1, 32, 1)
        flame_params = flame_params.squeeze(-1)  # (1, 32)

        fusion = torch.cat([audio_features, flame_params], dim=-1)  # (1, 64)
        fusion = self.fusion.forward(fusion)  # (1, n_vertices * 3)
        fusion = rearrange(fusion.squeeze(0), '(n_vertices c) -> n_vertices c', n_vertices=5443)

        return fusion, emb_loss


class RiggedGaussianSplatting(pl.LightningModule):
    """
    We learn the gaussian splatting on a
    """

    def __init__(
        self,
        gaussian_splatting_settings: GaussianSplattingSettings | DictConfig,
        learning_rates: DictConfig,
        enable_viewer: bool = True,
        train_sequences: list[int] | None = TRAIN_SEQUENCES,
        ckpt_path: str | None = None,
        training_mode: Literal['static', 'rigged', 'dynamic', 'inside_mouth'] = 'static',
    ) -> None:
        """
        Initializes the Gaussian splatting model. You must choose one of the following
        training modes:

            - static:       We optimize the splats directly but freeze the rigging parameters and
                            the per-gaussian motion adjustments.
            - rigged:       We optimize the rigging parameters and the per-gaussian motion
                            adjustments but freeze the splats.
            - dynamic:      We optimize everything.
            - inside_mouth: The splats and rigging parameters are frozen for the normal
                            splats, but the inside mouth splats and rigging params are optimized.

        Args:
            sequence (int): Sequence to train on.
            frame (int): Frame to train on.
            gaussian_splatting_settings (GaussianSplattingSettings | DictConfig): Gaussian
                splatting settings.
            learning_rates (DictConfig): Learning rates.
            enable_viewer (bool): Whether to enable the viewer.
            train_sequences (list[int]): List of sequences to train on.
            checkpoint_path (str | None): Path to the checkpoint. Needs to be provided if
                loading a pre-trained model, since torch will throw size mismatch errors
                otherwise.
            training_mode (str): The mode of the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if train_sequences is None:
            train_sequences = TRAIN_SEQUENCES
        self.train_sequences = train_sequences
        self.training_mode = training_mode

        # Save the settings
        self.learning_rates = learning_rates
        self.gaussian_splatting_settings = gaussian_splatting_settings
        if isinstance(gaussian_splatting_settings, DictConfig):
            gaussian_splatting_settings = GaussianSplattingSettings(**gaussian_splatting_settings)

        # Process settings
        self.max_sh_degree = gaussian_splatting_settings.sh_degree
        self.default_background = torch.tensor([
            gaussian_splatting_settings.background_r,
            gaussian_splatting_settings.background_g,
            gaussian_splatting_settings.background_b,
        ]).cuda()
        self.enable_viewer = enable_viewer

        # Load flame params
        self.canonical_flame_params = UnbatchedFlameParams(
            *(a.to('cuda') for a in CANONICAL_FLAME_PARAMS))

        # Initialize splats
        initialization_mode = gaussian_splatting_settings.initialization_mode
        if ckpt_path is not None:
            initialization_mode = "pre_trained"
        match initialization_mode:
            case "random":
                self.splats = nn.ParameterDict(
                    random_initialization(
                        num_splats=gaussian_splatting_settings.initialization_points,
                        scene_scale=gaussian_splatting_settings.scene_scale,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_spherical_harmonics=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
            case "point_cloud":
                self.splats = nn.ParameterDict(
                    point_cloud_initialization(
                        num_splats=gaussian_splatting_settings.initialization_points,
                        scene_scale=gaussian_splatting_settings.scene_scale,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_spherical_harmonics=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
            case "flame":
                self.splats = nn.ParameterDict(
                    flame_initialization(
                        flame_params=self.canonical_flame_params,
                        scene_scale=gaussian_splatting_settings.scene_scale,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_spherical_harmonics=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
            case "pre_trained":
                if ckpt_path is None:
                    ckpt_path = gaussian_splatting_settings.initialization_checkpoint
                self.splats = nn.ParameterDict(pre_trained_initialization(ckpt_path))
            case _:
                raise ValueError("Unknown initialization mode: "
                                 f"{gaussian_splatting_settings.initialization_mode}")

        # Initialize the densification strategy
        refine_stop_iteration = gaussian_splatting_settings.refine_stop_iteration
        if isinstance(refine_stop_iteration, float):
            assert 0 < refine_stop_iteration < 1, \
                "refine_stop_iteration should be in (0, 1) if float"
            refine_stop_iteration = int(refine_stop_iteration
                                        * gaussian_splatting_settings.train_iterations)
        match gaussian_splatting_settings.densification_mode:
            case "default":
                self.strategy = DefaultStrategy(
                    refine_start_iter=gaussian_splatting_settings.refine_start_iteration,
                    refine_stop_iter=refine_stop_iteration,
                    verbose=True,
                )

            case "monte_carlo_markov_chain":
                self.strategy = MCMCStrategy(
                    refine_start_iter=gaussian_splatting_settings.refine_start_iteration,
                    refine_stop_iter=refine_stop_iteration,
                    verbose=True,
                    cap_max=gaussian_splatting_settings.cap_max,
                )

            case _:
                raise ValueError("Unknown densification mode: "
                                 f"{gaussian_splatting_settings.densification_mode}")
        self.strategy_state = self.strategy.initialize_state()

        # deformation_field (pre-processing)
        # non-optimizable
        self.flame_head = FlameHeadWithInnerMouth()
        self.flame_knn = FlameKNN(k=3, canonical_params=self.canonical_flame_params)
        self.flame_mesh_extractor = FlameMeshSE3Extraction(self.flame_head)

        # Fine Tuning

        # View-dependent color module (pre-processing)
        if gaussian_splatting_settings.use_view_dependent_color_mlp:
            self.view_dependent_color_mlp = ViewDependentColorMLP(
                feature_dim=gaussian_splatting_settings.feature_dim,
                sh_degree=gaussian_splatting_settings.sh_degree,
                num_cameras=len(TRAIN_CAMS),
            )

        # Get the rasterization function
        match gaussian_splatting_settings.rasterization_mode:
            case "default" | "3dgs":
                self.rasterize = partial(
                    rasterization,
                    radius_clip=gaussian_splatting_settings.radius_clip,
                    rasterize_mode="antialiased"
                    if gaussian_splatting_settings.antialiased else "default",
                )
            case "2dgs":
                self.rasterize = partial(
                    rasterization_2dgs,
                    radius_clip=gaussian_splatting_settings.radius_clip,
                    distloss=gaussian_splatting_settings.dist_loss is not None,
                    depth_mode='median',
                )
            case _:
                raise ValueError("Unknown rasterization mode: "
                                 f"{gaussian_splatting_settings.rasterization_mode}")

        # Screen-space denoising (post-processing)
        match gaussian_splatting_settings.screen_space_denoising_mode:
            case 'none':
                self.screen_space_denoiser = lambda img, alphas: img
            case 'cnn':
                self.screen_space_denoiser = ScreenSpaceDenoising()
            case _:
                raise ValueError("Unknown screen-space denoising mode: "
                                 f"{gaussian_splatting_settings.screen_space_denoising_mode}")

        # Learnable color correction (post-processing)
        if gaussian_splatting_settings.learnable_color_correction:
            self.learnable_color_correction = LearnableColorCorrection(len(TRAIN_CAMS))

        # Set up loss functions
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=gaussian_splatting_settings.lpips_network, normalize=True)

    # def on_save_checkpoint(self, checkpoint) -> None:
    #     """
    #     Remove specific state_dict keys before saving.

    #     Args:
    #         checkpoint (dict): The checkpoint dictionary.
    #     """
    #     # Remove specific state_dict keys before saving
    #     state_dict = checkpoint["state_dict"]

    #     # Create a list of keys to remove
    #     match self.training_mode:
    #         case 'static':
    #             for key in state_dict.keys():
    #                 if key.startswith("rigging_params"):
    #                     del state_dict[key]
    #                 if key.startswith("rotation_and_scale_adjustments"):
    #                     del state_dict[key]
    #                 if key.startswith("lpips"):
    #                     del state_dict[key]
    #         case _:
    #             for key in state_dict.keys():
    #                 if key.startswith("lpips"):
    #                     del state_dict[key]

    # def on_load_checkpoint(self, checkpoint) -> None:
    #     """ Load the checkpoint. """
    #     state_dict = checkpoint["state_dict"]
    #     self.load_state_dict(state_dict)

    def configure_optimizers(self):
        """
        Returns:
            A tuple containing the splat optimizers and other optimizers.
        """
        mode = self.training_mode

        # ---> splat optimizers
        batch_size = self.gaussian_splatting_settings.camera_batch_size
        batch_scaling = math.sqrt(batch_size)
        scene_scale = self.gaussian_splatting_settings.scene_scale
        splat_optimizers = {}
        splats_learning_rates = {
            "means": self.learning_rates.means_lr * batch_scaling * scene_scale,
            "scales": self.learning_rates.scales_lr * batch_scaling,
            "quats": self.learning_rates.quats_lr * batch_scaling,
            "opacities": self.learning_rates.opacities_lr * batch_scaling,
            "features": self.learning_rates.features_lr * batch_scaling,
        }
        enable_splats = 1.0 if mode in ['static', 'dynamic'] else 0.0
        splat_optimizers = {
            name:
                Adam(
                    [{
                        "params": self.splats[name],
                        "lr": lr * enable_splats,
                        "name": name,
                    }],
                    eps=1e-15 / batch_scaling,
                    # TODO: check betas logic when cfg.batch_size is larger than 10 betas[0]
                    #       will be zero.
                    betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
                ) for name, lr in splats_learning_rates.items()
        }
        if not self.gaussian_splatting_settings.use_view_dependent_color_mlp:
            splat_optimizers["sh0"] = Adam(
                [self.splats["sh0"]],
                lr=self.learning_rates.sh0_lr * batch_scaling * enable_splats,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )
            splat_optimizers["shN"] = Adam(
                [self.splats["shN"]],
                lr=(self.learning_rates.sh0_lr / 20) * batch_scaling * enable_splats,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )
        else:
            splat_optimizers['colors'] = Adam(
                [self.splats['colors']],
                lr=self.learning_rates.color_lr * batch_scaling * enable_splats,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )

        # ---> pre-processing optimizers
        other_optimizers = {}
        if hasattr(self, "rigging_params"):
            if mode in ['rigged', 'dynamic']:
                other_optimizers["rigging_params_flame_vertices"] = Adam(
                    self.rigging_params.flame_code_books.parameters(),
                    lr=self.learning_rates.rigging_params_flame_vertices_lr * batch_scaling,
                )
            if mode in ['inside_mouth', 'rigged', 'dynamic']:
                other_optimizers["rigging_params_inner_mouth_vertices"] = Adam(
                    self.rigging_params.inner_mouth_code_books.parameters(),
                    lr=self.learning_rates.rigging_params_inner_mouth_vertices_lr * batch_scaling,
                )

        if hasattr(self, "rotation_and_scale_adjustments"):
            if mode in ['rigged', 'dynamic']:
                other_optimizers["rotation_and_scale_adjustments"] = Adam(
                    self.rotation_and_scale_adjustments.parameters(),
                    lr=self.learning_rates.motion_adjustment_lr * batch_scaling,
                )

        if hasattr(self, "view_dependent_color_mlp"):
            other_optimizers["view_dependent_color_mlp_head"] = Adam(
                self.view_dependent_color_mlp.color_head.parameters(),
                lr=self.learning_rates.color_mlp_lr * batch_scaling,
            )
            other_optimizers['view_dependent_color_mlp_embedding'] = AdamW(
                self.view_dependent_color_mlp.embeds.parameters(),
                lr=self.learning_rates.color_mlp_lr * batch_scaling * 10.0,
                weight_decay=self.learning_rates.color_mlp_weight_decay,
            )

        # ---> post-processing optimizers
        if hasattr(self, "learnable_color_correction"):
            other_optimizers["learnable_color_correction"] = Adam(
                self.learnable_color_correction.parameters(),
                lr=self.learning_rates.color_correction_lr * batch_scaling,
            )
        if isinstance(self.screen_space_denoiser, nn.Module):
            other_optimizers["screen_space_denoiser"] = Adam(
                self.screen_space_denoiser.parameters(),
                lr=self.learning_rates.screen_space_denoiser_lr * batch_scaling,
            )

        # schedulers
        schedulers = {}
        schedulers["means"] = torch.optim.lr_scheduler.ExponentialLR(
            splat_optimizers["means"],
            gamma=0.01**(1.0 / self.gaussian_splatting_settings.train_iterations))

        optimizer_list = list(splat_optimizers.values()) + list(other_optimizers.values())
        self.splat_optimizer_keys = list(splat_optimizers.keys())
        scheduler_list = list(schedulers.values())
        self.n_optimizers = len(optimizer_list)

        if self.training_mode == 'rigged':
            return [
                other_optimizers['rigging_params_flame_vertices'],
                other_optimizers['rigging_params_inner_mouth_vertices']
            ]
        return optimizer_list, scheduler_list

    @property
    def step(self) -> int:
        """Returns the current step."""
        return self.global_step // self.n_optimizers

    # ================================================================================ #
    #                                 Rasterization                                    #
    # ================================================================================ #

    def pre_processing(
        self,
        infos: dict,
        cam_2_world: Float[torch.Tensor, "cam 4 4"],
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
        cur_sh_degree: int | None = None,
        se3_transform: UnbatchedSE3Transform | None = None,
        rigging_params: Float[torch.Tensor, 'n_vertices 3'] | None = None,
    ) -> tuple[
            Float[torch.Tensor, "n_gaussians 3"],
            Float[torch.Tensor, "n_gaussians 4"],
            Float[torch.Tensor, "n_gaussians 3"],
            Float[torch.Tensor, "n_gaussians"],
            Float[torch.Tensor, "cam n_gaussians 3"],
            dict,
    ]:
        """
        Pre-processing step for the rasterization. Note that the rigging params are actually
        shared between the inside mouth and the default splats. The inside mouth branch is always
        behind the default branch and is mostly used, as the name suggests, for the inside mouth
        splats. But we don't actually mask the splats so they could also be used for supplementing
        the default splats.

        Args:
            infos (dict): Dictionary to store additional information.
            cam_2_world (torch.Tensor): Camera to world transformation, shape: `(cam, 4, 4)`.
            camera_indices (torch.Tensor): Camera indices. Used to choose the correct camera
                color correction matrix. Shape: `(cam,)`.
            cur_sh_degree (int): Current spherical harmonic degree. If `None`, the maximum
                degree is used.
            se3_transform (torch.Tensor): SE3 transform matrix, shape: `(cam, 4, 4)`. If
                `None`, no transformation is applied.
            rigging_params (torch.Tensor): The deformed flame vertices. Shape: `(n_vertices, 3)`.


        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): Means, shape: `(n_gaussians, 3)`.
                - (*torch.Tensor*): Quaternions, shape: `(n_gaussians, 4)`.
                - (*torch.Tensor*): Scales, shape: `(n_gaussians, 3)`.
                - (*torch.Tensor*): Opacities, shape: `(n_gaussians, 1)`.
                - (*torch.Tensor*): Colors, shape: `(n_gaussians, 3)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """

        # ---> Get the splats
        means = self.splats["means"]
        quats = self.splats["quats"]
        features = self.splats["features"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.splats["colors"]
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        # ---> Apply the deformation field
        if rigging_params is not None:
            # and self.training_mode in ['rigged', 'dynamic', 'inside_mouth']:
            # query deformation_field for windowed rotation and translations
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

            # apply the deformation field
            means = means + gaussian_translations
            quats = nn.functional.normalize(quats, p=2, dim=-1)
            quats = quaternion_multiplication(gaussian_rotations[:, 0], quats)

            # per gaussian fine tuning
            if self.gaussian_splatting_settings.per_gaussian_motion_adjustment:
                quats, scales = self.rotation_and_scale_adjustments.forward(
                    means=means,
                    quats=quats,
                    translations=gaussian_translations,
                    rotations=gaussian_rotations,
                    barycentric_weights=barycentric_weights,
                    scales=scales,
                    per_gaussian_latent=features,
                )

        # ---> se3 transformation
        if se3_transform is not None:
            rotation = se3_transform.rotation
            translation = se3_transform.translation
        else:
            rotation = DEFAULT_SE3_ROTATION.unsqueeze(0).cuda()
            translation = DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda()
        means = apply_se3_to_point(rotation, translation, means)
        quats = nn.functional.normalize(quats, p=2, dim=-1)
        quats = apply_se3_to_orientation(rotation, quats)

        # ---> view-dependent color
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.view_dependent_color_mlp.forward(
                features=features,
                camera_ids=camera_indices,
                means=means,
                colors=colors,
                cam_2_world=cam_2_world,
                cur_sh_degree=cur_sh_degree,
            )

        return means, quats, scales, opacities, colors, infos

    def post_processing(
        self,
        infos: dict,
        render_images: Float[torch.Tensor, "cam H W 3"],
        render_alphas: Float[torch.Tensor, "cam H W 1"],
        background: Float[torch.Tensor, "3"],
        color_correction: Float[torch.Tensor, "cam 3 3"] | None = None,
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
    ) -> tuple[Float[torch.Tensor, "cam H W 3"], Float[torch.Tensor, "cam H W 1"], dict]:
        """
        Post-processing step for the rasterization.

        Args:
            infos (dict): Dictionary to store additional information.
            render_images (torch.Tensor): Rendered images, shape: `(cam, H, W, 3)`.
            inside_mouth_render_images (torch.Tensor): Inside mouth rendered images, shape:
                `(cam, H, W, 3)`.
            background (torch.Tensor): Background color, shape: `(3,)`.
            color_correction (torch.Tensor): Color correction matrix, shape: `(cam, 3, 3)`.
            camera_indices (torch.Tensor): Camera indices. Used to choose the correct camera
                color correction matrix. Shape: `(cam,)`.

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): RGB images, shape: `(cam, H, W, 3)`.
                - (*torch.Tensor*): Alphas, shape: `(cam, H, W, 1)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """
        # Add images to the infos
        infos['default_rendered_images'] = render_images
        infos['default_rendered_alphas'] = render_alphas

        # Apply screen-space denoising
        infos['pre_denoised_rendered_images'] = render_images
        render_images = self.screen_space_denoiser(render_images, render_alphas)

        # Apply background
        image_height, image_width = render_images.shape[1:3]
        background = repeat(
            background,
            "f -> cam H W f",
            cam=render_images.shape[0],
            H=image_height,
            W=image_width,
        )
        render_images = render_images*render_alphas + (1-render_alphas) * background

        # Apply color correction
        if (color_correction is not None and
                self.gaussian_splatting_settings.camera_color_correction):
            # Reshape color_correction to (batch * height * width, 3, 3)
            batch_size, height, width, _ = render_images.shape
            color_correction = color_correction.expand(batch_size * height * width, -1, -1)

            # Reshape images to (batch * height * width, 3, 1)
            render_images = render_images.view(batch_size, height, width, 3, 1)
            render_images = render_images.permute(0, 1, 2, 4, 3).contiguous()
            render_images = render_images.view(-1, 3, 1)

            # Perform batch matrix multiplication
            corrected_colors = torch.bmm(color_correction, render_images)

            # Reshape the result back to (batch, height, width, 3)
            corrected_colors = corrected_colors.view(batch_size, height, width, 1, 3)
            corrected_colors = corrected_colors.permute(0, 1, 2, 4, 3).contiguous()
            corrected_colors = corrected_colors.view(batch_size, height, width, 3)
            render_images = corrected_colors

        # Apply learnable color correction
        if hasattr(self, "learnable_color_correction") and camera_indices is not None:
            render_images = self.learnable_color_correction.forward(camera_indices, render_images)

        return render_images, render_alphas, infos

    def forward(
        self,
        intrinsics: Float[torch.Tensor, "cam 3 3"],
        world_2_cam: Float[torch.Tensor, "cam 4 4"] | None,
        cam_2_world: Float[torch.Tensor, "cam 4 4"] | None,
        image_height: int,
        image_width: int,
        color_correction: Float[torch.Tensor, "cam 3 3"] | None = None,
        cur_sh_degree: int | None = None,
        se3_transform: UnbatchedSE3Transform | None = None,
        rigging_params: Float[torch.Tensor, 'n_vertices 3'] | None = None,
        background: Float[torch.Tensor, "3"] | None = None,
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
    ) -> tuple[
            Float[torch.Tensor, "cam H W 3"],
            Float[torch.Tensor, "cam H W 1"],
            Float[torch.Tensor, "cam H W 1"],
            dict,
    ]:
        """
        Args:
            intrinsics (torch.Tensor): Camera intrinsic, shape: `(cam, 3, 3)`.
            world_2_cam (torch.Tensor): World to camera transformation, shape: `(cam, 4, 4)`.
            image_height (int): Image height.
            image_width (int): Image width.
            color_correction (torch.Tensor): Color correction matrix, shape: `(cam, 3, 3)`.
            cur_sh_degree (int): Current spherical harmonic degree. If `None`, the maximum
                degree is used.
            se3_transform (torch.Tensor): SE3 transform matrix, shape: `(cam, 4, 4)`. If
                `None`, no transformation is applied.
            rigging_params (torch.Tensor): The deformed flame vertices. Shape: `(n_vertices, 3)`.
            background (torch.Tensor): Background color, shape: `(3,)`.
            camera_indices (torch.Tensor): Camera indices. Used to choose the correct camera
                color correction matrix. Shape: `(cam,)`.

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): RGB images, shape: `(cam, H, W, 3)`.
                - (*torch.Tensor*): Alphas, shape: `(cam, H, W, 1)`.
                - (*torch.Tensor*): Depth maps, shape: `(cam, H, W, 1)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """

        # Set up
        if cur_sh_degree is None:
            cur_sh_degree = self.max_sh_degree

        if background is None:
            background = self.default_background
            background.to(self.splats['means'].device)
        infos = {}
        infos = {"background": background}
        assert (world_2_cam is None or cam_2_world is None) \
             and (world_2_cam is not None or cam_2_world is not None), \
                'Either world_2_cam or cam_2_world should be provided'  # noqa E711
        if world_2_cam is not None:
            cam_2_world = torch.zeros_like(world_2_cam)
            cam_2_world[..., :3, :3] = world_2_cam[..., :3, :3].transpose(-2, -1)
            cam_2_world[..., :3, 3] = -torch.bmm(
                world_2_cam[..., :3, :3].transpose(-2, -1),
                world_2_cam[..., :3, 3].unsqueeze(-1),
            ).squeeze(-1)
            cam_2_world[..., 3, 3] = 1
        else:
            world_2_cam = torch.zeros_like(cam_2_world)
            world_2_cam[..., :3, :3] = cam_2_world[..., :3, :3].transpose(-2, -1)
            world_2_cam[..., :3, 3] = -torch.bmm(
                cam_2_world[..., :3, :3].transpose(-2, -1),
                cam_2_world[..., :3, 3].unsqueeze(-1),
            ).squeeze(-1)
            world_2_cam[..., 3, 3] = 1

        # ------------------------------- Pre-processing ------------------------------ #
        means, quats, scales, opacities, colors, infos = self.pre_processing(
            infos=infos,
            cam_2_world=cam_2_world,
            camera_indices=camera_indices,
            cur_sh_degree=cur_sh_degree,
            se3_transform=se3_transform,
            rigging_params=rigging_params,
        )

        # ------------------------------- Rasterization ------------------------------- #
        # Render the images
        ret = self.rasterize(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            render_mode="RGB+ED",
            viewmats=world_2_cam,
            Ks=intrinsics,
            width=image_width,
            height=image_height,
            absgrad=self.gaussian_splatting_settings.densification_mode == 'default',
            sh_degree=cur_sh_degree if not hasattr(self, "view_dependent_color_mlp") else None,
            packed=True,
            sparse_grad=False,
        )
        match self.gaussian_splatting_settings.rasterization_mode:
            case "default" | "3dgs":
                render_images, render_alphas, new_infos = ret
            case "2dgs":
                render_images, render_alphas, _, _, _, _, new_infos = ret
        infos['default_infos'] = new_infos
        depth_maps = render_images[:, :, :, 3:]  # get depth maps
        render_images = render_images[:, :, :, :3]  # get RGB channels

        # ------------------------------- Post-processing ------------------------------ #
        render_images, render_alphas, infos = self.post_processing(
            infos=infos,
            render_images=render_images,
            render_alphas=render_alphas,
            background=background,
            color_correction=color_correction,
            camera_indices=camera_indices,
        )

        return render_images, render_alphas, depth_maps, infos

    @torch.no_grad()
    def render(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
        time_step: int = 0,
        render_mode: Literal['color', 'depth'] = 'color',
        depth_bounds: tuple[float, float] = (0.0, 1.0),
        is_training: bool = False,
    ) -> Float[np.ndarray, "H W 3"]:
        """Render function for NerfView."""

        image_width, image_height = img_wh
        c2w = camera_state.c2w
        # hacky world transform for old data
        # hacky_world_transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0],
        # [0, 0, 0, 1]],
        #                                  dtype=np.float32)
        # hacky world transform for new data
        hacky_world_transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                         dtype=np.float32)
        c2w = hacky_world_transform @ c2w
        cam_2_world = torch.tensor(c2w).unsqueeze(0).float().cuda()
        intrinsics = torch.tensor(camera_state.get_K(img_wh)).unsqueeze(0).float().cuda()
        se3 = UnbatchedSE3Transform(
            rotation=DEFAULT_SE3_ROTATION.unsqueeze(0).cuda(),
            translation=DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda(),
        )

        # get the rigging params
        sequence = self.viewer_sequence
        # frame = time_step if self.viewer_frame is None else self.viewer_frame
        frame = time_step
        rigging_params = self.rigging_params.forward(sequence, frame)

        # render
        image, _, depth, _ = self.forward(
            intrinsics=intrinsics,
            world_2_cam=None,
            cam_2_world=cam_2_world,
            image_height=image_height,
            image_width=image_width,
            se3_transform=se3,
            rigging_params=rigging_params,
        )
        if render_mode == 'color':
            return image[0].detach().cpu().numpy()
        else:
            colormap = 'viridis'
            cmap = plt.get_cmap(colormap)
            depth = depth[0].detach().cpu().numpy().squeeze(-1)
            # normalize with provided percentile values
            # l = min(max(depth_bounds[0], 0), 1)
            # r = min(max(depth_bounds[1], 0), 1)
            # l = min(l, r)
            # depth_lower = np.percentile(depth, l * 100)
            # depth_upper = np.percentile(depth, r * 100)
            depth_lower, depth_upper = depth_bounds
            depth = (depth-depth_lower) / (depth_upper-depth_lower)
            depth = cmap(1 - depth)[:, :, :3]  # brighter should be closer
            return depth

    # ================================================================================ #
    #                                 Train / Val Steps                                #
    # ================================================================================ #

    def get_cur_sh_degree(self, step: int) -> int:
        """Returns the current spherical harmonic degree."""
        return min(
            step // self.gaussian_splatting_settings.sh_increase_interval,
            self.max_sh_degree,
        )

    #       - implement inside mouth masking
    # TODO: do i need a rigidity loss? probably not, since my rigging params already take care of
    #       it
    def compute_loss(
        self,
        rendered_images: Float[torch.Tensor, "cam H W 3"],
        rendered_alphas: Float[torch.Tensor, "cam H W 1"],
        target_images: Float[torch.Tensor, "cam H W 3"],
        target_alphas: Float[torch.Tensor, "cam H W"],
        target_segmentation_mask: Float[torch.Tensor, "cam H W 3"],
        infos: dict,
        denoised_images: Float[torch.Tensor, "cam H W 3"] | None = None,
    ) -> dict[str, Float[torch.Tensor, '']]:
        """
        Computes the loss. We will compute the loss once for the default rendering and once for
        the merged rendering. The merged rendering is the default rendering with the inside mouth
        branch added behind it.

        Args:
            rendered_images (torch.Tensor): Rendered images, shape: `(cam, H, W, 3)`.
            rendered_alphas (torch.Tensor): Rendered alphas, shape: `(cam, H, W, 1)`.
            target_images (torch.Tensor): Target images, shape: `(cam, H, W, 3)`.
            target_alphas (torch.Tensor): Target alphas, shape: `(cam, H, W)`.
            target_segmentation_mask (torch.Tensor): Target segmentation mask, shape:
                `(cam, H, W, 3)`.
            infos (dict): Dictionary containing additional information.
            mode (str): Mode, either 'default' or 'merged'.
            denoised_images (torch.Tensor): Denoised images, shape: `(cam, H, W, 3)`.
        """
        # set up
        loss_dict = {}
        loss = 0.0
        segmentation_classes = assign_segmentation_class(target_segmentation_mask)
        if self.gaussian_splatting_settings.jumper_is_background:
            background_mask = torch.where(segmentation_classes == 0, 1, 0)
            jumper_mask = torch.where(segmentation_classes == 2, 1, 0)
            target_alphas = target_alphas * (1-jumper_mask) * (1-background_mask)
        alpha_map = repeat(target_alphas, "cam H W -> cam H W f", f=3)
        background = infos['background']
        background = repeat(
            background,
            "f -> cam H W f",
            cam=alpha_map.shape[0],
            H=alpha_map.shape[1],
            W=alpha_map.shape[2])
        target_images = target_images*alpha_map + (1-alpha_map) * background
        rendered_images = rendered_images*alpha_map + (1-alpha_map) * background

        # raw l1 foreground loss
        if self.gaussian_splatting_settings.l1_image_loss is not None:
            l1_foreground_loss = torch.sum(
                torch.abs(rendered_images - target_images), dim=-1).mean()
            loss_dict["l1_foreground_loss"] = l1_foreground_loss
            loss = loss + l1_foreground_loss * self.gaussian_splatting_settings.l1_image_loss
        # raw ssim foreground loss
        if self.gaussian_splatting_settings.ssim_image_loss is not None:
            pred = rearrange(rendered_images, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            ssim_foreground_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["ssim_foreground_loss"] = ssim_foreground_loss
            loss = loss + ssim_foreground_loss * self.gaussian_splatting_settings.ssim_image_loss
        # denoised ssim foreground loss
        if self.gaussian_splatting_settings.ssim_denoised_image_loss is not None:
            assert denoised_images is not None
            pred = rearrange(denoised_images, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            ssim_denoised_foreground_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["ssim_denoised_foreground_loss"] = ssim_denoised_foreground_loss
            loss = loss + ssim_denoised_foreground_loss \
                * self.gaussian_splatting_settings.ssim_denoised_image_loss
        # lpips foreground loss
        if self.gaussian_splatting_settings.lpips_image_loss is not None:
            pred = rearrange(rendered_images, "cam H W f -> cam f H W").clip(0, 1)
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            lpips_foreground_loss = self.lpips.forward(pred, tgt).mean()
            loss_dict["lpips_foreground_loss"] = lpips_foreground_loss
            loss = loss + lpips_foreground_loss * self.gaussian_splatting_settings.lpips_image_loss
        # background loss
        if self.gaussian_splatting_settings.background_loss is not None:
            background_loss = nn.functional.mse_loss(rendered_alphas.squeeze(-1),
                                                     target_alphas).mean()
            loss_dict["background_loss"] = background_loss
            loss = loss + background_loss * self.gaussian_splatting_settings.background_loss
        # aniostropy loss
        if self.gaussian_splatting_settings.anisotropy_loss is not None:

            @torch.compiler.disable
            def f():
                scales = self.splats['scales']
                log_scale_ratio = log_scale_ratio = scales.max(dim=1).values - scales.min(
                    dim=1).values
                scale_ratio = torch.exp(log_scale_ratio)
                max_ratio = self.gaussian_splatting_settings.anisotropy_max_ratio
                anisotropy_loss = (
                    torch.where(scale_ratio > max_ratio, scale_ratio, max_ratio).mean()
                    - max_ratio)
                return anisotropy_loss

            anisotropy_loss = f()
            loss_dict["anisotropy_loss"] = anisotropy_loss
            loss = loss + anisotropy_loss * self.gaussian_splatting_settings.anisotropy_loss
        # max scale loss
        if self.gaussian_splatting_settings.max_scale_loss is not None:

            @torch.compiler.disable
            def f():
                scales = self.splats['scales']
                max_scale = self.gaussian_splatting_settings.max_scale
                max_log_scales = torch.max(scales, dim=1).values
                max_log_scales_found = torch.exp(max_log_scales)
                max_scale_loss = torch.relu(max_log_scales_found - max_scale).mean()
                return max_scale_loss

            max_scale_loss = f()
            loss_dict["max_scale_loss"] = max_scale_loss
            loss = loss + max_scale_loss * self.gaussian_splatting_settings.max_scale_loss
        # distloss
        if (self.gaussian_splatting_settings.dist_loss is not None and
                self.gaussian_splatting_settings.rasterization_mode == '2dgs'):
            distloss = infos['render_distort'].mean()
            loss_dict["render_distort"] = distloss
            loss = loss + distloss * self.gaussian_splatting_settings.dist_loss
        # face nose l1
        if self.gaussian_splatting_settings.face_nose_l1_loss is not None:
            face_mask = torch.where(segmentation_classes == 3, 1, 0)
            nose_mask = torch.where(segmentation_classes == 9, 1, 0)
            face_nose_mask = torch.logical_or(face_mask, nose_mask)
            face_nose_mask = repeat(face_nose_mask, "cam H W -> cam H W f", f=3)
            face_nose_loss = torch.sum(
                torch.abs(rendered_images - target_images) * face_nose_mask, dim=-1).mean()
            loss_dict["face_nose_l1"] = face_nose_loss
            loss = loss + face_nose_loss * self.gaussian_splatting_settings.face_nose_l1_loss
        # hair l1
        if self.gaussian_splatting_settings.hair_l1_loss is not None:
            hair_mask = torch.where(segmentation_classes == 4, 1, 0)
            hair_mask = repeat(hair_mask, "cam H W -> cam H W f", f=3)
            hair_loss = torch.sum(
                torch.abs(rendered_images - target_images) * hair_mask, dim=-1).mean()
            loss_dict["hair_l1"] = hair_loss
            loss = loss + hair_loss * self.gaussian_splatting_settings.hair_l1_loss
        # neck l1
        if self.gaussian_splatting_settings.neck_l1_loss is not None:
            neck_mask = torch.where(segmentation_classes == 1, 1, 0)
            neck_mask = repeat(neck_mask, "cam H W -> cam H W f", f=3)
            neck_loss = torch.sum(
                torch.abs(rendered_images - target_images) * neck_mask, dim=-1).mean()
            loss_dict["neck_l1"] = neck_loss
            loss = loss + neck_loss * self.gaussian_splatting_settings.neck_l1_loss
        # ears l1
        if self.gaussian_splatting_settings.ears_l1_loss is not None:
            left_ear_mask = torch.where(segmentation_classes == 5, 1, 0)
            right_ear_mask = torch.where(segmentation_classes == 6, 1, 0)
            ears_mask = torch.logical_or(left_ear_mask, right_ear_mask)
            ears_mask = repeat(ears_mask, "cam H W -> cam H W f", f=3)
            ears_loss = torch.sum(
                torch.abs(rendered_images - target_images) * ears_mask, dim=-1).mean()
            loss_dict["ears_l1"] = ears_loss
            loss = loss + ears_loss * self.gaussian_splatting_settings.ears_l1_loss
        # lips l1
        if self.gaussian_splatting_settings.lips_l1_loss is not None:
            upper_lip_nask = torch.where(segmentation_classes == 7, 1, 0)
            lower_lip_mask = torch.where(segmentation_classes == 8, 1, 0)
            lips_mask = torch.logical_or(upper_lip_nask, lower_lip_mask)
            lips_mask = repeat(lips_mask, "cam H W -> cam H W f", f=3)
            lips_loss = torch.sum(
                torch.abs(rendered_images - target_images) * lips_mask, dim=-1).mean()
            loss_dict["lips_l1"] = lips_loss
            loss = loss + lips_loss * self.gaussian_splatting_settings.lips_l1_loss
        # eyes l1
        if self.gaussian_splatting_settings.eyes_l1_loss is not None:
            left_eye_mask = torch.where(segmentation_classes == 10, 1, 0)
            right_eye_mask = torch.where(segmentation_classes == 11, 1, 0)
            eyes_mask = torch.logical_or(left_eye_mask, right_eye_mask)
            eyes_mask = repeat(eyes_mask, "cam H W -> cam H W f", f=3)
            eyes_loss = torch.sum(
                torch.abs(rendered_images - target_images) * eyes_mask, dim=-1).mean()
            loss_dict["eyes_l1"] = eyes_loss
            loss = loss + eyes_loss * self.gaussian_splatting_settings.eyes_l1_loss
        # inner mouth l1
        if self.gaussian_splatting_settings.inner_mouth_l1_loss is not None:
            inner_mouth_mask = torch.where(segmentation_classes == 14, 1, 0)
            inner_mouth_mask = repeat(inner_mouth_mask, "cam H W -> cam H W f", f=3)
            inner_mouth_loss = torch.sum(
                torch.abs(rendered_images - target_images) * inner_mouth_mask, dim=-1).mean()
            loss_dict["inner_mouth_l1"] = inner_mouth_loss
            loss = loss + inner_mouth_loss * self.gaussian_splatting_settings.inner_mouth_l1_loss
        # eyebrows l1
        if self.gaussian_splatting_settings.eyebrows_l1_loss is not None:
            left_eyebrow_mask = torch.where(segmentation_classes == 12, 1, 0)
            right_eyebrow_mask = torch.where(segmentation_classes == 13, 1, 0)
            eyebrows_mask = torch.logical_or(left_eyebrow_mask, right_eyebrow_mask)
            eyebrows_mask = repeat(eyebrows_mask, "cam H W -> cam H W f", f=3)
            eyebrows_loss = torch.sum(
                torch.abs(rendered_images - target_images) * eyebrows_mask, dim=-1).mean()
            loss_dict["eyebrows_l1"] = eyebrows_loss
            loss = loss + eyebrows_loss * self.gaussian_splatting_settings.eyebrows_l1_loss
        # hair ssim
        if self.gaussian_splatting_settings.hair_ssim_loss is not None:
            hair_mask = torch.where(segmentation_classes == 4, 1, 0)
            hair_mask = repeat(hair_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(rendered_images * hair_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images * hair_mask, "cam H W f -> cam f H W")
            hair_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["hair_ssim"] = hair_ssim_loss
            loss = loss + hair_ssim_loss * self.gaussian_splatting_settings.hair_ssim_loss
        # lips ssim
        if self.gaussian_splatting_settings.lips_ssim_loss is not None:
            upper_lip_nask = torch.where(segmentation_classes == 7, 1, 0)
            lower_lip_mask = torch.where(segmentation_classes == 8, 1, 0)
            lips_mask = torch.logical_or(upper_lip_nask, lower_lip_mask)
            lips_mask = repeat(lips_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(rendered_images * lips_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images * lips_mask, "cam H W f -> cam f H W")
            lips_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["lips_ssim"] = lips_ssim_loss
            loss = loss + lips_ssim_loss * self.gaussian_splatting_settings.lips_ssim_loss
        # eyes ssim
        if self.gaussian_splatting_settings.eyes_ssim_loss is not None:
            left_eye_mask = torch.where(segmentation_classes == 10, 1, 0)
            right_eye_mask = torch.where(segmentation_classes == 11, 1, 0)
            eyes_mask = torch.logical_or(left_eye_mask, right_eye_mask)
            eyes_mask = repeat(eyes_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(rendered_images * eyes_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images * eyes_mask, "cam H W f -> cam f H W")
            eyes_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["eyes_ssim"] = eyes_ssim_loss
            loss = loss + eyes_ssim_loss * self.gaussian_splatting_settings.eyes_ssim_loss
        # inner mouth ssim
        if self.gaussian_splatting_settings.inner_mouth_ssim_loss is not None:
            inner_mouth_mask = torch.where(segmentation_classes == 14, 1, 0)
            inner_mouth_mask = repeat(inner_mouth_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(rendered_images * inner_mouth_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images * inner_mouth_mask, "cam H W f -> cam f H W")
            inner_mouth_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["inner_mouth_ssim"] = inner_mouth_ssim_loss
            loss = loss + (
                inner_mouth_ssim_loss * self.gaussian_splatting_settings.inner_mouth_ssim_loss)
        # eyebrows ssim
        if self.gaussian_splatting_settings.eyebrows_ssim_loss is not None:
            left_eyebrow_mask = torch.where(segmentation_classes == 12, 1, 0)
            right_eyebrow_mask = torch.where(segmentation_classes == 13, 1, 0)
            eyebrows_mask = torch.logical_or(left_eyebrow_mask, right_eyebrow_mask)
            eyebrows_mask = repeat(eyebrows_mask * eyebrows_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(rendered_images * eyebrows_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            eyebrows_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["eyebrows_ssim"] = eyebrows_ssim_loss
            loss = loss + eyebrows_ssim_loss * self.gaussian_splatting_settings.eyebrows_ssim_loss

        loss_dict["loss"] = loss
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        return loss_dict

    def log_image(
        self,
        left_image: Float[torch.Tensor, "H W 3"],
        right_image: Float[torch.Tensor, "H W 3"],
        step: int,
        name: str,
    ) -> None:
        """
        Logs an image to tensorboard.

        Args:
            gt_image (torch.Tensor): The ground truth image.
            pred_image (torch.Tensor): The predicted image.
            step (int): The current step.
            name (str): The name of the image.
        """
        canvas = (
            torch.concatenate(
                [left_image.clamp(0, 1), right_image.clamp(0, 1)],
                dim=1,
            ).detach().cpu().numpy())
        canvas = (canvas * 255).astype(np.uint8)
        writer = self.logger.experiment
        writer.add_image(name, canvas, step, dataformats="HWC")

    def training_step(
        self,
        batch: SingleFrameData,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (SingleFrameData): The batch.
            batch_idx (int): The batch index.

        Returns:
            torch.Tensor: The loss.
        """

        # Pause the viewer if needed
        if self.enable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()
        t = time.time()

        # Forward pass
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        for opt in optimizers:
            opt.zero_grad()
        if self.training_mode != 'rigged':
            splat_optimizers = {k: optimizers[i] for i, k in enumerate(self.splat_optimizer_keys)}

        rigging_params = self.rigging_params.forward(batch.sequence_id, batch.time_step)
        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            intrinsics=batch.intrinsics,
            world_2_cam=batch.world_2_cam,
            cam_2_world=None,
            image_height=int(batch.image.shape[1]),
            image_width=int(batch.image.shape[2]),
            color_correction=batch.color_correction,
            cur_sh_degree=self.get_cur_sh_degree(self.step),
            se3_transform=batch.se3_transform,
            camera_indices=batch.camera_indices,
            rigging_params=rigging_params,
        )

        # Pre-backward densification
        if self.training_mode != 'rigged':
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=splat_optimizers,
                state=self.strategy_state,
                step=self.step,
                info=infos['default_infos'],
            )

        # Loss computation and logging
        if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
            denoised_images = rendered_images
        else:
            denoised_images = None

        loss_dict = self.compute_loss(
            rendered_images=infos['default_rendered_images'],
            rendered_alphas=infos['default_rendered_alphas'],
            target_images=batch.image,
            target_alphas=batch.alpha_map,
            target_segmentation_mask=batch.segmentation_mask,
            infos=infos,
            denoised_images=denoised_images,
        )
        loss = loss_dict['loss']

        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))
        if 'per_gaussian_movement' in infos:
            self.log(
                'per_gaussian_movement',
                infos['default_per_gaussian_movement'],
                on_step=True,
                on_epoch=False)

        self.log('num_gaussians', self.splats['means'].shape[0], on_step=True, on_epoch=False)

        # Backward pass and optimization
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        if self.training_mode != 'rigged':
            schedulers.step()

        # Post-backward densification
        if self.training_mode != 'rigged':
            match self.gaussian_splatting_settings.densification_mode:
                case "default":
                    self.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=splat_optimizers,
                        state=self.strategy_state,
                        step=self.step,
                        info=infos['default_infos'],
                    )

                case "monte_carlo_markov_chain":
                    self.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=splat_optimizers,
                        state=self.strategy_state,
                        step=self.step,
                        info=infos['default_infos'],
                        lr=schedulers.get_last_lr()[0],
                    )

                case _:
                    raise ValueError("Unknown densification mode: "
                                     f"{self.gaussian_splatting_settings.densification_mode}")

        # Log images
        if self.step % self.gaussian_splatting_settings.log_images_interval == 0:
            # gt, merged (and optionally denoised)
            self.log_image(
                left_image=batch.image[0],
                right_image=rendered_images[0],
                step=self.step,
                name="train/gt_and_merged",
            )
            # rendered denoised (if screen space denoising is enabled)
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                self.log_image(
                    left_image=infos['pre_denoised_rendered_images'][0],
                    right_image=rendered_images[0],
                    step=self.step,
                    name="train/noisy_and_denoised",
                )

        # Iteration time logging
        time_elapsed = time.time() - t
        its = 1 / time_elapsed
        fps = rendered_images.shape[0] / time_elapsed
        self.log('train/its', its, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/fps', fps, on_step=True, on_epoch=False, prog_bar=True)

        # Resume the viewer if needed
        if self.enable_viewer:
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_step = rendered_images.shape[0] * rendered_images.shape[
                1] * rendered_images.shape[2]
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(self.step, num_train_rays_per_step)

        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: SingleFrameData,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch (SingleFrameData): The batch.
            batch_idx (int): The batch index.

        Returns:
            torch.Tensor: The loss.
        """

        # Pause the viewer if needed
        if self.enable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()
        t = time.time()

        # Forward pass
        rigging_params = self.rigging_params.forward(batch.sequence_id, batch.time_step)
        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            intrinsics=batch.intrinsics,
            world_2_cam=batch.world_2_cam,
            cam_2_world=None,
            image_height=int(batch.image.shape[1]),
            image_width=int(batch.image.shape[2]),
            color_correction=batch.color_correction,
            cur_sh_degree=self.get_cur_sh_degree(self.step),
            se3_transform=batch.se3_transform,
            camera_indices=batch.camera_indices,
            rigging_params=rigging_params,
        )

        # Loss computation and logging
        if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
            denoised_images = rendered_images
        else:
            denoised_images = None

        loss_dict = self.compute_loss(
            rendered_images=infos['default_rendered_images'],
            rendered_alphas=infos['default_rendered_alphas'],
            target_images=batch.image,
            target_alphas=batch.alpha_map,
            target_segmentation_mask=batch.segmentation_mask,
            infos=infos,
            denoised_images=denoised_images,
        )
        loss = loss_dict['loss']

        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))

        # Log images
        if self.step % self.gaussian_splatting_settings.log_images_interval == 0:
            # gt, merged (and optionally denoised)
            self.log_image(
                left_image=batch.image[0],
                right_image=rendered_images[0],
                step=self.step,
                name="val/gt_and_merged",
            )

            # rendered denoised (if screen space denoising is enabled)
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                self.log_image(
                    left_image=infos['pre_denoised_rendered_images'][0],
                    right_image=rendered_images[0],
                    step=self.step,
                    name="val/noisy_and_denoised",
                )

        # Iteration time logging
        time_elapsed = time.time() - t
        its = 1 / time_elapsed
        fps = rendered_images.shape[0] / time_elapsed
        self.log('val/its', its, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val/fps', fps, on_step=True, on_epoch=False, prog_bar=True)

        # Resume the viewer if needed
        if self.enable_viewer:
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_step = rendered_images.shape[0] * rendered_images.shape[
                1] * rendered_images.shape[2]
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(self.step, num_train_rays_per_step)

        return loss

    # ================================================================================ #
    #                                 Viewer                                           #
    # ================================================================================ #

    def start_viewer(
        self,
        port: int | None = None,
        mode: Literal['training', 'rendering'] = 'rendering',
        sequence: int = CANONICAL_SEQUENCE_FRAME[0],
        frame: int | None = CANONICAL_SEQUENCE_FRAME[1],
    ) -> None:
        """ Starts the viewer. """
        self.viewer_sequence = sequence
        self.viewer_frame = frame

        if port is None:

            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]

            port = find_free_port()
        self.server = viser.ViserServer(port=port, verbose=True)
        self.eval()
        sm = SequenceManager(sequence)
        n_frames = len(sm)
        # n_frames = 1 if frame is not None else n_frames
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.render,
            mode=mode,
            num_frames=n_frames,
        )

    # ================================================================================= #
    #                                 Video Loops                                       #
    # ================================================================================= #

    def render_video(
        self,
        intrinsics: Float[torch.Tensor, 'time 3 3'],
        world_2_cam: Float[torch.Tensor, 'time 4 4'],
        image_height: int,
        image_width: int,
        se_transforms: UnbatchedSE3Transform,
        rigging_params: Float[torch.Tensor, 'time n_vertices 3'],
        background: Float[torch.Tensor, '3'] | None = None,
    ) -> UInt8[np.ndarray, 'time H W 3']:
        """
        Renders a video.

        Args:
            intrinsics (torch.Tensor): The intrinsics, shape: `(time, 3, 3)`.
            world_2_cam (torch.Tensor): The world to camera transformation, shape: `(time, 4, 4)`.
            image_height (int): The image height.
            image_width (int): The image width.
            se_transforms (UnbatchedSE3Transform): The SE3 transforms.
            rigging_params (torch.Tensor): The rigging parameters, shape: `(time, n_vertices, 3)`.
            background (torch.Tensor): The background, shape: `(time, 3)`.

        Returns:
            np.ndarray: The video.
        """
        self.eval()
        self.cuda()
        video = torch.zeros((len(intrinsics), image_height, image_width, 3), device='cuda')

        for t in tqdm(range(len(intrinsics)), desc="Rendering video"):
            cur_intrinsics = intrinsics[t:t + 1]
            cur_world_2_cam = world_2_cam[t:t + 1]
            cur_se_transform = UnbatchedSE3Transform(
                rotation=se_transforms.rotation[t:t + 1],
                translation=se_transforms.translation[t:t + 1],
            )
            rigging_params_t = rigging_params[t]

            rendered_images, _, _, _ = self.forward(
                intrinsics=cur_intrinsics,
                world_2_cam=cur_world_2_cam,
                cam_2_world=None,
                image_height=image_height,
                image_width=image_width,
                color_correction=None,
                cur_sh_degree=None,
                se3_transform=cur_se_transform,
                camera_indices=None,
                rigging_params=rigging_params_t,
                background=background,
            )
            video[t] = rendered_images[0]

        video = video * 255
        video = video.cpu().numpy().astype(np.uint8)
        return video


# ==================================================================================== #
#                                 Training Loops                                       #
# ==================================================================================== #


def train_static(
    config_path: str,
    sequence: int = CANONICAL_SEQUENCE_FRAME[0],
    time_step: int = CANONICAL_SEQUENCE_FRAME[1],
    port: int | None = None,
) -> int:
    """
    Train loop for the static mode.

    Args:
        config_path (str): Path to the configuration file.
        sequence (int): The sequence.
        time_step (int): The time step.
        overwrite (bool): Whether to overwrite the existing model.

    Returns:
        int: The port number. Useful when fine-tuning the rigging sparameters.
    """

    # set up model
    config = load_config(config_path)
    model = RiggedGaussianSplatting(
        train_sequences=config.train_sequences,
        gaussian_splatting_settings=config.gaussian_splatting_settings,
        learning_rates=config.learning_rates,
        enable_viewer=config.enable_viewer,
        training_mode='static',
    )
    if config.compile:
        # model._compile()
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    torch.set_float32_matmul_precision('high')

    # sanity checks
    params = model.splats
    optimizers, _ = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get datasets
    train_set = SingleSequenceDataset(
        cameras=TRAIN_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
        length_multiplier=500,
    )
    val_set = SingleSequenceDataset(
        cameras=TEST_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
    )

    # get loaders
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=config.num_train_workers,
        persistent_workers=True)
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        shuffle=False,
        num_workers=config.num_val_workers,
        persistent_workers=True,
    )

    # start viewer
    model.cuda()
    if config.enable_viewer:
        port = model.start_viewer(mode="training", sequence=sequence, frame=time_step, port=port)

    # train
    logger = TensorBoardLogger("tb_logs/rigged_gs/static", name=config.name)
    optimizers, _ = model.configure_optimizers()  # needed to get number of optimizers
    n_optimizers = len(optimizers)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=None,
        max_steps=config.gaussian_splatting_settings.train_iterations * n_optimizers,
        val_check_interval=500,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model, train_loader, val_loader)

    return port


def fine_tune_static(
    config_path: str,
    fine_tuning_steps: int,
    n_additional_gaussians: int,
    sequence: int = CANONICAL_SEQUENCE_FRAME[0],
    time_step: int = CANONICAL_SEQUENCE_FRAME[1],
    port: int | None = None,
) -> int:
    """
    Train loop for the static mode.

    Args:
        config_path (str): Path to the configuration file.
        sequence (int): The sequence.
        time_step (int): The time step.
        overwrite (bool): Whether to overwrite the existing model.

    Returns:
        int: The port number. Useful when fine-tuning the rigging parameters.
    """
    raise NotImplementedError("Fine-tuning static mode is not yet implemented.")
    # Change config params
    config = load_config(config_path)

    # set up model
    model = RiggedGaussianSplatting(
        train_sequences=config.train_sequences,
        gaussian_splatting_settings=config.gaussian_splatting_settings,
        learning_rates=config.learning_rates,
        enable_viewer=config.enable_viewer,
        training_mode='static',
    )
    if config.compile:
        # model._compile()
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    torch.set_float32_matmul_precision('high')

    # sanity checks
    params = model.splats
    optimizers, _ = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get datasets
    train_set = SingleSequenceDataset(
        cameras=TRAIN_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
        length_multiplier=500,
    )
    val_set = SingleSequenceDataset(
        cameras=TEST_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
    )

    # get loaders
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=config.num_train_workers,
        persistent_workers=True)
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        shuffle=False,
        num_workers=config.num_val_workers,
        persistent_workers=True,
    )

    # start viewer
    model.cuda()
    if config.enable_viewer:
        port = model.start_viewer(mode="training", sequence=sequence, frame=time_step, port=port)

    # train
    logger = TensorBoardLogger("tb_logs/rigged_gs/static", name=config.name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=None,
        max_steps=config.gaussian_splatting_settings.train_iterations * model.n_optimizers,
        val_check_interval=500,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model, train_loader, val_loader)

    return port


def fine_tune_rigged_params(
    checkpoint_path: str,
    fine_tuning_steps: int,
    sequence: int = CANONICAL_SEQUENCE_FRAME[0],
    time_step: int = CANONICAL_SEQUENCE_FRAME[1],
    port: int | None = None,
    compile: bool = True,
    num_train_workers: int = 12,
    num_val_workers: int = 4,
    enable_viewer: bool = True,
) -> int:
    """
    Train loop for the static mode.

    Args:
        checkpoint_path (str): Path to the configuration file.
        sequence (int): The sequence.
        time_step (int): The time step.
        overwrite (bool): Whether to overwrite the existing model.

    Returns:
        int: The port number. Useful when fine-tuning the rigging parameters.
    """
    # TODO: vertex adjustments as well!

    # set up model
    # model = RiggedGaussianSplatting.load_from_checkpoint(
    #     checkpoint_path, ckpt_path=checkpoint_path, training_mode='rigged')
    model = RiggedGaussianSplatting.load_from_checkpoint(
        checkpoint_path, ckpt_path=checkpoint_path, training_mode='rigged')
    gaussian_splatting_settings = model.gaussian_splatting_settings

    if compile:
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    torch.set_float32_matmul_precision('high')

    # get datasets
    train_set = SingleSequenceDataset(
        cameras=TRAIN_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=gaussian_splatting_settings.camera_batch_size,
        length_multiplier=500,
    )
    val_set = SingleSequenceDataset(
        cameras=TEST_CAMS,
        sequence=sequence,
        start_idx=time_step,
        end_idx=time_step + 1,
        n_cameras_per_frame=gaussian_splatting_settings.camera_batch_size,
    )

    # get loaders
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=num_train_workers,
        persistent_workers=True)
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        shuffle=False,
        num_workers=num_val_workers,
        persistent_workers=True,
    )

    # start viewer
    model.cuda()
    if enable_viewer:
        port = model.start_viewer(mode="training", sequence=sequence, frame=time_step, port=port)

    # train
    logger = TensorBoardLogger(
        "tb_logs/rigged_gs/fine_tuning", name=f'seq_{sequence}_f_{time_step}')
    optimizers = model.configure_optimizers()  # needed to get number of optimizers
    n_optimizers = len(optimizers)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=None,
        max_steps=fine_tuning_steps * n_optimizers,
        val_check_interval=500,
        check_val_every_n_epoch=None,
    )
    # clear cuda cache
    torch.cuda.empty_cache()
    trainer.fit(model, train_loader, val_loader)

    return port


def train_dynamic(
    checkpoint_path: str,
    sequence: int,
    compile: bool = True,
    num_train_workers: int = 12,
    num_val_workers: int = 4,
    enable_viewer: bool = True,
    port: int | None = None,
) -> int:
    """
    Train loop for the dynamic mode for a single sequence.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        int: The port number. Useful when fine-tuning the rigging parameters.
    """
    # set up model
    model = RiggedGaussianSplatting.load_from_checkpoint(
        checkpoint_path,
        ckpt_path=checkpoint_path,
        training_mode='static',  # we don't fine tune deformation fields yet
    )
    gaussian_splatting_settings = model.gaussian_splatting_settings
    gaussian_splatting_settings.cap_max = 100_000  # TODO: hard coded fix
    if compile:
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    torch.set_float32_matmul_precision('high')

    # sanity checks
    params = model.splats
    optimizers, _ = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get datasets
    train_set = SequentialMultiSequenceDataset(
        sequences=TRAIN_SEQUENCES,
        cameras=TRAIN_CAMS,
        n_cameras_per_frame=gaussian_splatting_settings.camera_batch_size,
    )
    val_set = SequentialMultiSequenceDataset(
        sequences=TRAIN_SEQUENCES,
        cameras=TEST_CAMS,
        n_cameras_per_frame=gaussian_splatting_settings.camera_batch_size,
    )

    # get loaders
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=False,  # lets try to learn it in order
        num_workers=num_train_workers,
        persistent_workers=True)
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        shuffle=True,
        num_workers=num_val_workers,
        persistent_workers=True,
    )

    # start viewer
    model.cuda()
    if enable_viewer:
        port = model.start_viewer(mode="training", sequence=sequence, frame=0, port=port)

    # train
    logger = TensorBoardLogger("tb_logs/rigged_gs/dynamic", name=f'seq_{sequence}')
    optimizers, _ = model.configure_optimizers()  # needed to get number of optimizers
    n_optimizers = len(optimizers)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=None,
        max_steps=n_optimizers * 100_000,  # TODO: remove hard coding!!
        val_check_interval=250,
        check_val_every_n_epoch=None,
        limit_val_batches=10,
    )
    trainer.fit(model, train_loader, val_loader)

    return port


# ==================================================================================== #
#                                 MAIN FUNCTION                                        #
# ==================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting model on sequences.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/rigged_gs.yml",
        help="Path to the configuration file.")
    parser.add_argument(
        "-v", "--visualize", type=str, help="Path to checkpoint file to visualize.")
    parser.add_argument("-f", "--finetune", type=str, help="Path to checkpoint file to fine_tune.")
    parser.add_argument(
        "-td", "--train_dynamic", type=str, help="Path to checkpoint file to train_dynamic.")
    args = parser.parse_args()

    if args.visualize:
        model = RiggedGaussianSplatting.load_from_checkpoint(
            args.visualize, ckpt_path=args.visualize)
        model.cuda()
        print("Starting viewer...")
        model.start_viewer(mode='rendering', sequence=79)
        print("To exit, press Ctrl+C.")
        time.sleep(100000)

    elif args.finetune:
        print("Starting fine-tuning...")
        fine_tune_rigged_params(
            args.finetune,
            fine_tuning_steps=1000,
            sequence=3,
            time_step=0,
        )

    elif args.train_dynamic:
        print("Starting dynamic training...")
        train_dynamic(args.train_dynamic, sequence=3, enable_viewer=True)

    else:
        if args.config is not None:
            if not (args.config.endswith(".yml") or args.config.endswith(".yaml")):
                args.config = f'configs/{args.config}.yml'
        print("Using config file:", args.config)
        # for debugging only
        train_static(args.config)
        print("Starting training...")
