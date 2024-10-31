""" Gaussian Splatting on a single time step. Mostly for debugging. """

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
from jaxtyping import Float, Int
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thesis.config import GaussianSplattingSettings, load_config
from thesis.constants import (
    DEFAULT_SE3_ROTATION,
    DEFAULT_SE3_TRANSLATION,
    TEST_CAMS,
    TRAIN_CAMS,
)
from thesis.data_management import SequenceManager, SingleSequenceDataset
from thesis.data_management.data_classes import SingleFrameData, UnbatchedSE3Transform
from thesis.deformation_field.direct_prediction import DirectPrediction
from thesis.gaussian_splatting.camera_color_correction import LearnableColorCorrection
from thesis.gaussian_splatting.initialize_splats import (
    flame_initialization,
    point_cloud_initialization,
    pre_trained_initialization,
    random_initialization,
)
from thesis.gaussian_splatting.view_dependent_coloring import ViewDependentColorMLP
from thesis.utils import (
    apply_se3_to_orientation,
    apply_se3_to_point,
    assign_segmentation_class,
)


class GaussianSplattingSingleFrame(pl.LightningModule):
    """Gaussian splatting on a single time step."""

    def __init__(
        self,
        sequence: int,
        frame_idx: int,
        gaussian_splatting_settings: GaussianSplattingSettings | DictConfig,
        learning_rates: DictConfig,
        enable_viewer: bool = True,
        ckpt_path: str | None = None,
    ) -> None:
        """
        Initializes the Gaussian splatting model.

        Args:
            sequence (int): Sequence number.
            frame (int): Frame number.
            gaussian_splatting_settings (GaussianSplattingSettings | DictConfig): Gaussian
                splatting settings.
            learning_rates (DictConfig): Learning rates.
            enable_viewer (bool): Whether to enable the viewer.
            checkpoint_path (str | None): Path to the checkpoint. Needs to be provided if
                loading a pre-trained model, since torch will throw size mismatch errors
                otherwise.
        """
        super().__init__()
        self.save_hyperparameters()
        frame = frame_idx
        self.automatic_optimization = False

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
                # load flame params
                sequence_manager = SequenceManager(sequence)
                flame_params = sequence_manager.flame_params[frame:frame + 1]
                self.splats = nn.ParameterDict(
                    flame_initialization(
                        flame_params=flame_params,
                        scene_scale=gaussian_splatting_settings.scene_scale,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_spherical_harmonics=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
            case "pre_trained":
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

        # Latent adjustments (pre-processing)
        match gaussian_splatting_settings.latent_adjustments_mode:
            case "none":
                pass
            case "direct_prediction":
                self.latent_adjustments = DirectPrediction(
                    window_size=gaussian_splatting_settings.prior_window_size,
                    per_gaussian_latent_dim=gaussian_splatting_settings.feature_dim,
                    use_audio_latents=gaussian_splatting_settings
                    .latent_adjustments_use_audio_latents,
                    use_per_gaussian_latents=gaussian_splatting_settings
                    .latent_adjustments_use_per_gaussian_latents,
                    use_flame_params=gaussian_splatting_settings
                    .latent_adjustments_use_flame_params,
                )
            case _:
                raise ValueError("Unknown latent adjustments mode: "
                                 f"{gaussian_splatting_settings.latent_adjustments}")

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

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            A tuple containing the splat optimizers and other optimizers.
        """

        # splat optimizers
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
        splat_optimizers = {
            name:
                Adam(
                    [{
                        "params": self.splats[name],
                        "lr": lr,
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
                lr=self.learning_rates.sh0_lr * batch_scaling,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )
            splat_optimizers["shN"] = Adam(
                [self.splats["shN"]],
                lr=(self.learning_rates.sh0_lr / 20) * batch_scaling,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )
        else:
            splat_optimizers['colors'] = Adam(
                [self.splats['colors']],
                lr=self.learning_rates.color_lr * batch_scaling,
                eps=1e-15 / batch_scaling,
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )

        # other optimizers
        other_optimizers = {}
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

        if hasattr(self, "learnable_color_correction"):
            other_optimizers["learnable_color_correction"] = Adam(
                self.learnable_color_correction.parameters(),
                lr=self.learning_rates.color_correction_lr * batch_scaling,
            )
        if hasattr(self, "latent_adjustments"):
            other_optimizers["latent_adjustments"] = Adam(
                self.latent_adjustments.parameters(),
                lr=self.learning_rates.latent_adjustments_lr * batch_scaling,
            )

        # schedulers
        schedulers = {}
        schedulers["means"] = torch.optim.lr_scheduler.ExponentialLR(
            splat_optimizers["means"],
            gamma=0.01**(1.0 / self.gaussian_splatting_settings.train_iterations))
        optimizer_list = list(splat_optimizers.values()) + list(other_optimizers.values())
        self.splat_optimizer_keys = list(splat_optimizers.keys())
        scheduler_list = list(schedulers.values())
        return optimizer_list, scheduler_list

    # ================================================================================ #
    #                                 Rasterization                                    #
    # ================================================================================ #

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
        background: Float[torch.Tensor, "cam H W 3"] | None = None,
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
            background (torch.Tensor): Background color, shape: `(cam, H, W, 3)`. If `None`,
                the default background is used.
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
        means = self.splats["means"]
        quats = self.splats["quats"]
        if background is None:
            background = self.default_background
            background.to(means.device)
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
        match self.gaussian_splatting_settings.latent_adjustments_mode:
            case "none":
                features = self.splats["features"]
            case "direct_prediction":
                features = self.latent_adjustments.forward(
                    means=means, per_gaussian_latents=self.splats["features"])

        if se3_transform is not None:
            rotation = se3_transform.rotation
            translation = se3_transform.translation
        else:
            rotation = DEFAULT_SE3_ROTATION.unsqueeze(0).cuda()
            translation = DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda()
        means = apply_se3_to_point(rotation, translation, means)
        quats = apply_se3_to_orientation(rotation, quats)
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.view_dependent_color_mlp.forward(features, camera_indices, means,
                                                           self.splats['colors'], cam_2_world,
                                                           cur_sh_degree)
        else:
            colors = colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        # ------------------------------- Rasterization ------------------------------- #
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
            packed=False,
        )
        match self.gaussian_splatting_settings.rasterization_mode:
            case "default" | "3dgs":
                images, alphas, new_infos = ret
            case "2dgs":
                images, alphas, _, _, _, _, new_infos = ret
        infos = infos | new_infos
        depth_maps = images[:, :, :, 3:]  # get depth maps
        images = images[:, :, :, :3]  # get RGB channels

        # ------------------------------- Post-processing ------------------------------ #
        # Apply screen-space denoising
        infos['raw_rendered_images'] = images
        images = self.screen_space_denoiser(images, alphas)

        # Apply background
        background = repeat(
            background, "f -> cam H W f", cam=images.shape[0], H=image_height, W=image_width)
        images = images*alphas + (1-alphas) * background

        # Apply color correction
        if (color_correction is not None and
                self.gaussian_splatting_settings.camera_color_correction):
            # Reshape color_correction to (batch * height * width, 3, 3)
            batch_size, height, width, _ = images.shape
            color_correction = color_correction.expand(batch_size * height * width, -1, -1)

            # Reshape images to (batch * height * width, 3, 1)
            images = images.view(batch_size, height, width, 3, 1)
            images = images.permute(0, 1, 2, 4, 3).contiguous()
            images = images.view(-1, 3, 1)

            # Perform batch matrix multiplication
            corrected_colors = torch.bmm(color_correction, images)

            # Reshape the result back to (batch, height, width, 3)
            corrected_colors = corrected_colors.view(batch_size, height, width, 1, 3)
            corrected_colors = corrected_colors.permute(0, 1, 2, 4, 3).contiguous()
            corrected_colors = corrected_colors.view(batch_size, height, width, 3)
            images = corrected_colors

        # Apply learnable color correction
        if hasattr(self, "learnable_color_correction") and camera_indices is not None:
            images = self.learnable_color_correction.forward(camera_indices, images)

        return images, alphas, depth_maps, infos

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
        # maybe flip them here?
        hacky_world_transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                         dtype=np.float32)
        c2w = hacky_world_transform @ c2w
        cam_2_world = torch.tensor(c2w).unsqueeze(0).float().cuda()
        intrinsics = torch.tensor(camera_state.get_K(img_wh)).unsqueeze(0).float().cuda()
        se3 = UnbatchedSE3Transform(
            rotation=DEFAULT_SE3_ROTATION.unsqueeze(0).cuda(),
            translation=DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda(),
        )
        image, _, depth, _ = self.forward(
            intrinsics=intrinsics,
            world_2_cam=None,
            cam_2_world=cam_2_world,
            image_height=image_height,
            image_width=image_width,
            se3_transform=se3,
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
            depth = cmap(depth)[:, :, :3]
            return depth

    @torch.no_grad()
    def render_depth(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
    ) -> Float[np.ndarray, "H W"]:
        """Renders the depth image."""
        raise NotImplementedError

    # ================================================================================ #
    #                                 Train / Val Steps                                #
    # ================================================================================ #

    def get_cur_sh_degree(self, step: int) -> int:
        """Returns the current spherical harmonic degree."""
        return min(
            step // self.gaussian_splatting_settings.sh_increase_interval,
            self.max_sh_degree,
        )

    def compute_loss(
        self,
        rendered_images: Float[torch.Tensor, "cam H W 3"],
        rendered_alphas: Float[torch.Tensor, "cam H W 1"],
        target_images: Float[torch.Tensor, "cam H W 3"],
        target_alphas: Float[torch.Tensor, "cam H W"],
        target_segmentation_mask: Float[torch.Tensor, "cam H W 3"],
        infos: dict,
    ) -> dict[str, Float[torch.Tensor, '']]:
        """ Computes the loss. """
        # set up
        loss_dict = {}
        loss = 0.0
        segmentation_classes = assign_segmentation_class(target_segmentation_mask)
        if self.gaussian_splatting_settings.jumper_is_background:
            jumper_mask = torch.where(segmentation_classes == 2, 1, 0)
            target_alphas = target_alphas * (1-jumper_mask)
        alpha_map = repeat(target_alphas, "cam H W -> cam H W f", f=3)
        background = infos['background']
        background = repeat(
            background,
            "f -> cam H W f",
            cam=alpha_map.shape[0],
            H=alpha_map.shape[1],
            W=alpha_map.shape[2])
        raw_rendered_images = infos['raw_rendered_images']
        target_images = target_images*alpha_map + (1-alpha_map) * background

        # raw l1 foreground loss
        if self.gaussian_splatting_settings.l1_image_loss is not None:
            l1_foreground_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images), dim=-1).mean()
            loss_dict["l1_foreground_loss"] = l1_foreground_loss
            loss = loss + l1_foreground_loss * self.gaussian_splatting_settings.l1_image_loss
        # raw ssim foreground loss
        if self.gaussian_splatting_settings.ssim_image_loss is not None:
            pred = rearrange(raw_rendered_images, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            ssim_foreground_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["ssim_foreground_loss"] = ssim_foreground_loss
            loss = loss + ssim_foreground_loss * self.gaussian_splatting_settings.ssim_image_loss
        # denoised ssim foreground loss
        if self.gaussian_splatting_settings.ssim_denoised_image_loss is not None:
            pred = rearrange(rendered_images, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            ssim_denoised_foreground_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["ssim_denoised_foreground_loss"] = ssim_denoised_foreground_loss
            loss = loss + ssim_denoised_foreground_loss \
                * self.gaussian_splatting_settings.ssim_denoised_image_loss
        # lpips foreground loss
        if self.gaussian_splatting_settings.lpips_image_loss is not None:
            pred = rearrange(raw_rendered_images, "cam H W f -> cam f H W").clip(0, 1)
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
                torch.abs(raw_rendered_images - target_images) * face_nose_mask, dim=-1).mean()
            loss_dict["face_nose_l1"] = face_nose_loss
            loss = loss + face_nose_loss * self.gaussian_splatting_settings.face_nose_l1_loss
        # hair l1
        if self.gaussian_splatting_settings.hair_l1_loss is not None:
            hair_mask = torch.where(segmentation_classes == 4, 1, 0)
            hair_mask = repeat(hair_mask, "cam H W -> cam H W f", f=3)
            hair_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * hair_mask, dim=-1).mean()
            loss_dict["hair_l1"] = hair_loss
            loss = loss + hair_loss * self.gaussian_splatting_settings.hair_l1_loss
        # neck l1
        if self.gaussian_splatting_settings.neck_l1_loss is not None:
            neck_mask = torch.where(segmentation_classes == 1, 1, 0)
            neck_mask = repeat(neck_mask, "cam H W -> cam H W f", f=3)
            neck_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * neck_mask, dim=-1).mean()
            loss_dict["neck_l1"] = neck_loss
            loss = loss + neck_loss * self.gaussian_splatting_settings.neck_l1_loss
        # ears l1
        if self.gaussian_splatting_settings.ears_l1_loss is not None:
            left_ear_mask = torch.where(segmentation_classes == 5, 1, 0)
            right_ear_mask = torch.where(segmentation_classes == 6, 1, 0)
            ears_mask = torch.logical_or(left_ear_mask, right_ear_mask)
            ears_mask = repeat(ears_mask, "cam H W -> cam H W f", f=3)
            ears_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * ears_mask, dim=-1).mean()
            loss_dict["ears_l1"] = ears_loss
            loss = loss + ears_loss * self.gaussian_splatting_settings.ears_l1_loss
        # lips l1
        if self.gaussian_splatting_settings.lips_l1_loss is not None:
            upper_lip_nask = torch.where(segmentation_classes == 7, 1, 0)
            lower_lip_mask = torch.where(segmentation_classes == 8, 1, 0)
            lips_mask = torch.logical_or(upper_lip_nask, lower_lip_mask)
            lips_mask = repeat(lips_mask, "cam H W -> cam H W f", f=3)
            lips_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * lips_mask, dim=-1).mean()
            loss_dict["lips_l1"] = lips_loss
            loss = loss + lips_loss * self.gaussian_splatting_settings.lips_l1_loss
        # eyes l1
        if self.gaussian_splatting_settings.eyes_l1_loss is not None:
            left_eye_mask = torch.where(segmentation_classes == 10, 1, 0)
            right_eye_mask = torch.where(segmentation_classes == 11, 1, 0)
            eyes_mask = torch.logical_or(left_eye_mask, right_eye_mask)
            eyes_mask = repeat(eyes_mask, "cam H W -> cam H W f", f=3)
            eyes_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * eyes_mask, dim=-1).mean()
            loss_dict["eyes_l1"] = eyes_loss
            loss = loss + eyes_loss * self.gaussian_splatting_settings.eyes_l1_loss
        # inner mouth l1
        if self.gaussian_splatting_settings.inner_mouth_l1_loss is not None:
            inner_mouth_mask = torch.where(segmentation_classes == 14, 1, 0)
            inner_mouth_mask = repeat(inner_mouth_mask, "cam H W -> cam H W f", f=3)
            inner_mouth_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * inner_mouth_mask, dim=-1).mean()
            loss_dict["inner_mouth_l1"] = inner_mouth_loss
            loss = loss + inner_mouth_loss * self.gaussian_splatting_settings.inner_mouth_l1_loss
        # eyebrows l1
        if self.gaussian_splatting_settings.eyebrows_l1_loss is not None:
            left_eyebrow_mask = torch.where(segmentation_classes == 12, 1, 0)
            right_eyebrow_mask = torch.where(segmentation_classes == 13, 1, 0)
            eyebrows_mask = torch.logical_or(left_eyebrow_mask, right_eyebrow_mask)
            eyebrows_mask = repeat(eyebrows_mask, "cam H W -> cam H W f", f=3)
            eyebrows_loss = torch.sum(
                torch.abs(raw_rendered_images - target_images) * eyebrows_mask, dim=-1).mean()
            loss_dict["eyebrows_l1"] = eyebrows_loss
            loss = loss + eyebrows_loss * self.gaussian_splatting_settings.eyebrows_l1_loss
        # hair ssim
        if self.gaussian_splatting_settings.hair_ssim_loss is not None:
            hair_mask = torch.where(segmentation_classes == 4, 1, 0)
            hair_mask = repeat(hair_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(raw_rendered_images * hair_mask, "cam H W f -> cam f H W")
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
            pred = rearrange(raw_rendered_images * lips_mask, "cam H W f -> cam f H W")
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
            pred = rearrange(raw_rendered_images * eyes_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images * eyes_mask, "cam H W f -> cam f H W")
            eyes_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["eyes_ssim"] = eyes_ssim_loss
            loss = loss + eyes_ssim_loss * self.gaussian_splatting_settings.eyes_ssim_loss
        # inner mouth ssim
        if self.gaussian_splatting_settings.inner_mouth_ssim_loss is not None:
            inner_mouth_mask = torch.where(segmentation_classes == 14, 1, 0)
            inner_mouth_mask = repeat(inner_mouth_mask, "cam H W -> cam H W f", f=3)
            pred = rearrange(raw_rendered_images * inner_mouth_mask, "cam H W f -> cam f H W")
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
            pred = rearrange(raw_rendered_images * eyebrows_mask, "cam H W f -> cam f H W")
            tgt = rearrange(target_images, "cam H W f -> cam f H W")
            eyebrows_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["eyebrows_ssim"] = eyebrows_ssim_loss
            loss = loss + eyebrows_ssim_loss * self.gaussian_splatting_settings.eyebrows_ssim_loss

        loss_dict["loss"] = loss
        return loss_dict

    def training_step(self, batch: SingleFrameData, batch_idx: int) -> torch.Tensor:
        """Training step."""

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
        splat_optimizers = {k: optimizers[i] for i, k in enumerate(self.splat_optimizer_keys)}
        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            intrinsics=batch.intrinsics,
            world_2_cam=batch.world_2_cam,
            cam_2_world=None,
            image_height=int(batch.image.shape[1]),
            image_width=int(batch.image.shape[2]),
            color_correction=batch.color_correction,
            cur_sh_degree=self.get_cur_sh_degree(self.global_step),
            se3_transform=batch.se3_transform,
            camera_indices=batch.camera_indices,
        )

        # Pre-backward densification
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=splat_optimizers,
            state=self.strategy_state,
            step=self.global_step,
            info=infos,
        )

        # Loss computation and logging
        loss_dict = self.compute_loss(
            rendered_images=rendered_images,
            rendered_alphas=rendered_alphas,
            target_images=batch.image,
            target_alphas=batch.alpha_map,
            target_segmentation_mask=batch.segmentation_mask,
            infos=infos,
        )
        loss = loss_dict["loss"]
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))

        # Backward pass and optimization
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        schedulers.step()

        # Post-backward densification
        match self.gaussian_splatting_settings.densification_mode:
            case "default":
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=splat_optimizers,
                    state=self.strategy_state,
                    step=self.global_step,
                    info=infos,
                )
            case "monte_carlo_markov_chain":
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=splat_optimizers,
                    state=self.strategy_state,
                    step=self.global_step,
                    info=infos,
                    lr=schedulers.get_last_lr()[0],
                )
            case _:
                raise ValueError("Unknown densification mode: "
                                 f"{self.gaussian_splatting_settings.densification_mode}")

        # Log images
        if self.global_step % self.gaussian_splatting_settings.log_images_interval == 0:
            # denoised images
            canvas = (
                torch.concatenate(
                    [batch.image[0].clamp(0, 1), rendered_images[0].clamp(0, 1)],
                    dim=1,
                ).detach().cpu().numpy())
            canvas = (canvas * 255).astype(np.uint8)
            writer = self.logger.experiment
            writer.add_image("train/denoised_render", canvas, self.global_step, dataformats="HWC")
            # raw render
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                canvas = (
                    torch.concatenate(
                        [batch.image[0].clamp(0, 1), infos['raw_rendered_images'][0].clamp(0, 1)],
                        dim=1,
                    ).detach().cpu().numpy())
                canvas = (canvas * 255).astype(np.uint8)
                writer.add_image("train/raw_render", canvas, self.global_step, dataformats="HWC")

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
            self.viewer.update(self.global_step, num_train_rays_per_step)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: SingleFrameData, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Pause the viewer if needed
        if self.enable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            intrinsics=batch.intrinsics,
            world_2_cam=batch.world_2_cam,
            cam_2_world=None,
            image_height=int(batch.image.shape[1]),
            image_width=int(batch.image.shape[2]),
            color_correction=batch.color_correction,
            cur_sh_degree=self.max_sh_degree,
            se3_transform=batch.se3_transform,
            camera_indices=batch.camera_indices,
        )

        # Loss computation and logging
        loss_dict = self.compute_loss(
            rendered_images=rendered_images,
            rendered_alphas=rendered_alphas,
            target_images=batch.image,
            target_alphas=batch.alpha_map,
            target_segmentation_mask=batch.segmentation_mask,
            infos=infos,
        )
        loss = loss_dict["loss"]
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))

        # Log denoised images
        canvas = (
            torch.concatenate(
                [batch.image[0].clamp(0, 1), rendered_images[0].clamp(0, 1)],
                dim=1,
            ).detach().cpu().numpy())
        canvas = (canvas * 255).astype(np.uint8)
        writer = self.logger.experiment
        writer.add_image("val/denoised_render", canvas, self.global_step, dataformats="HWC")
        # Log raw renders
        if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
            canvas = (
                torch.concatenate(
                    [batch.image[0].clamp(0, 1), infos['raw_rendered_images'][0].clamp(0, 1)],
                    dim=1,
                ).detach().cpu().numpy())
            canvas = (canvas * 255).astype(np.uint8)
            writer.add_image("val/raw_render", canvas, self.global_step, dataformats="HWC")

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
            self.viewer.update(self.global_step, num_train_rays_per_step)

        return loss

    # ================================================================================ #
    #                                 Viewer                                           #
    # ================================================================================ #

    def start_viewer(self,
                     port: int | None = None,
                     mode: Literal['training', 'rendering'] = 'rendering') -> None:
        """ Starts the viewer. """
        if port is None:

            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]

            port = find_free_port()
        self.server = viser.ViserServer(port=port, verbose=True)
        self.eval()
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.render,
            mode=mode,
            num_frames=1,
        )


# ==================================================================================== #
#                                 Train 8 Loop                                          #
# ==================================================================================== #


def train(config_path: str) -> None:
    """ Train loop for the static single frame model. """

    # set up model
    config = load_config(config_path)
    model = GaussianSplattingSingleFrame(
        sequence=config.sequence,
        frame_idx=config.frame,
        gaussian_splatting_settings=config.gaussian_splatting_settings,
        learning_rates=config.learning_rates,
        enable_viewer=config.enable_viewer,
    )
    if config.compile:
        # model._compile()
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    torch.set_float32_matmul_precision('high')

    # sanity check
    params = model.splats
    optimizers, schedulers = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get dataloaders
    train_set = SingleSequenceDataset(
        cameras=TRAIN_CAMS,
        sequence=config.sequence,
        start_idx=config.frame,
        end_idx=config.frame + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
        length_multiplier=500,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=config.num_train_workers,
        persistent_workers=True)
    val_set = SingleSequenceDataset(
        cameras=TEST_CAMS,
        sequence=config.sequence,
        start_idx=config.frame,
        end_idx=config.frame + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
    )
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
        model.start_viewer(mode="training")

    # train
    logger = TensorBoardLogger("tb_logs/single_frame", name=config.name)
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.gaussian_splatting_settings.train_iterations,
    )
    trainer.fit(model, train_loader, val_loader)


# ==================================================================================== #
#                                 MAIN FUNCTION                                        #
# ==================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Gaussian Splatting model on a single frame.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/single_frame.yml",
        help="Path to the configuration file.")
    parser.add_argument(
        "-v", "--visualize", type=str, help="Path to checkpoint file to visualize.")
    args = parser.parse_args()
    if args.visualize:
        model = GaussianSplattingSingleFrame.load_from_checkpoint(
            args.visualize, ckpt_path=args.visualize)
        model.cuda()
        print("Starting viewer...")
        model.start_viewer(mode='rendering')
        print("To exit, press Ctrl+C.")
        time.sleep(100000)

    else:
        train(args.config)
        print("Starting training...")
