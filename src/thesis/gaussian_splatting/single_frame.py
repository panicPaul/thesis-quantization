""" Gaussian Splatting on a single time step. Mostly for debugging. """

import argparse
import math
import socket
import time
from functools import partial
from typing import Literal

import lightning as pl
import nerfview
import numpy as np
import torch
import viser
from einops import rearrange, repeat
from gsplat import DefaultStrategy, MCMCStrategy, rasterization, rasterization_2dgs
from jaxtyping import Float
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thesis.config import GaussianSplattingSettings, load_config
from thesis.constants import DEFAULT_SE3_ROTATION, DEFAULT_SE3_TRANSLATION
from thesis.data_management import SingleSequenceDataset
from thesis.data_management.data_classes import SingleFrameData, UnbatchedSE3Transform
from thesis.gaussian_splatting.initialize_splats import (
    point_cloud_initialization,
    random_initialization,
)
from thesis.gaussian_splatting.view_dependent_coloring import ViewDependentColorMLP
from thesis.utils import apply_se3_to_orientation, apply_se3_to_point


class GaussianSplattingSingleFrame(pl.LightningModule):
    """Gaussian splatting on a single time step."""

    def __init__(
        self,
        gaussian_splatting_settings: GaussianSplattingSettings | DictConfig,
        learning_rates: DictConfig,
        enable_viewer: bool = True,
    ) -> None:
        """Initializes the Gaussian splatting model."""
        super().__init__()
        self.save_hyperparameters()
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
        match gaussian_splatting_settings.initialization_mode:
            case "random":
                self.splats = nn.ParameterDict(
                    random_initialization(
                        num_splats=gaussian_splatting_settings.initialization_points,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_colors=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
            case "point_cloud":
                self.splats = nn.ParameterDict(
                    point_cloud_initialization(
                        num_splats=gaussian_splatting_settings.initialization_points,
                        feature_dim=gaussian_splatting_settings.feature_dim,
                        colors_sh_degree=gaussian_splatting_settings.sh_degree,
                        initialize_colors=not gaussian_splatting_settings
                        .use_view_dependent_color_mlp,
                    ))
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

        # View-dependent color module
        if gaussian_splatting_settings.use_view_dependent_color_mlp:
            self.view_dependent_color_mlp = ViewDependentColorMLP(
                feature_dim=gaussian_splatting_settings.feature_dim,
                sh_degree=gaussian_splatting_settings.sh_degree,
            )

        # Screen-space denoising
        match gaussian_splatting_settings.screen_space_denoising_mode:
            case 'none':
                self.screen_space_denoiser = lambda img, alphas: img
            case _:
                raise ValueError("Unknown screen-space denoising mode: "
                                 f"{gaussian_splatting_settings.screen_space_denoising_mode}")

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
                    dist_loss=gaussian_splatting_settings.dist_loss,
                )
            case _:
                raise ValueError("Unknown rasterization mode: "
                                 f"{gaussian_splatting_settings.rasterization_mode}")

        # Set up loss functions
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=gaussian_splatting_settings.lpips_network, normalize=False)

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            A tuple containing the splat optimizers and other optimizers.
        """
        batch_size = self.gaussian_splatting_settings.camera_batch_size
        scaling = math.sqrt(batch_size)
        splat_optimizers = {
            name:
                Adam(
                    [{
                        "params": self.splats[name],
                        "lr": self.learning_rates[f"{name}_lr"] * math.sqrt(batch_size),
                        "name": name,
                    }],
                    eps=1e-15 / math.sqrt(batch_size),
                    betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
                ) for name in ["means", "scales", "quats", "opacities", "features"]
        }
        if not self.gaussian_splatting_settings.use_view_dependent_color_mlp:
            splat_optimizers["colors"] = Adam(
                [self.splats["colors"]],
                lr=self.learning_rates.color_lr * scaling,
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1-0.9), 1 - batch_size * (1-0.999)),
            )
        other_optimizers = {}
        if hasattr(self, "view_dependent_color_mlp"):
            other_optimizers["view_dependent_color_mlp"] = Adam(
                self.view_dependent_color_mlp.parameters(),
                lr=self.learning_rates.color_mlp_lr * scaling,
            )
        schedulers = {}
        schedulers["means"] = torch.optim.lr_scheduler.ExponentialLR(
            splat_optimizers["means"],
            gamma=0.01**(1.0 / self.gaussian_splatting_settings.train_iterations))
        optimizer_list = list(splat_optimizers.values()) + list(other_optimizers.values())
        self.splat_optimizer_keys = list(splat_optimizers.keys())
        scheduler_list = list(schedulers.values())
        return optimizer_list, scheduler_list

    def _compile(self, mode: Literal['default', 'max-autotune'] = "default"):
        """Compile some of the submodules."""
        if hasattr(self, "view_dependent_color_mlp"):
            self.view_dependent_color_mlp = self.view_dependent_color_mlp.compile(mode=mode)

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
    ) -> tuple[Float[torch.Tensor, "cam H W 3"], Float[torch.Tensor, "cam H W 1"], dict]:
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

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): RGB images, shape: `(cam, H, W, 3)`.
                - (*torch.Tensor*): Alphas, shape: `(cam, H, W)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """

        # Set up
        if cur_sh_degree is None:
            cur_sh_degree = self.max_sh_degree
        means = self.splats["means"]
        quats = self.splats["quats"]
        features = self.splats["features"]
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
        if se3_transform is not None:
            rotation = se3_transform.rotation
            translation = se3_transform.translation
        else:
            rotation = DEFAULT_SE3_ROTATION.unsqueeze(0).cuda()
            translation = DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda()
        means = apply_se3_to_point(rotation, translation, means)
        quats = apply_se3_to_orientation(rotation, quats)
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.view_dependent_color_mlp.forward(features, means, cam_2_world,
                                                           cur_sh_degree)
        else:
            colors = self.splats["colors"]

        # ------------------------------- Rasterization ------------------------------- #
        images, alphas, new_infos = self.rasterize(
            means=means,
            quats=quats,
            scales=self.splats["scales"],
            opacities=self.splats["opacities"],
            colors=colors,
            render_mode="RGB",
            viewmats=world_2_cam,
            Ks=intrinsics,
            width=image_width,
            height=image_height,
            sh_degree=cur_sh_degree if not hasattr(self, "view_dependent_color_mlp") else None,
        )
        infos = infos | new_infos

        # ------------------------------- Post-processing ------------------------------ #
        # Apply screen-space denoising
        infos['rasterized_images'] = images
        images = self.screen_space_denoiser(images, alphas)

        # Apply background
        background = repeat(
            background, "f -> cam H W f", cam=images.shape[0], H=image_height, W=image_width)
        images = images*alphas + (1-alphas) * background

        # Apply color correction
        if color_correction is not None:
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
        return images, alphas, infos

    @torch.no_grad()
    def render(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
        time_step: int = 0,
        is_training: bool = False,
    ) -> Float[np.ndarray, "H W 3"]:
        """Render function for NerfView."""

        image_width, image_height = img_wh
        cam_2_world = torch.tensor(camera_state.c2w).unsqueeze(0).float().cuda()
        intrinsics = torch.tensor(camera_state.get_K(img_wh)).unsqueeze(0).float().cuda()
        se3 = UnbatchedSE3Transform(
            rotation=DEFAULT_SE3_ROTATION.unsqueeze(0).cuda(),
            translation=DEFAULT_SE3_TRANSLATION.unsqueeze(0).cuda(),
        )
        image, _, _ = self.forward(
            intrinsics=intrinsics,
            world_2_cam=None,
            cam_2_world=cam_2_world,
            image_height=image_height,
            image_width=image_width,
            se3_transform=se3,
        )
        return image[0].detach().cpu().numpy()

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
        rendered_alphas: Float[torch.Tensor, "cam H W"],
        target_images: Float[torch.Tensor, "cam H W 3"],
        target_alphas: Float[torch.Tensor, "cam H W"],
        infos: dict,
    ) -> dict[str, Float[torch.Tensor, '']]:
        """ Computes the loss. """
        # set up
        loss_dict = {}
        loss = 0.0
        mask = repeat(target_alphas, "cam H W -> cam H W f", f=3)
        background = infos['background']
        background = repeat(
            background, "f -> cam H W f", cam=mask.shape[0], H=mask.shape[1], W=mask.shape[2])
        target_image_foreground = target_images*mask + (1-mask) * background
        rasterized_images = infos['rasterized_images']
        # l1 loss for rasterized images
        if self.gaussian_splatting_settings.l1_image_loss is not None:
            l1_loss = nn.functional.l1_loss(rasterized_images, target_image_foreground)
            loss_dict["l1_loss"] = l1_loss
            loss = loss + self.gaussian_splatting_settings.l1_image_loss * l1_loss
        # ssim loss for rasterized images
        if self.gaussian_splatting_settings.ssim_image_loss is not None:
            pred = rearrange(rasterized_images, "cam H W c -> cam c H W")
            tgt = rearrange(target_image_foreground, "cam H W c -> cam c H W")
            ssim_loss = self.ssim(pred, tgt)
            loss_dict["ssim_loss"] = ssim_loss
            loss = loss + self.gaussian_splatting_settings.ssim_image_loss * ssim_loss
        # ssim loss for denoised images
        if self.gaussian_splatting_settings.ssim_denoised_image_loss is not None:
            pred = rearrange(rendered_images, "cam H W c -> cam c H W")
            tgt = rearrange(target_images, "cam H W c -> cam c H W")
            ssim_loss = self.ssim(pred, tgt)
            loss_dict["ssim_denoised_loss"] = ssim_loss
            loss = loss + self.gaussian_splatting_settings.ssim_denoised_image_loss * ssim_loss
        # lpips for rendered images
        if self.gaussian_splatting_settings.lpips_image_loss is not None:
            pred = (rasterized_images-0.5) * 2
            target = (target_image_foreground-0.5) * 2
            pred = pred.clamp(-0.99, 0.99)
            target = target.clamp(-0.99, 0.99)
            lpips_loss = self.lpips(pred, target)
            loss_dict["lpips_loss"] = lpips_loss
            loss = loss + self.gaussian_splatting_settings.lpips_image_loss * lpips_loss
        # anisotropy loss
        if self.gaussian_splatting_settings.anisotropy_loss is not None:
            scales = self.splats["scales"]
            max_ratio = self.gaussian_splatting_settings.anisotropy_max_ratio
            log_scale_ratio = scales.max(dim=1).values - scales.min(dim=1).values
            scale_ratio = torch.exp(log_scale_ratio)
            anisotropy_loss = (
                torch.where(scale_ratio > max_ratio, scale_ratio, max_ratio).mean() - max_ratio)
            loss_dict["anisotropy_loss"] = anisotropy_loss
            loss = loss + self.gaussian_splatting_settings.anisotropy_loss * anisotropy_loss
        # max scale loss
        if self.gaussian_splatting_settings.max_scale_loss is not None:
            max_log_scales = torch.max(scales, dim=1).values
            max_log_scales_found = torch.exp(max_log_scales)
            max_scale_loss = torch.relu(max_log_scales_found
                                        - self.gaussian_splatting_settings.max_scale).mean()
            loss_dict["max_scale_loss"] = max_scale_loss
            loss = loss + self.gaussian_splatting_settings.max_scale_loss * max_scale_loss
        # local rigidity loss
        if self.gaussian_splatting_settings.local_rigidity_loss is not None:
            raise NotImplementedError('local rigidity loss is not implemented yet')
        # background loss
        if self.gaussian_splatting_settings.background_loss is not None:
            background_loss = nn.functional.l1_loss(rendered_alphas, target_alphas.unsqueeze(-1))
            loss_dict["background_loss"] = background_loss
            loss = loss + self.gaussian_splatting_settings.background_loss * background_loss

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
        rendered_images, rendered_alphas, infos = self.forward(
            intrinsics=batch.intrinsics,
            world_2_cam=batch.world_2_cam,
            cam_2_world=None,
            image_height=int(batch.image.shape[1]),
            image_width=int(batch.image.shape[2]),
            # color_correction=batch.color_correction, # TODO: fix color correction
            cur_sh_degree=self.get_cur_sh_degree(self.global_step),
            se3_transform=batch.se3_transform,
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
            target_alphas=batch.mask,
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
            self.writer.add_image("train/render", canvas, self.global_step, dataformats="HWC")
            # raw render
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                canvas = (
                    torch.concatenate(
                        [batch.image[0].clamp(0, 1), infos['rasterized_images'][0].clamp(0, 1)],
                        dim=1,
                    ).detach().cpu().numpy())
                canvas = (canvas * 255).astype(np.uint8)
                self.writer.add_image(
                    "train/raw_render", canvas, self.global_step, dataformats="HWC")

        # Iteration time logging
        time_elapsed = time.time() - t
        its = time_elapsed / batch.image.shape[0]
        self.log('train/its', its, on_step=True, on_epoch=False, prog_bar=True)

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
    def val_step(self, batch: SingleFrameData, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        pass

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
#                                 Train  Loop                                          #
# ==================================================================================== #


def train(config_path: str) -> None:
    """ Train loop for the static single frame model. """

    # set up model
    config = load_config(config_path)
    model = GaussianSplattingSingleFrame(
        config.gaussian_splatting_settings,
        config.learning_rates,
        enable_viewer=config.enable_viewer,
    )
    if config.compile:
        model._compile()

    # sanity check
    params = model.splats
    optimizers, schedulers = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get dataloaders
    train_set = SingleSequenceDataset(
        sequence=config.sequence,
        start_idx=config.frame,
        end_idx=config.frame + 1,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=config.num_train_workers,
        persistent_workers=True)

    # start viewer
    model.cuda()
    model.start_viewer(mode="training")

    # train
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.gaussian_splatting_settings.train_iterations,
    )
    trainer.fit(model, train_loader)


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
    parser.add_argument("-v", "--viewer", action="store_true", help="Start the viewer.")
    args = parser.parse_args()
    if args.viewer:
        config = load_config(args.config)
        model = GaussianSplattingSingleFrame(
            config.gaussian_splatting_settings, config.learning_rates, enable_viewer=True)
        model.cuda()
        print("Starting viewer...")
        model.viewer()

    else:
        train(args.config)
        print("Starting training...")
