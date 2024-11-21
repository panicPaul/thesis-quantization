""" Dynamic Gaussian Splatting Model"""

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
import torch.nn as nn
import viser
from gsplat import DefaultStrategy, MCMCStrategy, rasterization, rasterization_2dgs
from jaxtyping import Float, Int, UInt8
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis.config import DynamicGaussianSplattingSettings, load_config
from thesis.constants import (
    CANONICAL_FLAME_PARAMS,
    DEFAULT_SE3_ROTATION,
    DEFAULT_SE3_TRANSLATION,
    TEST_CAMS,
    TEST_SEQUENCES,
    TRAIN_CAMS,
    TRAIN_SEQUENCES,
)
from thesis.data_management import MultiSequenceDataset, SequenceManager
from thesis.data_management.data_classes import (
    SingleFrameData,
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)
from thesis.flame import FlameHeadWithInnerMouth
from thesis.gaussian_splatting.initialize_splats import (
    flame_initialization,
    point_cloud_initialization,
    pre_trained_initialization,
    random_initialization,
)
from thesis.gaussian_splatting.loss_computation import LossComputer
from thesis.gaussian_splatting.post_processing import PostProcessor
from thesis.gaussian_splatting.pre_processing import RiggedPreProcessor


class DynamicGaussianSplatting(pl.LightningModule):
    """ Flame and audio driven dynamic Gaussian splatting model. """

    # ================================================================================ #
    #                                Initialization                                    #
    # ================================================================================ #

    def __init__(
        self,
        name: str,
        gaussian_splatting_settings: DynamicGaussianSplattingSettings | DictConfig,
        learning_rates: DictConfig,
        enable_viewer: bool = True,
        viewer_sequences: list[int] = [3, 100],
        ckpt_path: str | None = None,
    ) -> None:
        """
        Initializes the Gaussian splatting model.

        Args:
            gaussian_splatting_settings (GaussianSplattingSettings | DictConfig): Gaussian
                splatting settings.
            learning_rates (DictConfig): Learning rates.
            enable_viewer (bool): Whether to enable the viewer.
            viewer_sequences (list[int]): Viewer sequences, will be concatenated.
            checkpoint_path (str | None): Path to the checkpoint. Needs to be provided if
                loading a pre-trained model, since torch will throw size mismatch errors
                otherwise.
        """
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        self.automatic_optimization = False

        # Save the settings
        self.learning_rates = learning_rates
        self.gaussian_splatting_settings = gaussian_splatting_settings
        if isinstance(gaussian_splatting_settings, DictConfig):
            gaussian_splatting_settings = DynamicGaussianSplattingSettings(
                **gaussian_splatting_settings)

        # Process settings
        self.max_sh_degree = gaussian_splatting_settings.sh_degree
        self.default_background = torch.tensor([
            gaussian_splatting_settings.background_r,
            gaussian_splatting_settings.background_g,
            gaussian_splatting_settings.background_b,
        ]).cuda()
        self.enable_viewer = enable_viewer

        # Load canonical flame params
        canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS)
        self.register_buffer("canonical_flame_shape", canonical_flame_params.shape)
        self.register_buffer("canonical_flame_expr", canonical_flame_params.expr)
        self.register_buffer("canonical_flame_neck", canonical_flame_params.neck)
        self.register_buffer("canonical_flame_jaw", canonical_flame_params.jaw)
        self.register_buffer("canonical_flame_eye", canonical_flame_params.eye)
        self.register_buffer("canonical_flame_scale", canonical_flame_params.scale)
        self.flame_head = FlameHeadWithInnerMouth()

        # Load viewer sequences
        viewer_flame_params_list = []
        viewer_audio_features_list = []
        for seq in viewer_sequences:
            sm = SequenceManager(seq)
            viewer_flame_params_list.append(sm.flame_params[:])
            viewer_audio_features_list.append(sm.audio_features[:])
        self.viewer_flame_params_shape = torch.concatenate(
            [fp.shape for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_flame_params_expr = torch.concatenate(
            [fp.expr for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_flame_params_neck = torch.concatenate(
            [fp.neck for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_flame_params_jaw = torch.concatenate(
            [fp.jaw for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_flame_params_eye = torch.concatenate(
            [fp.eye for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_flame_params_scale = torch.concatenate(
            [fp.scale for fp in viewer_flame_params_list], dim=0).cuda()
        self.viewer_audio_features = torch.concatenate(viewer_audio_features_list, dim=0).cuda()

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
                        flame_params=canonical_flame_params,
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

        # Initialize pre and post-processors
        self.pre_processor = RiggedPreProcessor(gaussian_splatting_settings, learning_rates)
        self.post_processor = PostProcessor(gaussian_splatting_settings, learning_rates)

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

        # initialize loss computer
        self.loss_computer = LossComputer(gaussian_splatting_settings)

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

    @property
    def step(self) -> int:
        """Returns the current step."""
        return self.global_step // self.n_optimizers

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            A tuple containing the splat optimizers and other optimizers.
        """

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

        # ---> other optimizers
        pre_processor_optimizer = self.pre_processor.setup_optimizer()
        post_processor_optimizer = self.post_processor.setup_optimizer()
        if post_processor_optimizer is not None:
            other_optimizers = [pre_processor_optimizer, post_processor_optimizer]
        else:
            other_optimizers = [pre_processor_optimizer]
        optimizer_list = list(splat_optimizers.values()) + other_optimizers
        self.n_optimizers = len(optimizer_list)

        # ---> schedulers
        schedulers = {}
        schedulers["means"] = torch.optim.lr_scheduler.ExponentialLR(
            splat_optimizers["means"],
            gamma=0.01**(1.0 /
                         (self.gaussian_splatting_settings.train_iterations * self.n_optimizers)))
        self.splat_optimizer_keys = list(splat_optimizers.keys())
        scheduler_list = list(schedulers.values())

        return optimizer_list, scheduler_list

    # ================================================================================ #
    #                                 Rasterization                                    #
    # ================================================================================ #

    def forward(
        self,
        se3_transform: UnbatchedSE3Transform,
        rigging_params: Float[torch.Tensor, "n_vertices 3"],
        intrinsics: Float[torch.Tensor, "cam 3 3"],
        image_width: int,
        image_height: int,
        cam_2_world: Float[torch.Tensor, "cam 4 4"] | None = None,
        world_2_cam: Float[torch.Tensor, "cam 4 4"] | None = None,
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
        cur_sh_degree: int | None = None,
        flame_params: UnbatchedFlameParams | None = None,
        audio_features: Float[torch.Tensor, "window_size 1024"] | None = None,
        background: Float[torch.Tensor, "3"] | None = None,
        _means_override: Float[torch.Tensor, "n_splats 3"] | None = None,
    ) -> tuple[
            Float[torch.Tensor, "cam H W 3"],
            Float[torch.Tensor, "cam H W 1"],
            Float[torch.Tensor, "cam H W 1"],
            dict,
    ]:
        """
        Args:
            se3_transform (UnbatchedSE3Transform): SE3 transform. Shape: `(cam, 3, 3)`.
            rigging_params (torch.Tensor): Rigging parameters. Shape: `(n_vertices, 3)`.
            intrinsics (torch.Tensor): Intrinsics. Shape: `(cam, 3, 3)`.
            image_width (int): Image width.
            image_height (int): Image height.
            cam_2_world (torch.Tensor): Camera to world matrix. Shape: `(cam, 4, 4)`. Either
                `cam_2_world` or `world_2_cam` must be provided.
            world_2_cam (torch.Tensor): World to camera matrix. Shape: `(cam, 4, 4)`. Either
                `cam_2_world` or `world_2_cam` must be provided.
            camera_indices (torch.Tensor): Camera indices. Shape: `(cam,)`.
            cur_sh_degree (int): Current SH degree.
            flame_params (UnbatchedFlameParams): Flame parameters.
            audio_features (torch.Tensor): Audio features. Shape: `(window_size, 1024)`.
            background (torch.Tensor): Background.
            _means_override (torch.Tensor): Means override. This is used when we want to smooth
                the trajectories of the splats.

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): RGB images. Shape: `(cam, H, W, 3)`.
                - (*torch.Tensor*): Alphas. Shape: `(cam, H, W, 1)`.
                - (*torch.Tensor*): Depth maps. Shape: `(cam, H, W, 1)`.
                - (*dict*): Infos.
        """

        # set up transformation matrices
        assert (cam_2_world is None) != (world_2_cam is None), \
            "Either `cam_2_world` or `world_2_cam` must be provided."
        if cam_2_world is None:
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

        # background
        if background is None:
            background = self.default_background

        # pre-processing
        infos = {}
        means, quats, scales, opacities, colors, infos = self.pre_processor.forward(
            splats=self.splats,
            se3_transform=se3_transform,
            rigging_params=rigging_params,
            cam_2_world=cam_2_world,
            world_2_cam=world_2_cam,
            camera_indices=camera_indices,
            cur_sh_degree=cur_sh_degree,
            flame_params=flame_params,
            audio_features=audio_features,
            infos=infos,
        )

        if _means_override is not None:  # used for trajectory smoothing
            means = _means_override.cuda()

        # rasterization
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
            sh_degree=cur_sh_degree
            if not self.gaussian_splatting_settings.use_view_dependent_color_mlp else None,
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

        # post-processing
        images, alphas, infos = self.post_processor.forward(
            render_images=images,
            render_alphas=alphas,
            background=background,
            camera_indices=camera_indices,
            infos=infos,
        )

        return images, alphas, depth_maps, infos

    # ================================================================================ #
    #                                 Viewer stuff                                     #
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
        num_frames = self.viewer_audio_features.shape[
            0] - self.gaussian_splatting_settings.prior_window_size + 1
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.render,
            mode=mode,
            num_frames=num_frames,
        )

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

        # get audio and flame params
        window_size = self.gaussian_splatting_settings.prior_window_size
        audio_features = self.viewer_audio_features[time_step:time_step + window_size]
        flame_params = UnbatchedFlameParams(
            shape=self.viewer_flame_params_shape[time_step:time_step + window_size],
            expr=self.viewer_flame_params_expr[time_step:time_step + window_size],
            neck=self.viewer_flame_params_neck[time_step:time_step + window_size],
            jaw=self.viewer_flame_params_jaw[time_step:time_step + window_size],
            eye=self.viewer_flame_params_eye[time_step:time_step + window_size],
            scale=self.viewer_flame_params_scale[time_step:time_step + window_size],
        )
        rigging_params = self.flame_head.forward(flame_params)  # (window_size, n_vertices, 3)
        rigging_params = rigging_params[rigging_params.shape[0] // 2]  # (n_vertices, 3)

        image, _, depth, _ = self.forward(
            intrinsics=intrinsics,
            world_2_cam=None,
            cam_2_world=cam_2_world,
            image_height=image_height,
            image_width=image_width,
            se3_transform=se3,
            rigging_params=rigging_params,
            flame_params=flame_params,
            audio_features=audio_features,
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
    #                                 Training step                                    #
    # ================================================================================ #

    def training_step(
        self,
        batch: tuple[SingleFrameData, UnbatchedFlameParams, Float[torch.Tensor, 'window 1024']],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """
        Args:
            batch (tuple): A tuple containing
                - (*SingleFrameData*): Single frame data.
                - (*UnbatchedFlameParams*): Flame parameters.
                - (*torch.Tensor*): Audio features.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """

        # Pause the viewer if needed
        if self.enable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()
        t = time.time()

        # Get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        for opt in optimizers:
            opt.zero_grad()
        splat_optimizers = {k: optimizers[i] for i, k in enumerate(self.splat_optimizer_keys)}

        # Get Rigging params
        frame, flame_params, audio_features = batch
        rigging_params = self.flame_head.forward(flame_params)  # (window_size, n_vertices, 3)]
        rigging_params = rigging_params[rigging_params.shape[0] // 2]  # (n_vertices, 3)

        # Forward pass
        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            se3_transform=frame.se3_transform,
            rigging_params=rigging_params,
            world_2_cam=frame.world_2_cam,
            intrinsics=frame.intrinsics,
            image_width=int(frame.image.shape[2]),
            image_height=int(frame.image.shape[1]),
            camera_indices=frame.camera_indices,
            cur_sh_degree=self.max_sh_degree,
            flame_params=flame_params,
            audio_features=audio_features,
            background=self.default_background,
        )

        # Pre-backward densification
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=splat_optimizers,
            state=self.strategy_state,
            step=self.step,
            info=infos,
        )

        # Compute loss
        loss_dict = self.loss_computer.forward(
            splats=self.splats,
            rendered_images=rendered_images,
            rendered_alphas=rendered_alphas,
            target_images=frame.image,
            target_alphas=frame.alpha_map,
            target_segmentation_mask=frame.segmentation_mask,
            background=self.default_background,
            infos=infos,
            cur_step=self.step,
        )
        loss = loss_dict["loss"]
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))
        if 'per_gaussian_movement' in infos:
            self.log(
                'per_gaussian_movement',
                infos['per_gaussian_movement'],
                on_step=True,
                on_epoch=False)
        self.log('num_gaussians', self.splats['means'].shape[0], on_step=True, on_epoch=False)

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
                    step=self.step,
                    info=infos,
                )
            case "monte_carlo_markov_chain":
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=splat_optimizers,
                    state=self.strategy_state,
                    step=self.step,
                    info=infos,
                    lr=schedulers.get_last_lr()[0],
                )
            case _:
                raise ValueError("Unknown densification mode: "
                                 f"{self.gaussian_splatting_settings.densification_mode}")

        # Log images
        if self.step % self.gaussian_splatting_settings.log_images_interval == 0:
            # denoised images
            canvas = (
                torch.concatenate(
                    [frame.image[0].clamp(0, 1), rendered_images[0].clamp(0, 1)],
                    dim=1,
                ).detach().cpu().numpy())
            canvas = (canvas * 255).astype(np.uint8)
            writer = self.logger.experiment
            writer.add_image("train/denoised_render", canvas, self.step, dataformats="HWC")
            # raw render
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                canvas = (
                    torch.concatenate(
                        [frame.image[0].clamp(0, 1), infos['raw_rendered_images'][0].clamp(0, 1)],
                        dim=1,
                    ).detach().cpu().numpy())
                canvas = (canvas * 255).astype(np.uint8)
                writer.add_image("train/raw_render", canvas, self.step, dataformats="HWC")

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

    # ================================================================================ #
    #                                 Validation step                                  #
    # ================================================================================ #

    def validation_step(
        self,
        batch: tuple[SingleFrameData, UnbatchedFlameParams, Float[torch.Tensor, 'window 1024']],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """
        Args:
            batch (tuple): A tuple containing
                - (*SingleFrameData*): Single frame data.
                - (*UnbatchedFlameParams*): Flame parameters.
                - (*torch.Tensor*): Audio features.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        # Pause the viewer if needed
        if self.enable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        # Get Rigging params
        frame, flame_params, audio_features = batch
        rigging_params = self.flame_head.forward(flame_params)  # (window_size, n_vertices, 3)]
        rigging_params = rigging_params[rigging_params.shape[0] // 2]  # (n_vertices, 3)

        # Forward pass
        rendered_images, rendered_alphas, rendered_depth, infos = self.forward(
            se3_transform=frame.se3_transform,
            rigging_params=rigging_params,
            intrinsics=frame.intrinsics,
            image_width=int(frame.image.shape[2]),
            image_height=int(frame.image.shape[1]),
            camera_indices=frame.camera_indices,
            cur_sh_degree=self.max_sh_degree,
            flame_params=flame_params,
            audio_features=audio_features,
            background=self.default_background,
            world_2_cam=frame.world_2_cam,
        )

        # Compute loss
        loss_dict = self.loss_computer.forward(
            splats=self.splats,
            rendered_images=rendered_images,
            rendered_alphas=rendered_alphas,
            target_images=frame.image,
            target_alphas=frame.alpha_map,
            target_segmentation_mask=frame.segmentation_mask,
            background=self.default_background,
            infos=infos,
            cur_step=self.step,
        )
        loss = loss_dict["loss"]
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))

        # Log images
        if batch_idx % 20 == 0:
            # Log denoised images
            canvas = (
                torch.concatenate(
                    [frame.image[0].clamp(0, 1), rendered_images[0].clamp(0, 1)],
                    dim=1,
                ).detach().cpu().numpy())
            canvas = (canvas * 255).astype(np.uint8)
            writer = self.logger.experiment
            writer.add_image("val/denoised_render", canvas, self.step, dataformats="HWC")
            # Log raw renders
            if self.gaussian_splatting_settings.screen_space_denoising_mode != "none":
                canvas = (
                    torch.concatenate(
                        [batch.image[0].clamp(0, 1), infos['raw_rendered_images'][0].clamp(0, 1)],
                        dim=1,
                    ).detach().cpu().numpy())
                canvas = (canvas * 255).astype(np.uint8)
                writer.add_image("val/raw_render", canvas, self.step, dataformats="HWC")

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
    #                                  Video rendering                                 #
    # ================================================================================ #

    @torch.no_grad()
    def compute_3d_trajectories(
        self,
        se_transforms: UnbatchedSE3Transform,
        rigging_params: Float[torch.Tensor, 'time n_vertices 3'],
        world_2_cam: Float[torch.Tensor, 'time 4 4'] | None = None,
        cam_2_world: Float[torch.Tensor, 'time 4 4'] | None = None,
        flame_params: UnbatchedFlameParams | None = None,
        audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
        background: Float[torch.Tensor, '3'] | None = None,
    ) -> Float[torch.Tensor, 'time n_splats 3']:
        """
        Computes the 3D trajectories for each splat.

        Args:
            se_transforms (UnbatchedSE3Transform): The SE3 transforms.
            rigging_params (torch.Tensor): The rigging parameters, shape: `(time, n_vertices, 3)`.
            world_2_cam (torch.Tensor): The world to camera matrices, shape: `(time, 4, 4)`.
            cam_2_world (torch.Tensor): The camera to world matrices, shape: `(time, 4, 4)`.
            flame_params (UnbatchedFlameParams): The flame parameters. Should have the same length
                as the se_transforms + window_size - 1.
            audio_features (torch.Tensor): The audio features, shape: `(time , 1024)`. Note that
                the audio features should have the same length as the
                se_transforms + window_size - 1.
            background (torch.Tensor): The background.
        """

        n_splats = self.splats['means'].shape[0]
        window_size = self.gaussian_splatting_settings.prior_window_size
        n_frames = len(se_transforms.rotation)
        trajectories = torch.zeros(n_frames - window_size + 1, n_splats, 3).cuda()

        for t in tqdm(
                range(window_size // 2, n_frames - window_size//2), desc="Computing trajectories"):

            # get current parameters
            if world_2_cam is not None:
                cur_world_2_cam = world_2_cam[t:t + 1]
                cur_cam_2_world = None
            else:
                cur_world_2_cam = None
                cur_cam_2_world = cam_2_world[t:t + 1]
            cur_se_transform = UnbatchedSE3Transform(
                rotation=se_transforms.rotation[t:t + 1],
                translation=se_transforms.translation[t:t + 1],
            )
            rigging_params_t = rigging_params[t]

            if flame_params is not None:
                flame_params_t = UnbatchedFlameParams(
                    shape=flame_params.shape[t - window_size//2:t + window_size//2 + 1],
                    expr=flame_params.expr[t - window_size//2:t + window_size//2 + 1],
                    neck=flame_params.neck[t - window_size//2:t + window_size//2 + 1],
                    jaw=flame_params.jaw[t - window_size//2:t + window_size//2 + 1],
                    eye=flame_params.eye[t - window_size//2:t + window_size//2 + 1],
                    scale=flame_params.scale[t - window_size//2:t + window_size//2 + 1],
                )
            else:
                flame_params_t = None

            if audio_features is not None:
                audio_features_t = audio_features[t - window_size//2:t + window_size//2 + 1]
            else:
                audio_features_t = None

            # set up transformation matrices
            assert (cur_cam_2_world is None) != (cur_world_2_cam is None), \
                "Either `cur_cam_2_world` or `cur_world_2_cam` must be provided."
            if cur_cam_2_world is None:
                cur_cam_2_world = torch.zeros_like(cur_world_2_cam)
                cur_cam_2_world[..., :3, :3] = cur_world_2_cam[..., :3, :3].transpose(-2, -1)
                cur_cam_2_world[..., :3, 3] = -torch.bmm(
                    cur_world_2_cam[..., :3, :3].transpose(-2, -1),
                    cur_world_2_cam[..., :3, 3].unsqueeze(-1),
                ).squeeze(-1)
                cur_cam_2_world[..., 3, 3] = 1
            else:
                cur_world_2_cam = torch.zeros_like(cur_cam_2_world)
                cur_world_2_cam[..., :3, :3] = cur_cam_2_world[..., :3, :3].transpose(-2, -1)
                cur_world_2_cam[..., :3, 3] = -torch.bmm(
                    cur_cam_2_world[..., :3, :3].transpose(-2, -1),
                    cur_cam_2_world[..., :3, 3].unsqueeze(-1),
                ).squeeze(-1)
                cur_world_2_cam[..., 3, 3] = 1

            # background
            if background is None:
                background = self.default_background

            # pre-processing
            infos = {}
            means, _, _, _, _, _ = self.pre_processor.forward(
                splats=self.splats,
                se3_transform=cur_se_transform,
                rigging_params=rigging_params_t,
                cam_2_world=cur_cam_2_world,
                world_2_cam=cur_world_2_cam,
                camera_indices=None,
                cur_sh_degree=None,
                flame_params=flame_params_t,
                audio_features=audio_features_t,
                infos=infos,
            )
            trajectories[t - window_size//2] = means

        return trajectories

    @torch.no_grad()
    def render_video(
        self,
        intrinsics: Float[torch.Tensor, 'time 3 3'],
        image_height: int,
        image_width: int,
        se_transforms: UnbatchedSE3Transform,
        rigging_params: Float[torch.Tensor, 'time n_vertices 3'],
        world_2_cam: Float[torch.Tensor, 'time 4 4'] | None = None,
        cam_2_world: Float[torch.Tensor, 'time 4 4'] | None = None,
        flame_params: UnbatchedFlameParams | None = None,
        audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
        background: Float[torch.Tensor, '3'] | None = None,
        trajectories: Float[torch.Tensor, 'time n_splats 3'] | None = None,
    ) -> UInt8[np.ndarray, 'time H W 3']:
        """
        Renders a video.

        Args:
            intrinsics (torch.Tensor): The intrinsics, shape: `(time, 3, 3)`.
            image_height (int): The image height.
            image_width (int): The image width.
            se_transforms (UnbatchedSE3Transform): The SE3 transforms.
            rigging_params (torch.Tensor): The rigging parameters, shape: `(time, n_vertices, 3)`.
            world_2_cam (torch.Tensor): The world to camera matrices, shape: `(time, 4, 4)`.
            cam_2_world (torch.Tensor): The camera to world matrices, shape: `(time, 4, 4)`.
            flame_params (UnbatchedFlameParams): The flame parameters. Should have the same length
                as the intrinsics + window_size - 1.
            audio_features (torch.Tensor): The audio features, shape: `(time , 1024)`. Note that
                the audio features should have the same length as the intrinsics + window_size - 1.
            background (torch.Tensor): The background.
            trajectories (torch.Tensor): Pre-computed trajectories. This is useful if we want to
                smooth the trajectories of the splats.

        Returns:
            np.ndarray: The video.
        """
        self.eval()
        self.cuda()
        video = []

        window_size = self.gaussian_splatting_settings.prior_window_size
        n_frames = len(intrinsics)
        assert se_transforms.rotation.shape[0] == n_frames
        assert rigging_params.shape[0] == n_frames
        if world_2_cam is not None and world_2_cam.ndim == 3:
            assert world_2_cam.shape[0] == n_frames
        if cam_2_world is not None and cam_2_world.ndim == 3:
            assert cam_2_world.shape[0] == n_frames
        if flame_params is not None:
            assert flame_params.expr.shape[0] == n_frames
        if audio_features is not None:
            assert audio_features.shape[0] == n_frames

        for t in tqdm(range(window_size // 2, n_frames - window_size//2), desc="Rendering video"):
            cur_intrinsics = intrinsics[t:t + 1]
            if world_2_cam is not None:
                cur_world_2_cam = world_2_cam[t:t + 1]
                cur_cam_2_world = None
            else:
                cur_world_2_cam = None
                cur_cam_2_world = cam_2_world[t:t + 1]
            cur_se_transform = UnbatchedSE3Transform(
                rotation=se_transforms.rotation[t:t + 1],
                translation=se_transforms.translation[t:t + 1],
            )
            rigging_params_t = rigging_params[t]

            if flame_params is not None:
                flame_params_t = UnbatchedFlameParams(
                    shape=flame_params.shape[t - window_size//2:t + window_size//2 + 1],
                    expr=flame_params.expr[t - window_size//2:t + window_size//2 + 1],
                    neck=flame_params.neck[t - window_size//2:t + window_size//2 + 1],
                    jaw=flame_params.jaw[t - window_size//2:t + window_size//2 + 1],
                    eye=flame_params.eye[t - window_size//2:t + window_size//2 + 1],
                    scale=flame_params.scale[t - window_size//2:t + window_size//2 + 1],
                )
            else:
                flame_params_t = None

            if audio_features is not None:
                audio_features_t = audio_features[t - window_size//2:t + window_size//2 + 1]
            else:
                audio_features_t = None

            if trajectories is not None:
                try:
                    cur_mean_positions = trajectories[t - window_size//2]
                except IndexError:
                    print(f'dropping frame {t - window_size//2} due to trajectory mismatch')
                    cur_mean_positions = None
            else:
                cur_mean_positions = None

            rendered_images, _, _, _ = self.forward(
                se3_transform=cur_se_transform,
                rigging_params=rigging_params_t,
                intrinsics=cur_intrinsics,
                image_width=image_width,
                image_height=image_height,
                cam_2_world=cur_cam_2_world,
                world_2_cam=cur_world_2_cam,
                camera_indices=None,
                cur_sh_degree=None,
                flame_params=flame_params_t,
                audio_features=audio_features_t,
                background=background,
                _means_override=cur_mean_positions,
            )
            # video[t] = rendered_images[0]
            image = rendered_images[0] * 255
            image = image.cpu().numpy().astype(np.uint8)
            video.append(image)

        video = np.stack(video, axis=0)
        return video


# ============================================================================================== #
#                                         Training Loop                                          #
# ============================================================================================== #


def training_loop(config_path: str) -> None:
    """
    Training loop for the dynamic Gaussian splatting model.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path, mode='dynamic')
    model = DynamicGaussianSplatting(
        name=config.name,
        gaussian_splatting_settings=config.gaussian_splatting_settings,
        learning_rates=config.learning_rates,
        enable_viewer=config.enable_viewer,
    )

    torch.set_float32_matmul_precision('high')
    if config.compile:
        print("Compiling...", end="\t")
        model.compile()
        print("Done.")

    # sanity check
    params = model.splats
    optimizers, schedulers = model.configure_optimizers()
    splat_optimizers = {k: optimizers[i] for i, k in enumerate(model.splat_optimizer_keys)}
    model.strategy.check_sanity(params, splat_optimizers)

    # get dataloaders
    train_set = MultiSequenceDataset(
        sequences=list(config.train_sequences)
        if config.train_sequences is not None else TRAIN_SEQUENCES,
        cameras=TRAIN_CAMS,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
        window_size=config.gaussian_splatting_settings.prior_window_size,
        image_downsampling_factor=config.gaussian_splatting_settings.image_downsampling_factor,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=config.gaussian_splatting_settings.shuffle_train_data,
        num_workers=config.num_train_workers,
        persistent_workers=True)
    val_set = MultiSequenceDataset(
        sequences=list(config.val_sequences)
        if config.val_sequences is not None else TEST_SEQUENCES,
        cameras=TEST_CAMS,
        n_cameras_per_frame=config.gaussian_splatting_settings.camera_batch_size,
        window_size=config.gaussian_splatting_settings.prior_window_size,
        image_downsampling_factor=config.gaussian_splatting_settings.image_downsampling_factor,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        shuffle=True,
        num_workers=config.num_val_workers,
        persistent_workers=True,
    )

    # start the viewer
    model.cuda()
    if config.enable_viewer:
        model.start_viewer(mode="training")

    # train
    logger = TensorBoardLogger("tb_logs/dynamic_gaussian_splatting", name=config.name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=None,
        max_steps=config.gaussian_splatting_settings.train_iterations * model.n_optimizers,
        limit_val_batches=25,
        val_check_interval=500,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model, train_loader, val_loader)


# ============================================================================================== #
#                                 Viewer and Evaluation                                          #
# ============================================================================================== #


def start_viewer(
    config_path: str,
    sequences: list[int] | int | None = None,
) -> None:
    """
    Starts the viewer.

    Args:
        config_path (str): Path to the configuration file.
        sequence (int): Sequence overwrite, by default [3, 100] are used.
    """
    config = load_config(config_path, mode='dynamic')
    model = DynamicGaussianSplatting(
        name=config.name,
        gaussian_splatting_settings=config.gaussian_splatting_settings,
        learning_rates=config.learning_rates,
        enable_viewer=True,
    )
    # overwrite the viewer sequences
    if sequences is not None:
        if isinstance(sequences, int):
            sequences = [sequences]
        viewer_flame_params_list = []
        viewer_audio_features_list = []
        for seq in sequences:
            sm = SequenceManager(seq)
            viewer_flame_params_list.append(sm.flame_params[:])
            viewer_audio_features_list.append(sm.audio_features[:])
        model.viewer_flame_params_shape = torch.concatenate(
            [fp.shape for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_flame_params_expr = torch.concatenate(
            [fp.expr for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_flame_params_neck = torch.concatenate(
            [fp.neck for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_flame_params_jaw = torch.concatenate(
            [fp.jaw for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_flame_params_eye = torch.concatenate(
            [fp.eye for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_flame_params_scale = torch.concatenate(
            [fp.scale for fp in viewer_flame_params_list], dim=0).cuda()
        model.viewer_audio_features = torch.concatenate(viewer_audio_features_list, dim=0).cuda()

    model.cuda()
    model.start_viewer(port=config.viewer_port, mode="rendering")


# ============================================================================================== #
#                                 MAIN                                                           #
# ============================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting model on sequences.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/dynamic_gaussian_splatting.yml",
        help="Path to the configuration file.")
    parser.add_argument(
        "-v", "--visualize", type=str, help="Path to checkpoint file to visualize.")
    args = parser.parse_args()

    # Visualization mode
    if args.visualize:
        model = DynamicGaussianSplatting.load_from_checkpoint(
            args.visualize,
            ckpt_path=args.visualize,
            viewer_sequences=[3, 100],
        )
        model.cuda()
        print("Starting viewer...")
        model.start_viewer(mode='rendering')
        print("To exit, press Ctrl+C.")
        time.sleep(100000)

    # Train mode
    else:
        print('Starting training...')
        training_loop(args.config)
