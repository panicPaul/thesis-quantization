""" Some settings where dataclasses make sense. """

from dataclasses import dataclass

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class GaussianSplattingSettings:
    """
    Shared settings for the Gaussian splatting experiments.

    Args:
        feature_dim (int): Dimension of the input features.
        sh_degree (int): Degree of the spherical harmonics basis.
    """

    # pre-processing settings
    use_view_dependent_color_mlp: bool = True

    # rasterization_settings
    sh_degree: int = 3
    rasterization_mode: str = "default"  # 2dgs vs 3dgs
    antialiased: bool = True
    radius_clip: float = 0.0
    dist_loss: bool = False  # only for 2D gs

    # post-processing settings
    background_r: float = 0.5  # hacky bullshit to shut up type checker
    background_g: float = 0.5
    background_b: float = 0.5
    screen_space_denoising_mode: str = "none"
    camera_color_correction: bool = False
    learnable_color_correction: bool = False

    # densification settings
    densification_mode: str = "default"
    refine_start_iteration: int = 500
    refine_stop_iteration: int | float = 0.5  # should be around 85-90% for mcmc
    cap_max: int = MISSING  # mcmc only

    # initialization settings
    initialization_mode: str = MISSING
    initialization_points: int = MISSING
    scene_scale: float = 0.2

    # losses
    l1_image_loss: float | None = MISSING
    ssim_image_loss: float | None = MISSING
    ssim_denoised_image_loss: float | None = MISSING
    lpips_image_loss: float | None = MISSING
    anisotropy_loss: float | None = MISSING
    max_scale_loss: float | None = MISSING
    local_rigidity_loss: float | None = MISSING
    background_loss: float | None = MISSING

    # loss kwargs
    lpips_network: str = 'vgg'
    anisotropy_max_ratio: float = 10.0
    max_scale: float = 0.05 * 0.2  # penalize when the scale is larger than this

    # train settings
    feature_dim: int = 32
    camera_batch_size: int = 1
    sh_increase_interval: int = 1_000
    train_iterations: int = 30_000
    log_images_interval: int = 500


def load_config(path: str) -> DictConfig:
    """Load the configuration file."""
    config = OmegaConf.load(path)
    gaussian_splatting_settings = GaussianSplattingSettings(**config.gaussian_splatting_settings)
    config.gaussian_splatting_settings = gaussian_splatting_settings
    return config
