""" Some settings where dataclasses make sense. """

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class GaussianSplattingSettings:
    """
    Shared settings for the Gaussian splatting experiments.

    Args:
        feature_dim (int): Dimension of the input features.
        sh_degree (int): Degree of the spherical harmonics basis.
    """

    feature_dim: int = 32
    sh_degree: int = 3
    initialization_mode: str = "random"
    densification_mode: str = "default"
    use_view_dependent_color_mlp: bool = True
    rasterization_mode: str = "default"

    # training
    camera_batch_size: int = 1
    sh_increase_interval: int = 1_000


def load_config(path: str) -> DictConfig:
    """Load the configuration file."""
    config = OmegaConf.load(path)
    gaussian_splatting_settings = GaussianSplattingSettings(
        **config.gaussian_splatting_settings
    )
    config.gaussian_splatting_settings = gaussian_splatting_settings
    return config
