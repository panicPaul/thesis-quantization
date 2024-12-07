""" Loss computation for Gaussian splatting. """

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch.nn import ParameterDict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thesis.config import DynamicGaussianSplattingSettings
from thesis.utils import assign_segmentation_class


class LossComputer(nn.Module):
    """
    Loss computation for Gaussian splatting. Should not be part of the lightning module to avoid
    checkpointing the loss networks.
    """

    def __init__(self, gaussian_splatting_settings: DynamicGaussianSplattingSettings) -> None:
        """
        Args:
            gaussian_splatting_settings: Settings for Gaussian splatting.
        """
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=gaussian_splatting_settings.lpips_network, normalize=True)
        self.gaussian_splatting_settings = gaussian_splatting_settings

    def forward(
        self,
        splats: ParameterDict,
        rendered_images: Float[torch.Tensor, "cam H W 3"],
        rendered_alphas: Float[torch.Tensor, "cam H W 1"],
        target_images: Float[torch.Tensor, "cam H W 3"],
        target_alphas: Float[torch.Tensor, "cam H W"],
        target_segmentation_mask: Float[torch.Tensor, "cam H W 3"],
        background: Float[torch.Tensor, "3"],
        infos: dict,
        cur_step: int,
    ) -> dict[str, Float[torch.Tensor, '']]:
        """
        Computes the loss.

        Args:
            rendered_images: Rendered images of shape (cam, H, W, 3).
            rendered_alphas: Rendered alphas of shape (cam, H, W, 1).
            target_images: Target images of shape (cam, H, W, 3).
            target_alphas: Target alphas of shape (cam, H, W).
            target_segmentation_mask: Target segmentation mask of shape (cam, H, W, 3).
            infos: Dictionary containing additional information.

        Returns:
            dict: Dictionary containing the loss values.
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
        if self.gaussian_splatting_settings.lpips_image_loss is not None and cur_step > \
                self.gaussian_splatting_settings.lpips_start_iteration:
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
        if self.gaussian_splatting_settings.anisotropy_loss is not None and \
           self.gaussian_splatting_settings.rasterization_mode == '3dgs':  # disable for 2dgs

            @torch.compiler.disable
            def f():
                scales = splats['scales']
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
                scales = splats['scales']
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
            tgt = rearrange(target_images * eyebrows_mask, "cam H W f -> cam f H W")
            eyebrows_ssim_loss = 1 - self.ssim(pred, tgt).mean()
            loss_dict["eyebrows_ssim"] = eyebrows_ssim_loss
            loss = loss + eyebrows_ssim_loss * self.gaussian_splatting_settings.eyebrows_ssim_loss
        # PSNR (metric only)
        psnr = self.psnr.forward(raw_rendered_images, target_images)
        loss_dict["psnr"] = psnr

        # Markov Chain Monte Carlos Losses
        if self.gaussian_splatting_settings.mcmc_opacity_regularization is not None and self.gaussian_splatting_settings.densification_mode == 'monte_carlo_markov_chain':
            opacity = nn.functional.sigmoid(splats['opacities']).abs()
            opacity_loss = opacity.mean()
            loss_dict["opacity_loss"] = opacity_loss
            loss = loss + opacity_loss * self.gaussian_splatting_settings.mcmc_opacity_regularization
        if self.gaussian_splatting_settings.mcmc_scale_regularization is not None and self.gaussian_splatting_settings.densification_mode == 'monte_carlo_markov_chain':
            scales = torch.exp(splats['scales']).abs()
            scale_loss = scales.mean()
            loss_dict["scale_loss"] = scale_loss
            loss = loss + scale_loss * self.gaussian_splatting_settings.mcmc_scale_regularization

        loss_dict["loss"] = loss
        return loss_dict
