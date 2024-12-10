""" Renders custom audio"""

from thesis.render_video import process_audio
import torch
from thesis.gaussian_splatting.dynamic_model import DynamicGaussianSplatting
from thesis.data_management import SequenceManager, UnbatchedFlameParams, UnbatchedSE3Transform
from jaxtyping import Float
import os
from typing import Literal
from thesis.evaluation_runner import save_compressed_array, save_as_video


# MARK: - Functions
def render_custom_audio(
    model: DynamicGaussianSplatting,
    audio_path: str,
    rigging_param_path: str,
    output_path: str,
    background: Float[torch.Tensor, '3'] = torch.ones(3),
    image_height: int = 1604,
    image_width: int = 1100,
    intrinsics: Float[torch.Tensor, 'time 3 3'] | None = None,
    world_2_cam: Float[torch.Tensor, 'time 4 4'] | None = None,
    compression: Literal['gzip', 'bz2', 'lzma'] = 'lzma',
    video_only: bool = True,
) -> None:

    rigging_params = torch.load(rigging_param_path).cuda()
    audio_features = process_audio(audio_path, n_frames=rigging_params.shape[0]).cuda()
    windowed_rigging_params = rigging_params
    n_frames = audio_features.shape[0]

    # Load everything from the sequence manager
    model.cuda()
    model.eval()
    sm = SequenceManager(10)  # longest sequence
    assert n_frames <= len(
        sm), f'Audio length ({n_frames}) exceeds maximum sequence length ({len(sm)})'

    se3_transforms = sm.se3_transforms[:n_frames]
    se3_transforms = UnbatchedSE3Transform(
        translation=se3_transforms.translation[:n_frames].cuda(),
        rotation=se3_transforms.rotation[:n_frames].cuda(),
    )

    # Get default intrinsics and world_2_cam
    if intrinsics is None:
        intrinsics = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                   [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                   [0.0000e+00, 0.0000e+00, 1.0000e+00]])
        intrinsics = intrinsics.repeat(audio_features.shape[0], 1, 1)
    intrinsics = intrinsics.cuda()
    if world_2_cam is None:
        world_2_cam = torch.tensor([[0.9058, -0.2462, -0.3448, -0.0859],
                                    [-0.0377, -0.8575, 0.5131, 0.1364],
                                    [-0.4220, -0.4518, -0.7860, 1.0642],
                                    [0.0000, 0.0000, 0.0000, 1.0000]])
        world_2_cam = world_2_cam.repeat(audio_features.shape[0], 1, 1)
    world_2_cam = world_2_cam.cuda()

    # Render the sequence
    video = model.render_video(
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
        se_transforms=se3_transforms,
        rigging_params=rigging_params,
        windowed_rigging_params=windowed_rigging_params,
        audio_features=audio_features,
        world_2_cam=world_2_cam,
        background=background.cuda(),
    )

    # Save the video
    if not video_only:
        npy_output = output_path.replace('.mp4', '.npy')
        print(f'Saving video to: {npy_output}')
        save_compressed_array(array=video, filepath=npy_output, compression=compression)
        print('Done!\n')

    # Render the video
    clip_by = (model.gaussian_splatting_settings.prior_window_size - 1) / 30
    save_as_video(
        video=video,
        output_path=output_path,
        audio_path=audio_path,
        clip_audio_by=clip_by,
    )


# MARK: - Main
if __name__ == '__main__':

    model_path = 'tb_logs/dynamic_gaussian_splatting/ablations_final/with_color_mlp_2dgs/version_0/checkpoints/epoch=2-step=240000.ckpt'

    # audio_path = '/home/schlack/CodeTalker/demo/wav/man.wav'
    audio_path = '/home/schlack/CodeTalker/quick_brown_fox.m4a'
    rigging_param_path = 'saved_vertex_preds/custom2.pt'
    output_path = 'custom3.mp4'
    model = DynamicGaussianSplatting.load_from_checkpoint(model_path, ckpt_path=model_path)

    render_custom_audio(
        model=model,
        audio_path=audio_path,
        rigging_param_path=rigging_param_path,
        output_path=output_path,
        video_only=True,
    )
