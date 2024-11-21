""" Renders a video."""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import Literal

import cv2
import librosa
import numpy as np
import pydub
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, UInt8
from scipy.signal import savgol_filter
from tqdm import tqdm

from thesis.audio_feature_processing.audio_cleaning import pydub_to_np
from thesis.audio_feature_processing.wav2vec import (
    get_processor_and_model,
    process_sequence_interpolation,
)
from thesis.code_talker.stage2_runner import Stage2Runner
from thesis.data_management import (
    SequenceManager,
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)
from thesis.flame import FlameHeadWithInnerMouth
from thesis.gaussian_splatting.dynamic_model import DynamicGaussianSplatting
from thesis.video_utils import add_audio


@torch.no_grad()
def process_audio(
        audio_path: str,
        fps: int = 30,  # ahhh it's not actually 30, just very close to it
) -> Float[torch.Tensor, 'time 1024']:
    """
    Loads the audio from disk and processes it to extract features.

    Args:
        audio_path: Path to the audio file.
        fps: Frames per second. Defaults to 30.
        device: Device to use. Defaults to 'cuda'.

    Returns:
        Float[torch.Tensor, 'time 1024']: Audio features.
    """

    # Get audio model
    print("Loading audio model...", end="")
    processor, wav2vec_model = get_processor_and_model()
    wav2vec_model = wav2vec_model.cuda()
    print("\t\tDone!")

    # Process audio features
    print("Processing audio features...", end="")
    file_format = audio_path.split(".")[-1]
    if file_format == "m4a":
        # for macbook recordings
        audio = pydub.AudioSegment.from_file(audio_path)
        audio, sr = pydub_to_np(audio)
    else:
        audio, sr = sf.read(audio_path)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
    sr = 16_000
    audio_length = len(audio) / sr  # in seconds
    n_frames = int(audio_length * fps)
    audio_features = process_sequence_interpolation(
        audio=audio,
        sampling_rate=sr,
        n_frames=n_frames,
        processor=processor,
        model=wav2vec_model,
        device='cuda',
    )
    print("\tDone!")

    return audio_features


@torch.no_grad()
def audio_to_flame(
    checkpoint_path: str,
    audio_features: Float[torch.Tensor, 'time 1024'],
) -> UnbatchedFlameParams:
    """
    Args:
        checkpoint_path: Path to the audio-to-flame model checkpoint.
        audio_features: Audio features.

    Returns:
        UnbatchedFlameParams: Flame parameters.
    """
    runner = Stage2Runner.load_from_checkpoint(checkpoint_path)
    runner.eval()
    runner.cuda()
    flame_params = runner.predict(audio_features)
    return flame_params


def load_flame_parameters(sequence: int) -> UnbatchedFlameParams:
    """
    Loads flame parameters from a file. This is done to compare the predicted flame parameters
    with the ground truth.

    Args:
        sequence: Sequence number.

    Returns:
        tuple: Tuple containing the expression and jaw parameters.
    """
    sm = SequenceManager(sequence)
    return sm.flame_params[:]


def _savgol_1d(data, window_length, poly_order):
    """Helper function to apply Savitzky-Golay filter to a single 1D array."""
    return savgol_filter(data, window_length, poly_order)


def smooth_trajectory(points, method='gaussian', **kwargs):
    """
    Smooth trajectories of multiple points over time.

    Parameters:
    points: torch.Tensor of shape (time, n_gaussians, 3)
        Tensor containing point coordinates over time
    method: str
        Smoothing method ('gaussian', 'moving_average', 'savgol', 'exponential')
    kwargs: dict
        Additional parameters for specific smoothing methods:
        - gaussian: kernel_size (int), sigma (float)
        - moving_average: kernel_size (int)
        - savgol: window_length (int), poly_order (int)
        - exponential: alpha (float)

    Returns:
    torch.Tensor of shape (time, n_gaussians, 3)
        Smoothed trajectories
    """
    device = points.device
    time_steps, n_gaussians, n_dims = points.shape

    match method:
        case 'gaussian':
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 1.0)

            # Ensure kernel_size is odd
            kernel_size = kernel_size + (kernel_size % 2 == 0)
            pad_size = kernel_size // 2

            # Reshape for batch processing
            # Combine n_gaussians and n_dims into batch dimension
            points_reshaped = points.permute(1, 2, 0).reshape(-1, 1, time_steps)

            # Apply padding
            padded = F.pad(points_reshaped, (pad_size, pad_size), mode='replicate')

            # Apply Gaussian smoothing
            smoothed = F.gaussian_blur1d(padded, kernel_size, sigma)

            # Reshape back to original dimensions
            return smoothed.view(n_gaussians, n_dims, time_steps).permute(2, 0, 1)

        case 'moving_average':
            kernel_size = kwargs.get('kernel_size', 5)

            # Ensure kernel_size is odd
            kernel_size = kernel_size + (kernel_size % 2 == 0)

            # Create averaging kernel
            kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
            pad_size = kernel_size // 2

            # Reshape for batch processing
            points_reshaped = points.permute(1, 2, 0).reshape(-1, 1, time_steps)

            # Apply padding
            padded = F.pad(points_reshaped, (pad_size, pad_size), mode='replicate')

            # Apply moving average
            smoothed = F.conv1d(padded, kernel)

            # Reshape back to original dimensions
            return smoothed.view(n_gaussians, n_dims, time_steps).permute(2, 0, 1)

        case 'savgol':
            window_length = kwargs.get('window_length', 11)
            poly_order = kwargs.get('poly_order', 3)
            parallel_method = kwargs.get('parallel_method', 'multiprocessing')
            # numpy is only marginal faster than single threaded
            # multiprocessing is significantly faster

            points_np = points.cpu().numpy()

            match parallel_method:
                case 'numpy':
                    # Reshape to 2D array where each row is a trajectory
                    trajectories = points_np.reshape(time_steps, -1)
                    # Apply savgol_filter to all trajectories at once using array_map
                    smoothed_trajectories = np.array([
                        savgol_filter(traj, window_length, poly_order)
                        for traj in tqdm(trajectories.T, desc='Applying Savgol filter')
                    ]).T
                    # Reshape back to original dimensions
                    smoothed_np = smoothed_trajectories.reshape(time_steps, n_gaussians, n_dims)

                case 'multiprocessing':
                    # Prepare data for parallel processing
                    trajectories = points_np.reshape(time_steps, -1).T
                    # Create partial function with fixed parameters
                    savgol_partial = partial(
                        _savgol_1d, window_length=window_length, poly_order=poly_order)
                    # Use multiprocessing pool to parallelize
                    with Pool() as pool:
                        smoothed_trajectories = np.array(
                            list(
                                tqdm(
                                    pool.imap(savgol_partial, trajectories),
                                    total=len(trajectories),
                                    desc='Applying Savgol filter'))).T
                    # Reshape back to original dimensions
                    smoothed_np = smoothed_trajectories.reshape(time_steps, n_gaussians, n_dims)

                case _:
                    raise ValueError("parallel_method must be either 'numpy' or 'multiprocessing'")

            return torch.from_numpy(smoothed_np).to(device)

        case 'exponential':
            alpha = kwargs.get('alpha', 0.3)
            smoothed = torch.zeros_like(points)
            smoothed[0] = points[0]

            # Apply exponential smoothing to each point's trajectory
            for t in range(1, time_steps):
                smoothed[t] = alpha * points[t] + (1-alpha) * smoothed[t - 1]

            return smoothed

        case _:
            raise ValueError(
                "Method must be one of: 'gaussian', 'moving_average', 'savgol', 'exponential'")


@torch.no_grad()
def render_video_dynamic_gaussian_splatting(
    checkpoint_path: str,
    # flame_params: UnbatchedFlameParams,
    rigging_params: Float[torch.Tensor, 'time n_vertices 3'],
    se_3_transforms: UnbatchedSE3Transform,
    flame_params: UnbatchedFlameParams | None = None,
    audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
    padding_mode: Literal['input', 'output', 'none'] = 'none',
    intrinsics: Float[torch.Tensor, '3 3'] = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                                           [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                                           [0.0000e+00, 0.0000e+00, 1.0000e+00]]),
    world_2_cam: Float[torch.Tensor, '4 4'] | Float[torch.Tensor, 'time 4 4'] = torch.tensor(
        [[0.9058, -0.2462, -0.3448, -0.0859], [-0.0377, -0.8575, 0.5131, 0.1364],
         [-0.4220, -0.4518, -0.7860, 1.0642], [0.0000, 0.0000, 0.0000, 1.0000]]),
    image_height: int = 1604,
    image_width: int = 1100,
    background_color: Float[torch.Tensor, '3'] = torch.tensor([0.66, 0.66, 0.66]),
    smoothing_mode: Literal['none', 'gaussian', 'moving_average', 'savgol',
                            'exponential'] = 'none',
) -> UInt8[np.ndarray, 't h w 3']:
    """
    Render dynamic gaussian splatting video.

    Args:
        checkpoint_path (str): Path to the Gaussian splatting model checkpoint.
        rigging_params (Float[torch.Tensor, 'time n_vertices 3']): Rigging parameters.
        se_3_transforms (UnbatchedSE3Transform): SE3 transforms.
        audio_features (Float[torch.Tensor, 'time 1024'] | None): Audio features.
        padding_mode (str): Padding mode. Defaults to 'output'.
        intrinsics (np.ndarray): Camera intrinsics.
        world_2_cam (np.ndarray): World to camera transformation. Defaults to the test view.
        image_height (int): Image height. Defaults to 1604.
        image_width (int): Image width. Defaults to 1100.
        background_color (torch.Tensor): Background color. Defaults to [0.5, 0.5, 0.5].
        use_se3 (bool): Whether to use SE3 transformations. Defaults to True.
        smoothing_mode (str): Smoothing mode. Defaults to 'none'.
    """
    model = DynamicGaussianSplatting.load_from_checkpoint(
        checkpoint_path, ckpt_path=checkpoint_path)
    model.eval()
    model.to('cuda')

    if padding_mode != 'none':
        raise NotImplementedError('Padding not yet implemented for dynamic gaussian splatting')

    # set up variables
    n_frames = rigging_params.shape[0]
    intrinsics = repeat(intrinsics, 'm n -> time m n', time=n_frames).cuda()
    if world_2_cam.shape == (4, 4):
        world_2_cam = repeat(world_2_cam, 'm n -> time m n', time=n_frames).cuda()
    else:
        world_2_cam = world_2_cam.cuda()
    se_3_transforms = UnbatchedSE3Transform(
        rotation=se_3_transforms.rotation.cuda(),
        translation=se_3_transforms.translation.cuda(),
    )
    rigging_params = rigging_params.cuda()
    background = background_color.cuda()
    audio_features = audio_features.cuda() if audio_features is not None else None
    flame_params = UnbatchedFlameParams(
        shape=flame_params.shape.cuda(),
        expr=flame_params.expr.cuda(),
        neck=flame_params.neck.cuda(),
        jaw=flame_params.jaw.cuda(),
        eye=flame_params.eye.cuda(),
        scale=flame_params.scale.cuda(),
    ) if flame_params is not None else None

    # Smooth trajectories
    if smoothing_mode != 'none':
        # TODO: start and finish do NOT seem to work yet. Not sure why.
        trajectories = model.compute_3d_trajectories(
            se_transforms=se_3_transforms,
            rigging_params=rigging_params,
            world_2_cam=world_2_cam,
            cam_2_world=None,
            flame_params=flame_params,
            audio_features=audio_features,
            background=background,
        ).detach()
        trajectories = smooth_trajectory(trajectories, method=smoothing_mode)
    else:
        trajectories = None

    # Render video
    video = model.render_video(
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
        se_transforms=se_3_transforms,
        rigging_params=rigging_params,
        world_2_cam=world_2_cam,
        flame_params=flame_params,
        audio_features=audio_features,
        background=background,
        trajectories=trajectories,
    )
    return video


def render_gt_video(
    sequence: int,
    camera: int = 8,
    output_dir='tmp/gt',
    trim_size: int = 0,
    quicktime_compatible: bool = False,
) -> None:
    """
    Renders a video from ground truth flame parameters.

    Args:
        sequence: Sequence number.
        camera: Camera number. Defaults to 8.
        output_dir: Output directory. Defaults to 'tmp/gt/'.
        trim_size: Will trim the video by this amount, half at the beginning and half at the end.
            Defaults to 0.
        quicktime_compatible: Whether to make the video compatible with QuickTime. Defaults to
            False.
    """

    output_path = f'{output_dir}/sequence_{sequence}.mp4'
    sm = SequenceManager(sequence, cameras=[camera])
    audio_path = os.path.join(
        sm.data_dir,
        "sequences",
        sm.sequence,
        "audio",
        "audio_recording_cleaned.ogg",
    )
    width = sm.image_width
    height = sm.image_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for i in tqdm(range(trim_size // 2, len(sm) - trim_size//2), desc='Rendering video'):
        image = sm.images[i][0].numpy() * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    out.release()
    add_audio(
        video_path=output_path,
        audio_path=audio_path,
        fps=30,
        quicktime_compatible=quicktime_compatible,
        trim_to_fit=True,
    )


def main(
    audio_path: str,
    gaussian_splats_checkpoint_path: str,
    audio_to_flame_checkpoint_path: str | None,
    flame_params_sequence: int | None,
    se_3_sequence: int = 10,  # longest sequence
    quicktime_compatible: bool = False,
    output_dir: str = 'tmp/pred',
    smoothing_method: Literal['none', 'gaussian', 'moving_average', 'savgol',
                              'exponential'] = 'none',
    intrinsics: Float[torch.Tensor, '3 3'] = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                                           [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                                           [0.0000e+00, 0.0000e+00, 1.0000e+00]]),
    world_2_cam: Float[torch.Tensor, '4 4'] | Float[torch.Tensor, 'time 4 4'] = torch.tensor(
        [[0.9058, -0.2462, -0.3448, -0.0859], [-0.0377, -0.8575, 0.5131, 0.1364],
         [-0.4220, -0.4518, -0.7860, 1.0642], [0.0000, 0.0000, 0.0000, 1.0000]]),
    image_height: int = 1604,
    image_width: int = 1100,
    background_color: Float[torch.Tensor, '3'] = torch.tensor([0.66, 0.66, 0.66])
) -> None:
    """
    Renders a video from audio.

    Args:
        audio_path (str): Path to the audio file.
        gaussian_splats_checkpoint_path (str): Path to the Gaussian splatting model checkpoint.
        output_path (str): Path to save the output video.
        audio_to_flame_checkpoint_path (str): Path to the audio-to-flame model checkpoint.
        load_flame_from_sequence (int): Load flame parameters from a sequence.
        predict_flame_from_audio (bool): Predict flame parameters from audio.
        quicktime_compatible (bool): Whether to make the video compatible with QuickTime.
        disable_se3 (bool): Whether to disable SE3 transformations.
        smoothing_method (str): Smoothing method for the trajectory. Defaults to 'none'.
    """

    # Set up
    audio_features = process_audio(audio_path)
    splats = DynamicGaussianSplatting.load_from_checkpoint(
        gaussian_splats_checkpoint_path, ckpt_path=gaussian_splats_checkpoint_path)
    splats.eval()
    splats.cuda()
    assert (audio_to_flame_checkpoint_path is not None) ^ (flame_params_sequence is not None), \
        "Either provide a checkpoint for audio-to-flame prediction or a sequence number to load " \
        "ground truth flame parameters."
    name = f'sequence_{sequence}' if flame_params_sequence is not None else audio_path.split(
        '/')[-3]
    output_dir = f'{output_dir}/{splats.name}/{mode}'
    if smoothing_mode != 'none':
        output_dir = f'{output_dir}/{smoothing_mode}'
    output_path = f'{output_dir}/{name}.mp4'
    n_frames = audio_features.shape[0]

    # Get rigging parameters
    if audio_to_flame_checkpoint_path is not None:
        flame_params = audio_to_flame(audio_to_flame_checkpoint_path, audio_features)
    else:
        sm = SequenceManager(flame_params_sequence)
        flame_params = sm.flame_params[:n_frames]
        flame_params = UnbatchedFlameParams(
            shape=flame_params.shape.cuda(),
            expr=flame_params.expr.cuda(),
            neck=flame_params.neck.cuda(),
            jaw=flame_params.jaw.cuda(),
            eye=flame_params.eye.cuda(),
            scale=flame_params.scale.cuda(),
        )
    flame_head = FlameHeadWithInnerMouth()
    flame_head = flame_head.cuda()
    rigging_params = flame_head.forward(flame_params)

    # Get SE3 transforms
    if flame_params_sequence is not None:
        se_3_sequence = flame_params_sequence
    sm = SequenceManager(se_3_sequence)
    se_3_transforms = sm.se3_transforms[:n_frames]

    # Render video
    video = render_video_dynamic_gaussian_splatting(
        checkpoint_path=gaussian_splats_checkpoint_path,
        rigging_params=rigging_params,
        se_3_transforms=se_3_transforms,
        audio_features=audio_features,
        flame_params=flame_params,
        intrinsics=intrinsics,
        world_2_cam=world_2_cam,
        image_height=image_height,
        image_width=image_width,
        background_color=background_color,
        smoothing_mode=smoothing_method,
    )

    # Save video
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (image_width, image_height))
    for i in tqdm(range(len(video)), desc='Rendering video'):
        image = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        out.write(image)

    out.release()
    add_audio(
        video_path=output_path,
        audio_path=audio_path,
        fps=30,
        quicktime_compatible=quicktime_compatible,
        trim_to_fit=True,
    )


# ==================================================================================== #
#                                       MAIN                                           #
# ==================================================================================== #

if __name__ == '__main__':
    # Arguments
    mode: Literal['flame', 'audio'
                  'gt'] = 'audio'
    sequence: int | None = 3
    audio_path: str | None = None
    gaussian_splats_checkpoint: str = 'tb_logs/dynamic_gaussian_splatting/2dgs_full_res_500k_noisy_audio/version_0/checkpoints/epoch=2-step=240000.ckpt'  # noqa
    # audio_to_flame_checkpoint: str | None = None
    audio_to_flame_checkpoint: str | None = 'tb_logs/audio_prediction/prediction_flame_default/version_0/checkpoints/epoch=99-step=7700.ckpt'  # noqa
    quicktime_compatible: bool = False
    smoothing_mode: Literal['none', 'gaussian', 'moving_average', 'savgol', 'exponential'] = 'none'
    background_color = torch.tensor([1.0, 1.0, 1.0]).cuda() * 0.8

    # Parse overridable arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--sequence', type=int)
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--gaussian_splats_checkpoint')
    parser.add_argument('--audio_to_flame_checkpoint')
    parser.add_argument('--quicktime_compatible', action='store_true')
    parser.add_argument('--smoothing_mode', type=str)

    args = parser.parse_args()
    if args.mode is not None:
        mode = args.mode
    if args.sequence is not None:
        sequence = args.sequence
    if args.audio_path is not None:
        audio_path = args.audio_path
    if args.gaussian_splats_checkpoint is not None:
        gaussian_splats_checkpoint = args.gaussian_splats_checkpoint
    if args.audio_to_flame_checkpoint is not None:
        audio_to_flame_checkpoint = args.audio_to_flame_checkpoint
    if args.quicktime_compatible:
        quicktime_compatible = True
    if args.smoothing_mode is not None:
        smoothing_mode = args.smoothing_mode

    # Processing
    if sequence is not None:
        audio_path = '../new_master_thesis/data/nersemble/Paul-audio-856/856/sequences/' \
            f'sequence_{sequence:04d}/audio/audio_recording.ogg'
        se_3_sequence = sequence
        flame_params_sequence = sequence
    else:
        se_3_sequence = 10
        flame_params_sequence = None

    # Render video
    match mode:
        case 'gt':
            model = DynamicGaussianSplatting.load_from_checkpoint(
                gaussian_splats_checkpoint, ckpt_path=gaussian_splats_checkpoint)
            window_size = model.gaussian_splatting_settings.prior_window_size
            render_gt_video(sequence=sequence, trim_size=window_size)

        case 'flame':
            assert sequence is not None, "Sequence number must be provided."
            main(
                audio_path=audio_path,
                gaussian_splats_checkpoint_path=gaussian_splats_checkpoint,
                audio_to_flame_checkpoint_path=None,
                flame_params_sequence=flame_params_sequence,
                se_3_sequence=se_3_sequence,
                quicktime_compatible=quicktime_compatible,
                smoothing_method=smoothing_mode,
                background_color=background_color,
            )

        case 'audio':
            assert audio_path is not None, "Audio path must be provided."
            main(
                audio_path=audio_path,
                gaussian_splats_checkpoint_path=gaussian_splats_checkpoint,
                audio_to_flame_checkpoint_path=audio_to_flame_checkpoint,
                flame_params_sequence=None,
                se_3_sequence=se_3_sequence,
                quicktime_compatible=quicktime_compatible,
                smoothing_method=smoothing_mode,
                background_color=background_color,
            )
