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
from jaxtyping import Float, Int, UInt8
from scipy.signal import savgol_filter
from tqdm import tqdm

from thesis.audio_feature_processing.audio_cleaning import pydub_to_np
from thesis.audio_feature_processing.wav2vec import (
    get_processor_and_model,
    process_sequence_interpolation,
)
from thesis.audio_to_flame.windowed import AudioToFlame, prediction_loop
from thesis.code_talker.stage2_runner import Stage2Runner
from thesis.data_management import (
    SequenceManager,
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)
from thesis.flame import FlameHeadWithInnerMouth
from thesis.gaussian_splatting_legacy.rigged_gaussian_splatting import (
    RiggedGaussianSplatting,
)
from thesis.gaussian_splatting_legacy.video import GaussianSplattingVideo
from thesis.video_utils import add_audio


@torch.no_grad()
def process_audio(
    audio_path: str,
    fps: int = 30,
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
    audio_features: Float[torch.Tensor, 'time 1024'],
    checkpoint_path: str,
    batch_size: int = 32,
    padding: bool = False,
) -> tuple[Float[torch.Tensor, 'time 100'], Float[torch.Tensor, 'time 3']]:
    """
    Predicts flame parameters from audio features.

    Args:
        audio_features: Audio features.
        checkpoint_path: Path to the audio-to-flame model checkpoint.

    Returns:
        tuple: Tuple containing the expression and jaw parameters.
    """

    model = AudioToFlame.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to('cuda')
    flame_params = prediction_loop(
        model_path=checkpoint_path,
        audio_features=audio_features,
        batch_size=batch_size,
        padding=padding,
    )
    return flame_params.expr, flame_params.jaw

    # window_size = model.window_size
    # n_windows = audio_features.shape[0] - window_size + 1
    # all_expressions = torch.zeros((n_windows, 100), device='cuda')
    # all_jaw_codes = torch.zeros((n_windows, 3), device='cuda')
    # for start_idx in range(0, n_windows, batch_size):
    #     end_idx = min(start_idx + batch_size, n_windows)

    #     # Create batch of windows
    #     batch_windows = torch.stack(
    #         [audio_features[i:i + window_size] for i in range(start_idx, end_idx)])

    #     # Forward pass
    #     expressions, jaw_codes = model(batch_windows)

    #     # Store results
    #     all_expressions[start_idx:end_idx] = expressions
    #     all_jaw_codes[start_idx:end_idx] = jaw_codes

    # return all_expressions, all_jaw_codes


@torch.no_grad()
def audio_to_rigging_params(
    checkpoint_path: str,
    audio_features: Float[torch.Tensor, 'time 1024'],
) -> Float[torch.Tensor, 'time n_vertices 3']:
    """
    Args:
        checkpoint_path: Path to the audio-to-flame model checkpoint.
        audio_features: Audio features.

    Returns:
        Float[torch.Tensor, 'time n_vertices 3']: Rigging parameters.
    """
    runner = Stage2Runner.load_from_checkpoint(checkpoint_path)
    runner.eval()
    runner.cuda()
    vertices = runner.predict(audio_features)
    return vertices


def load_flame_parameters(
        sequence: int) -> tuple[Float[torch.Tensor, 'time 100'], Float[torch.Tensor, 'time 3']]:
    """
    Loads flame parameters from a file. This is done to compare the predicted flame parameters
    with the ground truth.

    Args:
        sequence: Sequence number.

    Returns:
        tuple: Tuple containing the expression and jaw parameters.
    """
    sm = SequenceManager(sequence)
    expr = sm.flame_params[:].expr.cuda()
    jaw = sm.flame_params[:].jaw.cuda()
    return expr, jaw


def load_flame_loop(n_frames: int) -> tuple[UnbatchedFlameParams, UnbatchedSE3Transform]:
    """
    We only predict the expression and jaw movement. For the other parameters, we use a loop.

    Args:
        n_frames: Number of frames to predict.

    Returns:
        tuple: Tuple containing the flame parameters and SE3 transform.
    """

    assert n_frames <= 524, 'n_frames must be less than or equal to 524 for now'
    # NOTE: this is just because I haven't written the looping code yet
    sm = SequenceManager(sequence=10)
    flame_params = sm.flame_params[:n_frames]
    flame_params = UnbatchedFlameParams(
        shape=flame_params.shape.cuda(),
        expr=torch.zeros_like(flame_params.expr).cuda(),
        neck=flame_params.neck.cuda(),
        jaw=torch.zeros_like(flame_params.jaw).cuda(),
        eye=flame_params.eye.cuda(),
        scale=flame_params.scale.cuda(),
    )
    se_3_transforms = sm.se3_transforms[:n_frames]
    se_3_transforms = UnbatchedSE3Transform(
        rotation=se_3_transforms.rotation.cuda(),
        translation=se_3_transforms.translation.cuda(),
    )
    return flame_params, se_3_transforms


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
def render_video_rigged_gaussians(
    checkpoint_path: str,
    # flame_params: UnbatchedFlameParams,
    rigging_params: Float[torch.Tensor, 'time n_vertices 3'],
    se_3_transforms: UnbatchedSE3Transform,
    audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
    padding_mode: Literal['input', 'output', 'none'] = 'output',
    intrinsics: Float[torch.Tensor, '3 3'] = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                                           [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                                           [0.0000e+00, 0.0000e+00, 1.0000e+00]]),
    world_2_cam: Float[torch.Tensor, '4 4'] | Float[torch.Tensor, 'time 4 4'] = torch.tensor(
        [[0.9058, -0.2462, -0.3448, -0.0859], [-0.0377, -0.8575, 0.5131, 0.1364],
         [-0.4220, -0.4518, -0.7860, 1.0642], [0.0000, 0.0000, 0.0000, 1.0000]]),
    image_height: int = 1604,
    image_width: int = 1100,
    background_color: Float[torch.Tensor, '3'] = torch.tensor([0.66, 0.66, 0.66]),
    use_se3: bool = True,
    smoothing_mode: Literal['none', 'gaussian', 'moving_average', 'savgol',
                            'exponential'] = 'none',
) -> UInt8[np.ndarray, 'time h w 3']:
    """
    """
    model = RiggedGaussianSplatting.load_from_checkpoint(
        checkpoint_path, ckpt_path=checkpoint_path)
    model.eval()
    model.to('cuda')

    # set up variables
    intrinsics = repeat(intrinsics, 'm n -> time m n', time=rigging_params.shape[0]).cuda()
    world_2_cam = repeat(world_2_cam, 'm n -> time m n', time=rigging_params.shape[0]).cuda()
    se_3_transforms = UnbatchedSE3Transform(
        rotation=se_3_transforms.rotation.cuda(),
        translation=se_3_transforms.translation.cuda(),
    )
    background = background_color.cuda()
    video = model.render_video(
        intrinsics=intrinsics,
        world_2_cam=world_2_cam,
        image_height=image_height,
        image_width=image_width,
        se_transforms=se_3_transforms,
        rigging_params=rigging_params,
        background=background,
    )
    return video


@torch.no_grad()
def render_video(
    checkpoint_path: str,
    flame_params: UnbatchedFlameParams,
    se_3_transforms: UnbatchedSE3Transform,
    audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
    padding_mode: Literal['input', 'output', 'none'] = 'output',
    intrinsics: Float[torch.Tensor, '3 3'] = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                                           [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                                           [0.0000e+00, 0.0000e+00, 1.0000e+00]]),
    world_2_cam: Float[torch.Tensor, '4 4'] | Float[torch.Tensor, 'time 4 4'] = torch.tensor(
        [[0.9058, -0.2462, -0.3448, -0.0859], [-0.0377, -0.8575, 0.5131, 0.1364],
         [-0.4220, -0.4518, -0.7860, 1.0642], [0.0000, 0.0000, 0.0000, 1.0000]]),
    image_height: int = 1604,
    image_width: int = 1100,
    background_color: Float[torch.Tensor, '3'] = torch.tensor([0.0, 0.0, 0.0]),
    use_se3: bool = True,
    smoothing_mode: Literal['none', 'gaussian', 'moving_average', 'savgol',
                            'exponential'] = 'none',
) -> Int[np.ndarray, 'time h w 3']:
    """
    Renders a video from flame parameters.

    Args:
        checkpoint_path (str): Path to the Gaussian splatting model checkpoint.
        flame_params (UnbatchedFlameParams): Flame parameters.
        se_3_transforms (UnbatchedSE3Transform): SE3 transforms.
        audio_features (torch.Tensor | None): Audio features. Has a shape of 'time 1024'.
            Defaults to None.
        padding_mode (str): Padding mode. Defaults to 'output'.
        intrinsics (np.ndarray): Camera intrinsics.
        world_2_cam (np.ndarray): World to camera transformation. Defaults to the test view.
        image_height (int): Image height. Defaults to 1604.
        image_width (int): Image width. Defaults to 1100.
        background_color (torch.Tensor): Background color. Defaults to [0.5, 0.5, 0.5].
        use_se3 (bool): Whether to use SE3 transformations. Defaults to True.
        smoothing_mode (str): Smoothing mode. Defaults to 'none'.

    Returns:
        Int[torch.Tensor, 'time h w 3']: Video frames.
    """

    model = GaussianSplattingVideo.load_from_checkpoint(checkpoint_path, ckpt_path=checkpoint_path)
    model.eval()
    model.to('cuda')
    window_size = model.gaussian_splatting_settings.prior_window_size

    # Apply padding to the flame parameters and audio features.
    if padding_mode == 'input':
        left_shape_padding = flame_params.shape[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_shape_padding = flame_params.shape[-1].unsqueeze(0).repeat(window_size // 2, 1)
        left_expr_padding = flame_params.expr[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_expr_padding = flame_params.expr[-1].unsqueeze(0).repeat(window_size // 2, 1)
        left_neck_padding = flame_params.neck[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_neck_padding = flame_params.neck[-1].unsqueeze(0).repeat(window_size // 2, 1)
        left_jaw_padding = flame_params.jaw[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_jaw_padding = flame_params.jaw[-1].unsqueeze(0).repeat(window_size // 2, 1)
        left_eye_padding = flame_params.eye[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_eye_padding = flame_params.eye[-1].unsqueeze(0).repeat(window_size // 2, 1)
        left_scale_padding = flame_params.scale[0].unsqueeze(0).repeat(window_size // 2, 1)
        right_scale_padding = flame_params.scale[-1].unsqueeze(0).repeat(window_size // 2, 1)
        flame_params = UnbatchedFlameParams(
            shape=torch.cat([left_shape_padding, flame_params.shape, right_shape_padding], dim=0),
            expr=torch.cat([left_expr_padding, flame_params.expr, right_expr_padding], dim=0),
            neck=torch.cat([left_neck_padding, flame_params.neck, right_neck_padding], dim=0),
            jaw=torch.cat([left_jaw_padding, flame_params.jaw, right_jaw_padding], dim=0),
            eye=torch.cat([left_eye_padding, flame_params.eye, right_eye_padding], dim=0),
            scale=torch.cat([left_scale_padding, flame_params.scale, right_scale_padding], dim=0),
        )
        if audio_features is not None:
            left_audio_padding = audio_features[0].unsqueeze(0).repeat(window_size // 2, 1)
            right_audio_padding = audio_features[-1].unsqueeze(0).repeat(window_size // 2, 1)
            audio_features = torch.cat([left_audio_padding, audio_features, right_audio_padding],
                                       dim=0)

    # Set up inputs
    n_render_frames = flame_params.expr.shape[0] - window_size + 1
    video = np.zeros((n_render_frames, image_height, image_width, 3), dtype=np.uint8)
    assert world_2_cam.shape == (4, 4), 'camera movement not yet implemented'

    if not smoothing_mode == 'none':
        n_gaussians = model.splats['means'].shape[0]
        video_means = torch.zeros((n_render_frames, n_gaussians, 3)).cuda()
        cam_2_world = torch.zeros_like(world_2_cam).cuda()
        cam_2_world[:3, :3] = world_2_cam[:3, :3].T
        cam_2_world[:3, 3] = -world_2_cam[:3, :3].T @ world_2_cam[:3, 3]
        cam_2_world[3, 3] = 1
        cam_2_world = cam_2_world.unsqueeze(0)
        for i in tqdm(range(n_render_frames), desc='Computing 3D positions'):
            means, *_ = model.pre_processing(
                infos={},
                cam_2_world=cam_2_world,
                camera_indices=None,
                flame_params=UnbatchedFlameParams(
                    shape=flame_params.shape[i:i + window_size].cuda(),
                    expr=flame_params.expr[i:i + window_size].cuda(),
                    neck=flame_params.neck[i:i + window_size].cuda(),
                    jaw=flame_params.jaw[i:i + window_size].cuda(),
                    eye=flame_params.eye[i:i + window_size].cuda(),
                    scale=flame_params.scale[i:i + window_size].cuda()),
                audio_features=audio_features[i:i + window_size].cuda()
                if audio_features is not None else None,
                cur_sh_degree=model.max_sh_degree,
            )
            video_means[i] = means
        video_means = smooth_trajectory(video_means, method=smoothing_mode)

    # Render video
    for i in tqdm(range(n_render_frames), desc='Rendering video'):
        frame, alpha, depth, _ = model.forward(
            intrinsics=intrinsics.cuda().unsqueeze(0),
            world_2_cam=world_2_cam.cuda().unsqueeze(0),
            cam_2_world=None,
            image_height=image_height,
            image_width=image_width,
            cur_sh_degree=None,
            se3_transform=UnbatchedSE3Transform(
                rotation=se_3_transforms.rotation[i].cuda(),
                translation=se_3_transforms.translation[i].cuda()) if use_se3 else None,
            background=background_color.cuda(),
            camera_indices=None,
            flame_params=UnbatchedFlameParams(
                shape=flame_params.shape[i:i + window_size].cuda(),
                expr=flame_params.expr[i:i + window_size].cuda(),
                neck=flame_params.neck[i:i + window_size].cuda(),
                jaw=flame_params.jaw[i:i + window_size].cuda(),
                eye=flame_params.eye[i:i + window_size].cuda(),
                scale=flame_params.scale[i:i + window_size].cuda()),
            audio_features=audio_features[i:i + window_size].cuda()
            if audio_features is not None else None,
            means_overwrite=None if smoothing_mode == 'none' else video_means[i],
        )
        frame = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        video[i] = frame

    # Pad the video
    if padding_mode == 'output':
        left_padding = repeat(video[0], 'h w c -> p h w c', p=window_size // 2)
        right_padding = repeat(video[-1], 'h w c -> p h w c', p=window_size // 2)
        video = np.concatenate([left_padding, video, right_padding], axis=0)

    return video


def render_gt_video(sequence: int, camera: int = 8, output_path='tmp/gt/') -> None:
    """
    Renders a video from ground truth flame parameters.

    Args:
        sequence: Sequence number.
    """

    output_path = f'{output_path}sequence_{sequence}.mp4'
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
    for i in tqdm(range(len(sm)), desc='Rendering video'):
        image = sm.images[i][0].numpy() * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    out.release()
    add_audio(
        video_path=output_path,
        audio_path=audio_path,
        fps=30,
        quicktime_compatible=False,
        trim_to_fit=False,
    )


def main(
    audio_path: str,
    gaussian_splats_checkpoint_path: str,
    output_path: str | None,
    audio_to_flame_checkpoint_path: str | None,
    load_flame_from_sequence: int | None,
    predict_flame_from_audio: bool,
    quicktime_compatible: bool,
    disable_se3: bool = False,
    smoothing_method: Literal['none', 'gaussian', 'moving_average', 'savgol',
                              'exponential'] = 'none',
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
    flame_params, se_3_transforms = load_flame_loop(audio_features.shape[0])
    if predict_flame_from_audio:
        assert audio_to_flame_checkpoint_path is not None, \
            'audio_to_flame_checkpoint_path is required'
        # expressions, jaws = audio_to_flame(audio_features, audio_to_flame_checkpoint_path)
        flame_vertices = audio_to_rigging_params(audio_to_flame_checkpoint_path, audio_features)
    else:
        expressions, jaws = load_flame_parameters(load_flame_from_sequence)
        flame_params, se_3_transforms = load_flame_loop(expressions.shape[0])
        flame_params = UnbatchedFlameParams(
            shape=flame_params.shape.cuda(),
            expr=expressions.cuda(),
            neck=flame_params.neck.cuda(),
            jaw=jaws.cuda(),
            eye=flame_params.eye.cuda(),
            scale=flame_params.scale.cuda(),
        )
        flame_head = FlameHeadWithInnerMouth()
        flame_head = flame_head.cuda()
        flame_vertices = flame_head.forward(flame_params)

    # Render video
    # video = render_video(
    #     checkpoint_path=gaussian_splats_checkpoint_path,
    #     flame_params=flame_params,
    #     se_3_transforms=se_3_transforms,
    #     audio_features=audio_features,
    #     padding_mode='none',
    #     use_se3=not disable_se3,
    #     smoothing_mode=smoothing_method,
    # )
    video = render_video_rigged_gaussians(
        checkpoint_path=gaussian_splats_checkpoint_path,
        rigging_params=flame_vertices,
        se_3_transforms=se_3_transforms,
        audio_features=audio_features,
        padding_mode='none',
        use_se3=not disable_se3,
        smoothing_mode=smoothing_method,
    )

    # save video as numpy array to tmp/video.npy for debugging purposes only
    np.save('tmp/video.npy', video)

    # Save video
    # TODO: make the naming scheme better
    name = 'video_pred.mp4' if predict_flame_from_audio else 'video_gt.mp4'
    output_path = f'tmp/{name}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (video.shape[2], video.shape[1]))
    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the video writer
    out.release()

    add_audio(
        video_path=output_path,
        audio_path=audio_path,
        fps=30,
        quicktime_compatible=quicktime_compatible,
        trim_to_fit=False,
    )


# ==================================================================================== #
#                                       MAIN                                           #
# ==================================================================================== #

if __name__ == '__main__':
    # hacky manual defaults
    # audio_path = 'tmp/audio_recording_cleaned_s3.ogg'
    # audio_path = 'tmp/audio_recording_cleaned.ogg'
    audio_path = 'tmp/audio_recording_cleaned_s100.ogg'
    # audio_path = 'tmp/test.m4a'
    # gs_checkpoint = 'tb_logs/video/direct_pred/version_2/checkpoints/epoch=38-step=99000.ckpt'
    gs_checkpoint = 'tb_logs/rigged_gs/dynamic/seq_3/version_0/checkpoints/epoch=4-step=481500.ckpt'  # noqa
    # a2f_checkpoint = 'tb_logs/audio_prediction/prediction_new_vq_vae/version_4/checkpoints/epoch=99-step=7700.ckpt' # noqa
    # a2f_checkpoint = 'tb_logs/audio_prediction/prediction_new_vq_vae/version_6/checkpoints/epoch=99-step=7700.ckpt' # noqa
    a2f_checkpoint = 'tb_logs/audio_prediction/prediction_new_vq_vae_fsq/version_1/checkpoints/epoch=99-step=7700.ckpt'  # noqa
    output_path = 'tmp/video.mp4'
    load_sequence = 100
    load_gt = False
    quicktime_compatible = False
    disable_se3 = False
    smoothing_method = 'none'
    # smoothing_method = 'savgol'

    # cli overrides
    parser = argparse.ArgumentParser(description='Render a video from audio.')
    parser.add_argument('-a', '--audio_path', type=str, help='Path to the audio file.')
    parser.add_argument(
        '-gs',
        '--gaussian_splats',
        type=str,
        help='Path to the Gaussian splatting model checkpoint.')
    parser.add_argument(
        '-af', '--audio_to_flame', type=str, help='Path to the audio-to-flame model checkpoint.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the output video.')
    parser.add_argument(
        '-s', '--load_sequence', type=int, help='Load flame parameters from a sequence.')
    parser.add_argument(
        '-gt',
        '--load_ground_truth',
        action='store_true',
        help='Load ground truth flame parameters instead of predicting them from audio.')
    parser.add_argument(
        '-qc',
        '--quicktime_compatible',
        action='store_true',
        help='Make the video compatible with QuickTime.')
    parser.add_argument(
        '-ds3', '--disable_se3', action='store_true', help='Disable SE3 transformations.')
    parser.add_argument(
        '-sm',
        '--smoothing_method',
        type=str,
        help='Smoothing method for the trajectory. One of: '
        "'none', 'gaussian', 'moving_average', 'savgol', 'exponential'")
    args = parser.parse_args()

    if args.audio_path:
        audio_path = args.audio_path
    if args.gaussian_splats:
        gs_checkpoint = args.gaussian_splats
    if args.audio_to_flame:
        a2f_checkpoint = args.audio_to_flame
    if args.output_path:
        output_path = args.output_path
    if args.load_sequence:
        load_sequence = args.load_sequence
    if args.load_ground_truth:
        load_gt = True
    if args.quicktime_compatible:
        quicktime_compatible = True
    if args.disable_se3:
        disable_se3 = True
    if args.smoothing_method:
        smoothing_method = args.smoothing_method

    # pretty print the arguments
    print(f'audio_path: {audio_path}')
    print(f'gs_checkpoint: {gs_checkpoint}')
    print(f'audio_to_flame_checkpoint: {a2f_checkpoint}')
    print(f'output_path: {output_path}')
    print(f'load_sequence: {load_sequence}')
    print(f'load_ground_truth: {load_gt}')
    print(f'quicktime_compatible: {quicktime_compatible}')
    print(f'disable_se3: {disable_se3}')
    print(f'smoothing_method: {smoothing_method}')

    main(
        audio_path=audio_path,
        gaussian_splats_checkpoint_path=gs_checkpoint,
        output_path=output_path,
        audio_to_flame_checkpoint_path=a2f_checkpoint,
        load_flame_from_sequence=load_sequence,
        predict_flame_from_audio=not load_gt,
        quicktime_compatible=quicktime_compatible,
        disable_se3=disable_se3,
        smoothing_method=smoothing_method,
    )
