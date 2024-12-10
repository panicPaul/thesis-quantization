""" Runs the Gaussian Model Evaluation. """

import torch
import numpy as np
from jaxtyping import Float, UInt8
from typing import Literal
from thesis.data_management import SequenceManager, UnbatchedFlameParams, UnbatchedSE3Transform
from thesis.gaussian_splatting.dynamic_model import DynamicGaussianSplatting
import os
import pathlib
import gzip
import bz2
import lzma
import time
from moviepy import VideoFileClip, AudioFileClip, ImageSequenceClip
from thesis.video_utils import get_audio_path
from thesis.evaluation import _EvaluationComputer
import yaml
from thesis.utils import assign_segmentation_class
from einops import repeat
from tqdm import tqdm
from thesis.video_utils import side_by_side


# MARK: - Compression Functions
def save_compressed_array(array, filepath, compression='lzma') -> None:
    """
    Save a numpy array to a compressed file. The compression type can be one of 'gzip', 'bz2', or 'lzma'. They are ranked in order of speed, with 'gzip' being the fastest and 'lzma' being the slowest but most efficient.
    """
    filepath = pathlib.Path(filepath)
    t = time.time()

    # Ensure the file extension matches the compression type
    if not str(filepath).endswith(compression):
        filepath = filepath.with_suffix(f'.{compression}')

    # Choose the appropriate compression method
    if compression == 'gzip':
        open_func = gzip.open
    elif compression == 'bz2':
        open_func = bz2.open
    elif compression == 'lzma':
        open_func = lzma.open
    else:
        raise ValueError("Compression must be one of: 'gzip', 'bz2', 'lzma'")

    # Save the compressed array
    with open_func(filepath, 'wb') as f:
        np.save(f, array)

    print(f"Saved compressed array to: {filepath} in {time.time() - t:.2f} seconds")


def load_compressed_array(filepath):
    """ Load a numpy array from a compressed file. The compression type is determined from the file extension. """
    filepath = pathlib.Path(filepath)
    t = time.time()

    # Determine the compression type from the file extension
    suffix = filepath.suffix.lower()
    if suffix == '.gzip':
        open_func = gzip.open
    elif suffix == '.bz2':
        open_func = bz2.open
    elif suffix == '.lzma':
        open_func = lzma.open
    else:
        raise ValueError("File must have extension: .gzip, .bz2, or .lzma")

    # Load and decompress the array
    with open_func(filepath, 'rb') as f:
        return np.load(f)
    print(f"Loaded compressed array from: {filepath} in {time.time() - t:.2f} seconds")


# MARK: - Rendering and Evaluation Functions
def render_sequence(
    sequence: int,
    model: DynamicGaussianSplatting,
    rigging_params: Float[torch.Tensor, 'time n_vertices 3'] | None = None,
    image_height: int = 1604,
    image_width: int = 1100,
    intrinsics: Float[torch.Tensor, 'time 3 3'] | None = None,
    world_2_cam: Float[torch.Tensor, 'time 4 4'] | None = None,
    # cam_2_world: Float[torch.Tensor, 'time 4 4'] | None = None,
    background: Float[torch.Tensor, '3'] = torch.ones(3),
    output_dir='ablations',
    experiment: str = 'flame',
    compression: Literal['gzip', 'bz2', 'lzma'] = 'lzma',
    video_only: bool = False,
) -> UInt8[np.ndarray, 'time H W 3']:
    """
    Renders a sequence using the model. By default the sequence parameters are used, with the validation camera intrinsics and position.

    Args:
        sequence (int): The sequence to render.
        model (DynamicGaussianSplatting): The model to use for rendering.
        rigging_params (torch.Tensor | None): The rigging parameters to use for the sequence.
            Has shape (time, n_vertices, 3).
        image_height (int): The height of the image to render.
        image_width (int): The width of the image to render.
        intrinsics (torch.Tensor | None): The camera intrinsics to use for rendering.
            Has shape (time, 3, 3).
        world_2_cam (torch.Tensor | None): The world to camera transformation to use for rendering.
            Has shape (time, 4, 4).
        cam_2_world (torch.Tensor | None): The camera to world transformation to use for rendering.
            Has shape (time, 4, 4).
        background (torch.Tensor): The background color to use for rendering.
            Has shape (3, ).
        name_appendix (str | None): An optional appendix to add to the file name.

    Returns:
        np.ndarray: The rendered video, with shape (time, H, W, 3) in uint8 format.
    """

    # Load everything from the sequence manager
    model.cuda()
    model.eval()
    sm = SequenceManager(sequence)
    if rigging_params is None:
        flame_params = sm.flame_params[:]
        flame_params = UnbatchedFlameParams(
            shape=flame_params.shape.cuda(),
            expr=flame_params.expr.cuda(),
            neck=flame_params.neck.cuda(),
            jaw=flame_params.jaw.cuda(),
            eye=flame_params.eye.cuda(),
            scale=flame_params.scale.cuda(),
        )
        flame_head = model.flame_head
        rigging_params = flame_head.forward(flame_params)
    windowed_rigging_params = rigging_params
    audio_features = sm.audio_features[:].cuda()
    se3_transforms = sm.se3_transforms[:]
    se3_transforms = UnbatchedSE3Transform(
        translation=se3_transforms.translation.cuda(),
        rotation=se3_transforms.rotation.cuda(),
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
    output_dir = os.path.join(output_dir, model.name, experiment)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'sequence_{sequence}.{compression}')
    if not video_only:
        print(f'Saving video to: {output_path}')
        save_compressed_array(array=video, filepath=output_path, compression=compression)
        print('Done!\n')

    # Render the video
    video_output_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_output_dir, exist_ok=True)
    video_output_path = os.path.join(video_output_dir, f'sequence_{sequence}.mp4')
    audio_path = get_audio_path(sequence)
    clip_by = (model.gaussian_splatting_settings.prior_window_size - 1) / 30
    save_as_video(
        video=video,
        output_path=video_output_path,
        audio_path=audio_path,
        clip_audio_by=clip_by,
    )

    # Get the ground truth video
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    evaluation_output_path = os.path.join(output_dir, 'results', f'sequence_{sequence}.yaml')
    print('Loading ground truth video...')
    gt_video = get_gt_video(sequence=sequence)
    gt_video_path = os.path.join(video_output_dir, f'sequence_{sequence}_gt.mp4')
    save_as_video(
        video=gt_video,
        output_path=gt_video_path,
        audio_path=audio_path,
        clip_audio_by=clip_by,
    )

    # Save side by side video
    side_by_side_path = os.path.join(video_output_dir, f'sequence_{sequence}_side_by_side.mp4')
    side_by_side(
        video_gt=gt_video_path,
        video_pred=video_output_path,
        output_path=side_by_side_path,
    )

    # Evaluate the video
    print('Evaluating sequence...')
    evaluate_sequence(
        gt_video=gt_video,
        pred_video=video,
        output_path=evaluation_output_path,
    )

    return video


# MARK: - Video Functions
def save_as_video(
    video: UInt8[np.ndarray, 'time H W 3'],
    output_path: str,
    audio_path: str | None = None,
    clip_audio_by: float | None = None,
) -> None:
    """ 
    Saves a video to disk. If an audio file is provided, it will be added to the video.
    
    Args:
        video (np.ndarray): The video to save, with shape (time, H, W, 3).
        output_dir (str): The directory to save the video.
        audio_path (str | None): The path to an audio file to add to the video.
        clip_audio_by (float | None): If the model uses a windowed approach, the output video 
            will be shorter by window_size-1 frames. To combat this we have to clip the audio to 
            the correct length. Audio clip length is (window_size-1) / fps, i.e. the total amount 
            of time that the video is shorter by. The audio will be clipped by half this amount
            on either side.
    """
    # Default video settings
    fps = 30

    # Create video clip from numpy array
    video_clip = ImageSequenceClip([frame for frame in video], fps=fps)

    if audio_path is not None:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio = AudioFileClip(audio_path)

        # Handle audio clipping if specified
        if clip_audio_by is not None:
            if not isinstance(clip_audio_by, (int, float)):
                raise ValueError(
                    f"Expected number for audio_clip_length, got {type(clip_audio_by)}")

            # Calculate start and end times for audio clip
            clip_amount = clip_audio_by / 2
            start_time = clip_amount
            end_time = audio.duration - clip_amount

            # Clip audio
            audio = audio.subclipped(start_time, end_time)

        # Set audio for video clip
        video_clip = video_clip.with_audio(audio)

    # Write video file with compression
    video_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='mp3' if audio_path else None,
        fps=fps,
        preset='veryslow',  # Compression preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        bitrate='5000k')

    # Clean up
    video_clip.close()
    if audio_path is not None:
        audio.close()


# MARK: - Evaluation Functions
def get_gt_video(
        sequence: int,
        background: Float[torch.Tensor, "3"] = torch.ones(3),
) -> UInt8[np.ndarray, 'time H W 3']:
    """ 
    Loads the ground truth video for the sequence. 
    
    Args:
        sequence (int): The sequence to load.
        background (torch.Tensor): The background color.
    """
    sm = SequenceManager(sequence, cameras=[8])
    images = []

    for t in tqdm(range(10, sm.n_time_steps - 10), desc="Loading GT video"):
        gt_video = sm.images[t:t + 1, 0].cuda()
        alpha_map = sm.alpha_maps[t:t + 1, 0].cuda()
        segmentation_mask = sm.segmentation_masks[t:t + 1, 0].cuda()
        segmentation_classes = assign_segmentation_class(segmentation_mask)
        segmentation_classes = segmentation_classes
        background_mask = torch.where(segmentation_classes == 0, 1, 0)
        jumper_mask = torch.where(segmentation_classes == 2, 1, 0)
        alpha_map = alpha_map * (1-jumper_mask) * (1-background_mask)  # (time, H, W)
        alpha_map = repeat(alpha_map, "time H W -> time H W f", f=3)
        bg = repeat(
            background.cuda(),
            "f -> time H W f",
            time=alpha_map.shape[0],
            H=alpha_map.shape[1],
            W=alpha_map.shape[2],
        )
        gt_video = gt_video*alpha_map + bg * (1-alpha_map)
        gt_video = gt_video * 255
        images.append(gt_video.cpu())
    images = torch.cat(images, dim=0)
    return images.to(torch.uint8).cpu().numpy()


def evaluate_sequence(
    gt_video: UInt8[np.ndarray, 'time H W 3'],
    pred_video: UInt8[np.ndarray, 'time H W 3'],
    output_path: str,
    cut_lower_n_pixels: int = 350,
) -> None:
    """ Runs the evaluation and saves the results to disk. """

    eval_computer = _EvaluationComputer()
    eval_computer.cuda()

    # Cut the lower part of the video
    gt_video = torch.from_numpy(gt_video[:, cut_lower_n_pixels:]).float() / 255
    pred_video = torch.from_numpy(pred_video[:, cut_lower_n_pixels:]).float() / 255

    results = eval_computer.forward(gt_video=gt_video, pred_video=pred_video, device='cuda')
    for key, value in results.items():
        print(f"{key}: {value}")
    results['num_frames'] = gt_video.shape[0]

    # Save the results as yaml file
    with open(output_path, 'w') as f:
        yaml.dump(results, f)


# MARK: - Main
# ==================================================================================== #
#                                    MAIN FUNCTION                                     #
# ==================================================================================== #

if __name__ == '__main__':

    ablations = [
        'no_flame_prior',  # 0
        '/home/schlack/thesis-quantization/tb_logs/dynamic_gaussian_splatting/ablations_final/just_flame_prior/version_0/checkpoints/epoch=2-step=270000.ckpt',  # 1 just vanilla flame 
        '/home/schlack/thesis-quantization/tb_logs/dynamic_gaussian_splatting/ablations_final/just_flame_prior_inner_mouth/version_0/checkpoints/epoch=2-step=270000.ckpt',  # 2 just flame inner mouth
        '/home/schlack/thesis-quantization/tb_logs/dynamic_gaussian_splatting/ablations_final/with_per_gaussian/version_0/checkpoints/epoch=2-step=270000.ckpt',  # 3 with per gaussian
        '/home/schlack/thesis-quantization/tb_logs/dynamic_gaussian_splatting/ablations_final/with_color_mlp/version_0/checkpoints/epoch=2-step=240000.ckpt',  # 4 with color mlp
        '/home/schlack/thesis-quantization/tb_logs/dynamic_gaussian_splatting/ablations_final/oversample/version_0/checkpoints/epoch=2-step=240000.ckpt',  # 5 oversample
        'tb_logs/dynamic_gaussian_splatting/ablations_final/with_color_mlp_2dgs/version_0/checkpoints/epoch=2-step=240000.ckpt',  # 6 with 2dgs 
    ]

    # with 2dgs only has 24 for some reason...
    # i have 2, and 6,, five is out?
    model_path = ablations[4]  # missing are 1, 3, 4
    video_only = True
    audio = True

    # ------------------------------------------------------------------------------------ #

    experiment_name = 'flame' if not audio else 'audio'
    # Load the model
    model = DynamicGaussianSplatting.load_from_checkpoint(model_path, ckpt_path=model_path)

    # Render the sequence
    for sequence in range(80, 102):
        if audio:
            rigging_params = torch.load(
                f'/home/schlack/thesis-quantization/saved_vertex_preds/sequence_{sequence}.pt')
            # there are always 2 frames missing for some reason... pad with mirror
            rigging_params = torch.cat([rigging_params[:1], rigging_params, rigging_params[-1:]])
        else:
            rigging_params = None
        pred_video = render_sequence(
            sequence=sequence,
            model=model,
            video_only=video_only,
            experiment=experiment_name,
            rigging_params=rigging_params,
        )
