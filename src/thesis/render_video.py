""" Renders a video."""

import argparse
from typing import Literal

import cv2
import librosa
import numpy as np
import pydub
import soundfile as sf
import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from thesis.audio_feature_processing.audio_cleaning import pydub_to_np
from thesis.audio_feature_processing.wav2vec import (
    get_processor_and_model,
    process_sequence_interpolation,
)
from thesis.audio_to_flame.windowed import AudioToFlame
from thesis.data_management import (
    SequenceManager,
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)
from thesis.gaussian_splatting.video import GaussianSplattingVideo
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

    expressions = []
    jaws = []
    for i in tqdm(
            range(0, audio_features.shape[0], batch_size), desc='Predicting flame parameters'):
        batch = audio_features[i:i + batch_size].cuda()
        expr, jaw = model.forward(batch)
        expressions.append(expr)
        jaws.append(jaw)
    expressions = torch.cat(expressions, dim=0)
    jaws = torch.cat(jaws, dim=0)

    return expressions, jaws


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
    se_3_transforms = UnbatchedSE3Transform(
        rotation=torch.zeros(n_frames, 3, 3).cuda(),
        translation=torch.zeros(n_frames, 3).cuda(),
    )
    return flame_params, se_3_transforms


@torch.no_grad()
def render_video(
    checkpoint_path: str,
    flame_params: UnbatchedFlameParams,
    se_3_transforms: UnbatchedSE3Transform,
    audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
    padding_mode: Literal['input', 'output'] = 'output',
    intrinsics: Float[torch.Tensor, '3 3'] = torch.tensor([[4.0987e+03, 0.0000e+00, 5.4957e+02],
                                                           [0.0000e+00, 4.0988e+03, 8.0185e+02],
                                                           [0.0000e+00, 0.0000e+00, 1.0000e+00]]),
    world_2_cam: Float[torch.Tensor, '4 4'] | Float[torch.Tensor, 'time 4 4'] = torch.tensor(
        [[0.9058176279067993, -0.24615782499313354, -0.34481990337371826, -0.08591265231370926],
         [-0.03770417720079422, -0.8575000762939453, 0.5131004452705383, 0.13644646108150482],
         [-0.42198675870895386, -0.4517742693424225, -0.7860198020935059, 1.0642484426498413],
         [0.0, 0.0, 0.0, 1.0]]),
    image_height: int = 1604,
    image_width: int = 1100,
    background_color: Float[torch.Tensor, '3'] = torch.tensor([0.5, 0.5, 0.5]),
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
                translation=se_3_transforms.translation[i].cuda()),
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
        )
        frame = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        video[i] = frame

    # Pad the video
    # if padding_mode == 'output':
    #     left_padding = video[0][None, :].repeat((window_size // 2, 1, 1, 1))
    #     right_padding = video[-1][None, :].repeat((window_size // 2, 1, 1, 1))
    #     video = np.concatenate([left_padding, video, right_padding], axis=0)

    return video


def main(
    audio_path: str,
    gaussian_splats_checkpoint_path: str,
    output_path: str | None = None,
    audio_to_flame_checkpoint_path: str | None = None,
    load_flame_from_sequence: int | None = None,
    predict_flame_from_audio: bool = True,
) -> None:

    # Set up
    audio_features = process_audio(audio_path)
    if predict_flame_from_audio:
        expressions, jaws = audio_to_flame(audio_features, audio_to_flame_checkpoint_path)
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

    # Render video
    video = render_video(
        checkpoint_path=gaussian_splats_checkpoint_path,
        flame_params=flame_params,
        se_3_transforms=se_3_transforms,
        audio_features=audio_features,
        padding_mode='output',
    )

    # save video as numpy array to tmp/video.npy
    np.save('tmp/video.npy', video)

    # Save video
    if output_path is None:
        name = 'video_pred.mp4' if predict_flame_from_audio else 'video_gt.mp4'
        output_path = f'tmp/{name}'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=30.0,
        frameSize=(video.shape[1], video.shape[2]))
    for frame in tqdm(video, desc="Saving video"):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    # add_audio(video_path=output_path, audio_path=audio_path, fps=30)


# ==================================================================================== #
#                                       MAIN                                           #
# ==================================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a video from audio.')
    parser.add_argument('-a', '--audio_path', type=str, help='Path to the audio file.')
    parser.add_argument(
        '-gs',
        '--gaussian_splats_checkpoint_path',
        type=str,
        help='Path to the Gaussian splatting model checkpoint.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the output video.')
    parser.add_argument(
        '-af',
        '--audio_to_flame_checkpoint_path',
        type=str,
        help='Path to the audio-to-flame model checkpoint.')
    parser.add_argument(
        '-s',
        '--load_flame_from_sequence',
        type=int,
        help='Load flame parameters from a sequence.')
    parser.add_argument(
        '-gt',
        '--load_ground_truth',
        action='store_true',
        help='Load ground truth flame parameters instead of predicting them from audio.')
    args = parser.parse_args()

    main(
        audio_path=args.audio_path,
        gaussian_splats_checkpoint_path=args.gaussian_splats_checkpoint_path,
        output_path=args.output_path,
        audio_to_flame_checkpoint_path=args.audio_to_flame_checkpoint_path,
        load_flame_from_sequence=args.load_flame_from_sequence,
        predict_flame_from_audio=not args.load_ground_truth,
    )
