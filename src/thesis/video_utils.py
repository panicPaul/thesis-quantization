""" Video utilities for thesis project. """

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import soundfile as sf
import torch
from jaxtyping import Float, Int, UInt8
from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)

from thesis.audio_feature_processing.audio_cleaning import pydub_to_np


def add_audio(
    video_path: str,
    audio_path: str,
    fps: int = 30,
    quicktime_compatible: bool = False,
    trim_to_fit: bool = False,
) -> None:
    """
    Add audio to a video file, overwriting the original file.
    The audio sampling rate is inferred from the video duration and audio array length.

    Args:
        video_path (str): Path to the video file
        audio_array (np.ndarray): Audio array
        fps (int): Frames per second of the video
        quicktime_compatible (bool): Whether to make the video compatible with QuickTime
        trim_to_fit (bool): Whether to trim the audio to fit the video duration
    """

    # Load the audio
    file_format = audio_path.split(".")[-1]
    if file_format == "m4a":
        # for macbook recordings
        audio = pydub.AudioSegment.from_file(audio_path)
        audio_array, sr = pydub_to_np(audio)
    else:
        audio_array, sr = sf.read(audio_path)

    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name

        # Load the video to get its duration
        video = VideoFileClip(video_path)
        video_duration = video.duration

        # Calculate the audio sampling rate
        if not trim_to_fit:
            num_audio_samples = len(audio_array)
            inferred_sample_rate = int(num_audio_samples / video_duration)
        else:
            audio_duration = len(audio_array) / sr
            assert audio_duration >= video_duration, \
                f"Audio is shorter than video, audio length: {audio_duration}, " \
                f"video length: {video_duration}"
            inferred_sample_rate = sr
            total_audio_samples = int(audio_duration * sr)
            required_audio_samples = int(video_duration * sr)
            to_trim = total_audio_samples - required_audio_samples
            left_trim = to_trim // 2
            right_trim = to_trim - left_trim
            audio_array = audio_array[left_trim:-right_trim]
            inferred_sample_rate = int(audio_array.shape[0] / video_duration)

        # Write the audio to the temporary file
        sf.write(temp_audio_path, audio_array, inferred_sample_rate)

    # Load the audio
    audio = AudioFileClip(temp_audio_path)

    # Set the audio of the video
    final_video = video.set_audio(audio)

    if not quicktime_compatible:
        # Write the result, overwriting the original file
        temp_output_path = video_path + ".temp.mp4"
        final_video.write_videofile(
            temp_output_path,
            codec="libx264",
            audio_codec="libmp3lame",  # "aac",
            fps=fps,
            audio_bitrate="128k",
        )
    else:
        # Write the result, overwriting the original file (quicktime compatible)
        temp_output_path = video_path + ".temp.mp4"
        final_video.write_videofile(
            temp_output_path,
            codec="libx264",
            audio_codec="aac",  # Changed from libmp3lame to aac
            fps=fps,
            audio_bitrate="160k",  # Increased bitrate for better quality
            # Add these parameters for better QuickTime compatibility
            ffmpeg_params=[
                "-strict",
                "-2",  # Allow experimental codecs
                "-map",
                "0:v:0",  # Map video stream
                "-map",
                "1:a:0",  # Map audio stream
                "-c:a",
                "aac",  # Force AAC codec
                "-movflags",
                "+faststart"  # Optimize for streaming
            ])

    # Close the clips
    video.close()
    audio.close()

    # Remove the temporary audio file
    os.unlink(temp_audio_path)

    # Replace the original file with the new one
    os.replace(temp_output_path, video_path)


def side_by_side(
    video_gt: str,
    video_pred: str,
    output_path: str,
) -> None:
    """
    Combine two videos side by side.

    Args:
        video_gt: Path to the first video file
        video_pred: Path to the second video file
        output_path: Path where the combined video will be saved
    """
    # Load the videos
    video_1 = VideoFileClip(video_gt)
    video_2 = VideoFileClip(video_pred)

    # Resize videos to the same height if needed
    height = min(video_1.h, video_2.h)
    video_1_resized = video_1.resize(height=height)
    video_2_resized = video_2.resize(height=height)

    # Make sure both videos have the same duration
    assert video_1.duration == video_2.duration, \
        'Videos must have the same duration to be combined side by side'
    # min_duration = min(video_1.duration, video_2.duration)
    # video_1_trimmed = video_1_resized.subclip(0, min_duration)
    # video_2_trimmed = video_2_resized.subclip(0, min_duration)
    video_1_trimmed = video_1_resized
    video_2_trimmed = video_2_resized

    # Combine the videos side by side using clips_array
    final_video = clips_array([[video_1_trimmed, video_2_trimmed]])

    # Write the result
    final_video.write_videofile(output_path)

    # Close all clips
    video_1.close()
    video_2.close()
    video_1_resized.close()
    video_2_resized.close()
    video_1_trimmed.close()
    video_2_trimmed.close()
    final_video.close()


def render_mesh_image(
    vertex_positions: Float[torch.Tensor, 'n_vertices 3'],
    faces: Int[torch.Tensor, 'n_faces 3'],
    image_height: int = 1_000,
    image_width: int = 1_000,
) -> UInt8[np.ndarray, "h w 3"]:
    """
    Generates an image of the mesh at the given time step.

    Args:
        vertex_positions: The vertex positions.
        faces: The faces of the mesh.
        image_height: The height of the image.
        image_width: The width of the image.

    Returns:
        The image of the mesh.
    """

    # Generate the vertices and faces
    vertices = vertex_positions.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    # Generate the matplotlib image
    # matplotlib.use('agg')
    fig = plt.figure(figsize=(image_width / 100, image_height / 100))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        vertices[:, 2],
        vertices[:, 0],
        faces,
        vertices[:, 1],
        shade=True,
    )
    ax.view_init(azim=10, elev=10)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_axis_off()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def _load_video(path: str) -> Float[np.ndarray, "time height width 3"]:
    """
    Load a video as a numpy array.

    Args:
        path: Path to the video file.

    Returns:
        The video as a numpy array, of shape (time, height, width, 3).
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames) / 255.0


def get_video_info(path: str) -> Tuple[int, int, int, int]:
    """
    Get video metadata without loading the full video.

    Args:
        path: Path to the video file
    Returns:
        Tuple of (frame_count, height, width, fps)
    """
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, height, width, fps


def process_frame_chunk(args: Tuple[str, int, int, int]) -> np.ndarray:
    """
    Process a chunk of frames from the video.

    Args:
        args: Tuple of (video_path, start_frame, chunk_size, total_frames)
    Returns:
        Numpy array of processed frames
    """
    video_path, start_frame, chunk_size, total_frames = args
    cap = cv2.VideoCapture(video_path)

    # Set position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Pre-allocate memory for chunk
    end_frame = min(start_frame + chunk_size, total_frames)
    actual_chunk_size = end_frame - start_frame
    frames = np.empty((actual_chunk_size, int(cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                      dtype=np.uint8)

    for i in range(actual_chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB directly into pre-allocated array
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()
    return frames


def load_video(
    path: str,
    chunk_size: int = 32,
    num_workers: Optional[int] = None,
) -> Float[np.ndarray, "time height width 3"]:
    """
    Load a video as a numpy array with parallel processing.

    Args:
        path: Path to the video file
        chunk_size: Number of frames to process in each chunk
        num_workers: Number of parallel workers (defaults to CPU count)
    Returns:
        The video as a numpy array of shape (time, height, width, 3)
    """
    # Get video information
    frame_count, height, width, fps = get_video_info(path)

    # Calculate chunks
    # num_chunks = math.ceil(frame_count / chunk_size)
    chunk_starts = range(0, frame_count, chunk_size)

    # Create arguments for parallel processing
    chunk_args = [(path, start, chunk_size, frame_count) for start in chunk_starts]

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunks = list(executor.map(process_frame_chunk, chunk_args))

    # Concatenate chunks and convert to float32
    return np.concatenate(chunks).astype(np.float32) / 255.0


def save_video(
    video: Float[np.ndarray, "time height width 3"],
    path: str,
    fps: int = 30,
    audio_path: str | None = None,
) -> None:
    """
    Save a video as a numpy array to a file.

    Args:
        video: Video as a numpy array of shape (time, height, width, 3)
        path: Path to save the video file
        fps (int): Frames per second of the video
    """
    # Get video dimensions
    num_frames, height, width, channels = video.shape

    # Convert to uint8 if input is float in [0, 1]
    if video.dtype == np.float32 or video.dtype == np.float64:
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Add audio if provided
    if audio_path is not None:
        add_audio(
            video_path=path,
            audio_path=audio_path,
            fps=fps,
            quicktime_compatible=False,
            trim_to_fit=True,
        )


def get_audio_path(sequence: int) -> str:
    """
    Get the path to the audio file for the given sequence number.

    Args:
        sequence: Sequence number
    Returns:
        Path to the audio file
    """
    audio_path = '../new_master_thesis/data/nersemble/Paul-audio-856/856/sequences/' \
        f'sequence_{sequence:04d}/audio/audio_recording.ogg'
    return audio_path


def check_video_infos(pred_dir: str, gt_dir: str = 'tmp/gt/masked') -> None:
    """
    Checks that every video in the prediction directory has the same size as the corresponding
    video in the ground truth directory.

    Args:
        pred_dir: Directory containing the predicted videos
        gt_dir: Directory containing the ground truth videos
    """

    pred_videos = os.listdir(pred_dir)
    pred_videos = [
        video for video in pred_videos if video.endswith('.mp4') and video.startswith('sequence')
    ]

    for video in pred_videos:
        pred_path = os.path.join(pred_dir, video)
        gt_path = os.path.join(gt_dir, video)
        pred_frames, pred_height, pred_width, pred_fps = get_video_info(pred_path)
        gt_frames, gt_height, gt_width, gt_fps = get_video_info(gt_path)
        assert pred_height == gt_height, f'Height mismatch for {video}'
        assert pred_width == gt_width, f'Width mismatch for {video}'
        assert pred_fps == gt_fps, f'FPS mismatch for {video}'
        assert pred_frames == gt_frames, f'Frame count mismatch for {video}'

    print('All videos have the same size and frame count.')


def combine_videos(
    dir: str,
    sequences: list[int] | None = None,
    quicktime_compatible: bool = False,
) -> None:
    """
    Combine all videos in a directory into a single video.

    Args:
        dir: Directory containing the videos
        sequences: List of sequences to concatenate (None for all sequences)
        audio_codec: Audio codec to use (default: 'mp3')
    """

    check_video_infos(dir)
    audio_codec = 'mp3' if not quicktime_compatible else 'aac'

    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist")

    # Get all MP4 files that start with 'sequence'
    file_names = os.listdir(dir)
    file_names = [
        video for video in file_names if video.endswith('.mp4') and video.startswith('sequence')
    ]

    if not file_names:
        raise ValueError(f"No video sequences found in {dir}")

    # Get sequence numbers if not provided
    if sequences is None:
        sequences = [int(video.split('_')[-1].split('.')[0]) for video in file_names]
        sequences.sort()  # Sort sequences numerically

    # Generate video paths based on sequences
    videos = [f'sequence_{sequence}.mp4' for sequence in sequences]
    video_paths = [os.path.join(dir, video) for video in videos]

    # Verify all video files exist
    missing_videos = [path for path in video_paths if not os.path.exists(path)]
    if missing_videos:
        raise FileNotFoundError(f"Missing video files: {missing_videos}")

    clips = []
    try:
        # Load all video clips
        clips = [VideoFileClip(video_path) for video_path in video_paths]

        # Verify video properties are consistent
        if not clips:
            raise ValueError("No video clips were loaded")

        reference = clips[0]
        for i, clip in enumerate(clips[1:], 1):
            if clip.h != reference.h:
                raise ValueError(
                    f"Height mismatch: sequence_{sequences[0]}.mp4 ({reference.h}) vs "
                    f"sequence_{sequences[i]}.mp4 ({clip.h})")
            if clip.w != reference.w:
                raise ValueError(f"Width mismatch: sequence_{sequences[0]}.mp4 ({reference.w}) vs "
                                 f"sequence_{sequences[i]}.mp4 ({clip.w})")

        # Combine videos
        final_clip = concatenate_videoclips(clips)

        # Generate output filename
        output_path = os.path.join(dir, 'combined_sequence.mp4')

        # Write the final video
        final_clip.write_videofile(
            output_path, codec='libx264', audio_codec=audio_codec, fps=reference.fps)

        print(f"Successfully created combined video: {os.path.basename(output_path)}")

    except Exception as e:
        raise RuntimeError(f"Error processing videos: {str(e)}") from e

    finally:
        # Clean up resources
        for clip in clips:
            try:
                clip.close()
            except:  # noqa
                pass
    change_audio_codec_to_aac(output_path)


def change_audio_codec_to_aac(video_path: str) -> None:
    """
    Convert a video's audio codec to AAC, creating a new directory with the updated video.

    Args:
        video_path: Path to the input video file

    Raises:
        FileNotFoundError: If the input video doesn't exist
        RuntimeError: If video processing fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate output path
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    output_dir = os.path.join(directory, 'aac')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    video = None
    try:
        # Load the video
        video = VideoFileClip(video_path)

        # Write the video with AAC audio codec
        video.write_videofile(
            output_path, codec='libx264', audio_codec='aac', fps=video.fps, remove_temp=True)

        print(f"Successfully created video with AAC audio: {os.path.basename(output_path)}")

    except Exception as e:
        raise RuntimeError(f"Error converting audio codec: {str(e)}") from e

    finally:
        # Clean up resources
        if video is not None:
            try:
                video.close()
            except:  # noqa
                pass
