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
from moviepy.editor import AudioFileClip, VideoFileClip, clips_array

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
            inferred_sample_rate = sr
            audio_duration = len(audio_array) / sr
            assert audio_duration >= video_duration, \
                f"Audio is shorter than video, audio length: {audio_duration}, " \
                f"video length: {video_duration}"
            trim_length = int(video_duration * sr)
            left_trim_point = (len(audio_array) - trim_length) // 2
            right_trim_point = len(audio_array) - left_trim_point
            audio_array = audio_array[left_trim_point:right_trim_point]

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
    min_duration = min(video_1.duration, video_2.duration)
    video_1_trimmed = video_1_resized.subclip(0, min_duration)
    video_2_trimmed = video_2_resized.subclip(0, min_duration)

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
