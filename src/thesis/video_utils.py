""" Video utilities for thesis project. """

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydub
import soundfile as sf
import torch
from jaxtyping import Float, Int, UInt8
from moviepy.editor import AudioFileClip, VideoFileClip

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
            assert audio_duration >= video_duration, "Audio is shorter than video"
            trim_length = int(video_duration * sr)
            left_trim = (len(audio_array) - trim_length) // 2
            right_trim = len(audio_array) - left_trim
            audio_array = audio_array[left_trim:right_trim]

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


from moviepy.editor import VideoFileClip, clips_array


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
