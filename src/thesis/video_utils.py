""" Video utilities for thesis project. """

import os
import tempfile

import pydub
import soundfile as sf
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
