"""
Script to clean the audio signal by removing noise from the audio signal.
"""

import librosa
import noisereduce as nr  # type: ignore
import numpy as np
import pydub
import soundfile as sf  # type: ignore
from tqdm import tqdm

from thesis.constants import DATA_DIR_NERSEMBLE


# TODO: adapt this to the new code base
def pydub_to_np(audio: pydub.AudioSegment) -> tuple[np.ndarray, int]:
    """
    From https://stackoverflow.com/questions/38015319/\
        how-to-create-a-numpy-array-from-a-pydub-audiosegment

    Args:
        audio (pydub.AudioSegment): The audio segment to convert to numpy array.

    Returns:
        tuple[np.ndarray, int]: The audio signal as a numpy array and
        the sample rate.
    """
    return (
        np.array(audio.get_array_of_samples(), dtype=np.float32)
        .reshape((-1, audio.channels))
        .squeeze(1)
        / (1 << (8 * audio.sample_width - 1)),
        audio.frame_rate,
    )


def clean_single_audio_file(
    input_path: str,
    output_path: str | None = None,
    output_sampling_rate: int | None = 16_000,
    overwrite_input: bool = False,
) -> None:
    """
    Clean the audio signal by removing noise from the audio signal.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str | None): Path to the output audio file.
        output_sampling_rate (int): The sampling rate of the output audio file. Defaults
            to 16000 to work with the wav2vec2 model.
        overwrite_input (bool): Whether to overwrite the input audio file. Defaults to
            False.
    """
    if input_path == output_path and not overwrite_input:
        raise ValueError(
            "Input and output path cannot be the same without explicit overwrite."
        )
    if output_path is None and not overwrite_input:
        raise ValueError("Output path must be provided if overwrite_input is False.")
    if overwrite_input:
        output_path = input_path

    file_format = input_path.split(".")[-1]
    if file_format == "m4a":
        # for macbook recordings
        audio = pydub.AudioSegment.from_file(input_path)
        audio, sr = pydub_to_np(audio)
    else:
        audio, sr = sf.read(input_path)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, thresh_n_mult_nonstationary=1)
    if output_sampling_rate is not None:
        reduced_noise = librosa.resample(
            reduced_noise, orig_sr=sr, target_sr=output_sampling_rate
        )
    else:
        output_sampling_rate = sr
    sf.write(output_path, reduced_noise, samplerate=output_sampling_rate)


def clean_audio_files(
    data_directory: str = DATA_DIR_NERSEMBLE,
    sequence_names: list[str] | str = [f"sequence_{i:04d}" for i in range(3, 102)],
) -> None:
    """
    Clean the audio signal by removing noise from the audio signal for all audio files

    Args:
        data_directory (str): The directory containing the data.
        sequence_names (list[str] | str): The names of the sequences to clean. Defaults
            to all sequences from 3 to 101.
    """
    if isinstance(sequence_names, str):
        sequence_names = [sequence_names]

    for sequence_name in tqdm(sequence_names, desc="Cleaning audio files"):
        clean_single_audio_file(
            input_path=(
                f"{data_directory}/sequences/{sequence_name}/"
                "audio/audio_recording.ogg"
            ),
            output_path=f"{data_directory}/sequences/{sequence_name}/"
            "audio/audio_recording_cleaned.ogg",
        )


if __name__ == "__main__":

    clean_audio_files()
