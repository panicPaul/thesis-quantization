""" Some dataset statistics. """

import plotly.graph_objects as go

from thesis.constants import (
    DATA_DIR_NERSEMBLE,
    SEGMENTATION_CLASSES,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
)
from thesis.data_management import SequenceManager


def plot_dataset_stats(
    data_dir: str = DATA_DIR_NERSEMBLE,
    train_sequences: list[int] = TRAIN_SEQUENCES,
    test_sequences: list[int] = TEST_SEQUENCES,
) -> go.Figure:
    """
    Plots some statistics about the dataset.

    Args:
        data_dir: The directory containing the dataset.
        train_sequences: The training sequences.
        test_sequences: The testing sequences.
    """

    n_frames = []
    is_train = [True for _ in train_sequences] + [False for _ in test_sequences]
    sequence_numbers = []

    for sequence in train_sequences + test_sequences:
        sequence_numbers.append(sequence)
        sm = SequenceManager(sequence, data_dir=data_dir)
        n_frames.append(len(sm))

    # sort by number of frames
    sort_idx = sorted(range(len(n_frames)), key=lambda k: n_frames[k])
    n_frames = [n_frames[i] for i in sort_idx]
    is_train = [is_train[i] for i in sort_idx]
    sequence_numbers = [sequence_numbers[i] for i in sort_idx]

    # create bar plot figure, with train_sequences in blue and test_sequences in yellow
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sequence_numbers,
            y=n_frames,
            marker_color=['blue' if t else 'red' for t in is_train],
        ))
    fig.update_layout(
        title='Number of frames per sequence',
        xaxis_title='Sequence number',
        yaxis_title='Number of frames',
        barmode='group',
    )

    return fig
