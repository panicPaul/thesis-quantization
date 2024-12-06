""" Evaluation of two videos. """

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfvvdp
import torch
import torch.nn as nn
import yaml
from jaxtyping import Float
from plotly.subplots import make_subplots
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from thesis.video_utils import load_video


class _EvaluationComputer(nn.Module):
    """ Compute evaluation metrics. """

    def __init__(self) -> None:
        super().__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
        # self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(
        #     net_type='vgg', normalize=True)
        self.reference_image = pyfvvdp.load_image_as_array

    def forward(
        self,
        gt_video: Float[torch.Tensor, 'time height width 3'],
        pred_video: Float[torch.Tensor, 'time height width 3'],
        device: torch.device | str = 'cpu',
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            gt_video: Ground truth video.
            pred_video: Predicted video.
            device: Device to use for computation.

        Returns:
            Dictionary of evaluation metrics.
        """

        if gt_video.shape != pred_video.shape:
            print('Shapes of gt_video and pred_video are different: '
                  f'{gt_video.shape} != {pred_video.shape}')
        assert gt_video.max() <= 1.0 and gt_video.min() >= 0.0, 'gt_video should be in [0, 1]'
        assert pred_video.max() <= 1.0 and pred_video.min(
        ) >= 0.0, 'pred_video should be in [0, 1]'
        # trim to the minimum length
        if gt_video.shape[0] < pred_video.shape[0]:
            trim_length = pred_video.shape[0] - gt_video.shape[0]
            left_trim = trim_length // 2
            right_trim = trim_length - left_trim
            pred_video = pred_video[left_trim:-right_trim]
        elif gt_video.shape[0] > pred_video.shape[0]:
            trim_length = gt_video.shape[0] - pred_video.shape[0]
            left_trim = trim_length // 2
            right_trim = trim_length - left_trim
            gt_video = gt_video[left_trim:-right_trim]

        # PyTorch video tensors are in the format (time, channel, height, width)
        gt_video = gt_video.permute(0, 3, 1, 2)
        pred_video = pred_video.permute(0, 3, 1, 2)

        # Compute evaluation metrics
        psnr = 0.0
        ssim = 0.0
        lpips_alex = 0.0
        l1 = 0.0
        # lpips_vgg = 0.0
        chunk_size = 1  # NOTE: ssim breaks if chunk_size > 1
        for i in tqdm(range(0, len(gt_video), chunk_size), desc='Computing evaluation metrics'):
            cur_gt_video = gt_video[i * chunk_size:(i+1) * chunk_size]
            cur_gt_video = cur_gt_video.to(device)
            cur_pred_video = pred_video[i * chunk_size:(i+1) * chunk_size]
            cur_pred_video = cur_pred_video.to(device)
            psnr += self.psnr(cur_gt_video, cur_pred_video).item()
            ssim += self.ssim(cur_gt_video, cur_pred_video).item()
            lpips_alex += self.lpips_alex(cur_gt_video, cur_pred_video).item()
            l1 += torch.nn.functional.l1_loss(cur_gt_video, cur_pred_video).item()
            # lpips_vgg += self.lpips_vgg(cur_gt_video, cur_pred_video).item()

        # compute fov_video_vdp
        # NOTE: We can't always load the entire video into GPU memory, so we compute this on the
        #       CPU
        print('Computing fov_video_vdp')
        gt_video = gt_video.permute(0, 2, 3, 1)
        pred_video = pred_video.permute(0, 2, 3, 1)
        fov_score, _ = fov_video_vdp(gt_video, pred_video)

        return {
            'psnr': psnr / len(gt_video),
            'ssim': ssim / len(gt_video),
            'lpips_alex': lpips_alex / len(gt_video),
            'l1': l1 / len(gt_video),
            # 'lpips_vgg': lpips_vgg / len(gt_video),
            'fov_video_vdp': fov_score,
        }


def fov_video_vdp(
    gt_video: Float[torch.Tensor, 'time height width 3'],
    pred_video: Float[torch.Tensor, 'time height width 3'],
) -> tuple[float, Float[np.ndarray, 'time height width 3']]:
    """
    Compute the Foveated Video Quality Prediction (FVVP) metric.

    Args:
        gt_video: Ground truth video.
        pred_video: Predicted video.

    Returns:
        Foveated Video Quality Prediction (FVVP) score and heatmap.
    """

    metric = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')

    if gt_video.shape != pred_video.shape:
        print('Shapes of gt_video and pred_video are different: '
              f'{gt_video.shape} != {pred_video.shape}')
    assert gt_video.max() <= 1.0 and gt_video.min() >= 0.0, 'gt_video should be in [0, 1]'
    assert pred_video.max() <= 1.0 and pred_video.min() >= 0.0, 'pred_video should be in [0, 1]'
    # trim to the minimum length
    if gt_video.shape[0] < pred_video.shape[0]:
        trim_length = pred_video.shape[0] - gt_video.shape[0]
        left_trim = trim_length // 2
        right_trim = trim_length - left_trim
        pred_video = pred_video[left_trim:-right_trim]
    elif gt_video.shape[0] > pred_video.shape[0]:
        trim_length = gt_video.shape[0] - pred_video.shape[0]
        left_trim = trim_length // 2
        right_trim = trim_length - left_trim
        gt_video = gt_video[left_trim:-right_trim]
    gt_video = gt_video.unsqueeze(0)
    pred_video = pred_video.unsqueeze(0)

    # predict
    a, b = metric.predict(
        test_cont=pred_video, reference_cont=gt_video, dim_order='BFHWC', frames_per_second=30)
    heatmap = b['heatmap']
    heatmap = heatmap.squeeze(0).permute(1, 2, 3, 0).float().cpu().numpy()

    return a.item(), heatmap


# =============================================================================================== #
#                                    Evaluation Function                                          #
# =============================================================================================== #


def evaluate(
    gt_videos_directory: str,
    pred_videos_directory: str,
    output_directory: str | None = None,
    sequences: list[str] | list[int] | None = None,
    device: torch.device | str = 'cpu',
    cut_lower_n_pixels: int = 350,
) -> None:
    """
    Args:
        gt_videos_directory: Directory containing ground truth videos.
        pred_videos_directory: Directory containing predicted videos.
        output_directory: Directory to save evaluation results.
        sequences: List of sequences to evaluate.
        device: Device to use for computation.
        cut_lower_n_pixels: Number of pixels to cut from the lower part of the video. The
            segmentation masks don't always reach to the very bottom of the screen, which isn't
            ideal. The neck isn't really part of our evaluation anyways, so we can cut it off.
    """

    computer = _EvaluationComputer()
    computer.to(device)
    if output_directory is None:
        output_directory = pred_videos_directory

    # Evaluate each sequence
    results = {
        'psnr': [],
        'ssim': [],
        'lpips_alex': [],
        'l1': [],
        'fov_video_vdp': [],
        'sequence_length': [],
        'sequence_name': [],
    }

    # get sequences from the directory if not provided
    if sequences is None:
        gt_sequences = os.listdir(gt_videos_directory)
        pred_sequences = os.listdir(pred_videos_directory)
        sequences = list(set(gt_sequences) & set(pred_sequences))
        if len(sequences) == 0:
            raise ValueError('No common sequences found between the directories')

    for sequence in sequences:
        print()
        print(f'Evaluating {sequence}')
        if isinstance(sequence, int):
            sequence = f'sequence_{sequence}.mp4'
        print('Loading videos')
        gt_video = load_video(f'{gt_videos_directory}/{sequence}')
        gt_video = torch.from_numpy(gt_video)[:, :-cut_lower_n_pixels]
        pred_video = load_video(f'{pred_videos_directory}/{sequence}')
        pred_video = torch.from_numpy(pred_video)[:, :-cut_lower_n_pixels]
        print('Done!')
        results_sequence = computer(gt_video, pred_video, device)
        results['psnr'].append(results_sequence['psnr'])
        results['ssim'].append(results_sequence['ssim'])
        results['lpips_alex'].append(results_sequence['lpips_alex'])
        results['l1'].append(results_sequence['l1'])
        results['fov_video_vdp'].append(results_sequence['fov_video_vdp'])
        results['sequence_name'].append(sequence)
        results['sequence_length'].append(min(len(gt_video), len(pred_video)))

    # Compute average results
    results['psnr_avg'] = sum([
        score * weight for score, weight in zip(results['psnr'], results['sequence_length'])
    ]) / sum(results['sequence_length'])
    results['ssim_avg'] = sum([
        score * weight for score, weight in zip(results['ssim'], results['sequence_length'])
    ]) / sum(results['sequence_length'])
    results['lpips_alex_avg'] = sum([
        score * weight for score, weight in zip(results['lpips_alex'], results['sequence_length'])
    ]) / sum(results['sequence_length'])
    results['l1_avg'] = sum([
        score * weight for score, weight in zip(results['l1'], results['sequence_length'])
    ]) / sum(results['sequence_length'])
    results['fov_video_vdp_avg'] = sum([
        score * weight
        for score, weight in zip(results['fov_video_vdp'], results['sequence_length'])
    ]) / sum(results['sequence_length'])

    # Save results as YAML
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, 'evaluation_results.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(results, f)

    print(f'Average PSNR: {results["psnr_avg"]}')
    print(f'Average SSIM: {results["ssim_avg"]}')
    print(f'Average LPIPS (alex): {results["lpips_alex_avg"]}')
    print(f'Average L1: {results["l1_avg"]}')
    print(f'Average Foveated Video Quality Prediction: {results["fov_video_vdp_avg"]}')
    print(f'Results saved to {output_path}')


# =============================================================================================== #
#                                    Graphing Function                                            #
# =============================================================================================== #


def graph_evaluation(dir_path: str) -> None:
    """ Graph evaluation results. """

    path = os.path.join(dir_path, 'evaluation_results.yaml')
    with open(path, 'r') as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=results['sequence_name'], y=results['psnr'], name='PSNR'))
    fig.add_trace(go.Bar(x=results['sequence_name'], y=results['ssim'], name='SSIM'))
    fig.add_trace(go.Bar(x=results['sequence_name'], y=results['lpips_alex'], name='LPIPS (alex)'))
    fig.add_trace(go.Bar(x=results['sequence_name'], y=results['l1'], name='L1'))
    fig.add_trace(go.Bar(x=results['sequence_name'], y=results['fov_video_vdp'], name='FVVP'))
    fig.update_layout(barmode='group')
    fig.show()


def create_metrics_boxplot(data):
    """
    Create boxplots comparing the distributions of different quality metrics,
    with hover data showing sequence names
    """
    # Define metrics and their ranges
    metrics = {
        'PSNR': {
            'values': data['psnr'],
            'range': [25, 30]
        },
        'SSIM': {
            'values': data['ssim'],
            'range': [0.94, 0.97]
        },
        'L1 Loss': {
            'values': data['l1'],
            'range': [0.005, 0.015]
        },
        'LPIPS': {
            'values': data['lpips_alex'],
            'range': [0.08, 0.14]
        },
        'FOV VDP': {
            'values': data['fov_video_vdp'],
            'range': [6.5, 7.5]
        }
    }

    # Create subplot figure with 5 rows
    fig = make_subplots(rows=5, cols=1, subplot_titles=list(metrics.keys()), vertical_spacing=0.05)

    # Add each metric as a separate subplot
    for idx, (name, metric_data) in enumerate(metrics.items(), 1):
        fig.add_trace(
            go.Box(
                y=metric_data['values'],
                name=name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>" + "Value: %{y:.4f}<br>" + "Sequence: %{text}<br>"
                    + "<extra></extra>"  # This removes the secondary box
                ),
                text=data['sequence_name']  # Add sequence names to hover
            ),
            row=idx,
            col=1)

        # Update y-axis range for better visualization
        fig.update_yaxes(range=metric_data['range'], row=idx, col=1, title=name)

    # Update layout
    fig.update_layout(
        title='Distribution of Quality Metrics',
        height=1000,  # Increased height to accommodate all subplots
        showlegend=False,
        margin=dict(t=50),  # Adjust top margin
        hoverlabel=dict(
            bgcolor="white",  # white background
            font_size=14,  # larger font size
            font_family="Rockwell"  # custom font
        ))

    return fig


def create_sequence_comparison(data):
    """
    Create a line plot comparing metrics across sequences
    """
    fig = go.Figure()

    # Add traces for each metric
    fig.add_trace(
        go.Scatter(x=data['sequence_name'], y=data['psnr'], name='PSNR', mode='lines+markers'))

    fig.add_trace(
        go.Scatter(
            x=data['sequence_name'], y=data['ssim'], name='SSIM', mode='lines+markers',
            yaxis='y2'))

    # Update layout with dual y-axes
    fig.update_layout(
        title='PSNR and SSIM Across Sequences',
        xaxis_title='Sequence',
        yaxis_title='PSNR',
        yaxis2=dict(title='SSIM', overlaying='y', side='right'),
        showlegend=True,
        height=600)

    return fig


def create_correlation_heatmap(data):
    """
    Create a heatmap showing correlations between different metrics
    """
    # Calculate correlations between metrics
    metrics = ['psnr', 'ssim', 'l1', 'lpips_alex', 'fov_video_vdp']
    data['lpips_alex'] = [-x for x in data['lpips_alex']]
    data['l1'] = [-x for x in data['l1']]
    corr_matrix = np.corrcoef([data[m] for m in metrics])

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=['PSNR', 'SSIM', 'L1', 'LPIPS', 'FOV VDP'],
            y=['PSNR', 'SSIM', 'L1', 'LPIPS', 'FOV VDP'],
            colorscale='RdBu',
            zmid=0))

    fig.update_layout(title='Correlation Between Quality Metrics', height=600)

    return fig


def create_sequence_length_vs_quality(data):
    """
    Create a scatter plot comparing sequence length with quality metrics
    """
    fig = go.Figure()

    metrics = {'PSNR': data['psnr'], 'SSIM': data['ssim']}

    for name, values in metrics.items():
        fig.add_trace(
            go.Scatter(
                x=data['sequence_length'],
                y=values,
                mode='markers',
                name=name,
                text=data['sequence_name'],
                hovertemplate='<b>Sequence:</b> %{text}<br>' + '<b>Length:</b> %{x}<br>'
                + f'<b>{name}:</b> %{{y:.3f}}<br>'))

    fig.update_layout(
        title='Sequence Length vs. Quality Metrics',
        xaxis_title='Sequence Length (frames)',
        yaxis_title='Metric Value',
        height=600)

    return fig


# Function to prepare data from the provided format
def prepare_data(metrics_dict):
    """
    Convert the provided dictionary format into a pandas DataFrame
    """
    return pd.DataFrame({
        'sequence_name': metrics_dict['sequence_name'],
        'sequence_length': metrics_dict['sequence_length'],
        'psnr': metrics_dict['psnr'],
        'ssim': metrics_dict['ssim'],
        'l1': metrics_dict['l1'],
        'lpips_alex': metrics_dict['lpips_alex'],
        'fov_video_vdp': metrics_dict['fov_video_vdp']
    })


# Example usage
def plot_all_visualizations(dir_path: str):
    """
    Create and display all visualizations for the metrics data
    """
    path = os.path.join(dir_path, 'evaluation_results.yaml')
    metrics_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    data = prepare_data(metrics_dict)

    return {
        'boxplot': create_metrics_boxplot(data),
        'sequence_comparison': create_sequence_comparison(data),
        'correlation_heatmap': create_correlation_heatmap(data),
        'length_vs_quality': create_sequence_length_vs_quality(data)
    }


if __name__ == '__main__':

    gt_dir = 'tmp/gt/masked'
    pred_dir = 'tmp/pred/2dgs_full_res_500k_overnight_rigging_large_lpips/flame'
    # evaluate(
    #     gt_path=gt_dir,
    #     pred_path=pred_dir,
    #     sequences=[i for i in range(80, 102)],
    #     device='cuda',
    # )
    for key, value in plot_all_visualizations(pred_dir).items():
        print(key)
        value.show()
