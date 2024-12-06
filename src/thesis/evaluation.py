""" Evaluation of two videos. """

import gzip
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

from thesis.constants import TEST_CAMS
from thesis.data_management import SequenceManager
from thesis.utils import assign_segmentation_class
from thesis.video_utils import load_video


def discretize_frames(
    frame: Float[torch.Tensor,
                 '... height width 3'],) -> Float[torch.Tensor, '... height width 3']:
    """discretization_function"""
    frame = frame * 255
    frame = frame.to(torch.uint8)
    frame = frame.to(torch.float32)
    frame = frame / 255
    return frame


class _EvaluationComputer(nn.Module):
    """ Compute evaluation metrics. """

    def __init__(self) -> None:
        super().__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
        # self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        self.reference_image = pyfvvdp.load_image_as_array

    def forward(
        self,
        gt_video: Float[torch.Tensor, 'height width 3'],
        pred_video: Float[torch.Tensor, 'height width 3'],
        loss_dict: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            gt_video: Ground truth video.
            pred_video: Predicted video.
            loss_dict: Dictionary to store loss values.

        Returns:
            Dictionary of evaluation metrics.
        """

        gt_video = gt_video.permute(2, 0, 1).unsqueeze(0)
        pred_video = pred_video.permute(2, 0, 1).unsqueeze(0)
        psnr = self.psnr(pred_video, gt_video)
        ssim = self.ssim(pred_video, gt_video)
        lpips_alex = self.lpips_alex(pred_video, gt_video)
        # lpips_vgg = self.lpips_vgg(pred_video, gt_video)
        l1 = torch.nn.functional.l1_loss(pred_video, gt_video)

        if loss_dict is None:
            loss_dict = {
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'lpips_alex': lpips_alex.item(),
                'l1': l1.item(),
                # 'lpips_vgg': lpips_vgg.item(),
            }
        else:
            loss_dict['psnr'] += psnr.item()
            loss_dict['ssim'] += ssim.item()
            loss_dict['lpips_alex'] += lpips_alex.item()
            loss_dict['l1'] += l1.item()
            # loss_dict['lpips_vgg'] += lpips_vgg.item()

        return loss_dict


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
        raise ValueError('Shapes of gt_video and pred_video are different')
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


def _load_gt_frame(sequence_manager, idx, idx_offset, cut_lower_n_pixels):
    """ Load ground truth frame. """
    frame = sequence_manager.images[idx + idx_offset, 0]
    frame = frame[:-cut_lower_n_pixels]
    alpha_mask = sequence_manager.alpha_maps[idx + idx_offset, 0]
    alpha_mask = alpha_mask[:-cut_lower_n_pixels]
    segmentation_mask = sequence_manager.segmentation_masks[idx + idx_offset, 0]
    segmentation_mask = segmentation_mask[:-cut_lower_n_pixels]
    segmentation_class = assign_segmentation_class(segmentation_mask.unsqueeze(0)).squeeze(0)
    background_mask = torch.where(segmentation_class == 0, 1, 0)
    jumper_mask = torch.where(segmentation_class == 2, 1, 0)
    alpha_mask = alpha_mask * (1-background_mask) * (1-jumper_mask)
    alpha_mask = alpha_mask.unsqueeze(-1).repeat(1, 1, 3)
    return frame, alpha_mask


def _evaluate_single_sequence(
    sequence: int,
    pred_data_dir: str,
    device: torch.device | str = 'cuda',
    cut_lower_n_pixels: int = 350,
    background: Float[torch.Tensor, "3"] = torch.ones(3) * 0.66,
) -> dict[str, float]:
    """
    Evaluate a single sequence.

    Args:
        sequence: Sequence number.
        pred_data_dir: Directory containing the predicted videos.
        device: Device to use for computation.
        cut_lower_n_pixels: Number of pixels to cut from the bottom of the video.
        background: Background color.
    """

    torch.cuda.empty_cache()
    print(f'Evaluating {sequence}')
    print('Loading videos')
    loss_computer = _EvaluationComputer()
    loss_computer.to(device)
    pred_path = os.path.join(pred_data_dir, f'sequence_{sequence}.npy.gz')
    f = gzip.GzipFile(pred_path, "r")
    pred_video = np.load(f)
    pred_video = torch.from_numpy(pred_video)[:, :-cut_lower_n_pixels]
    pred_video = pred_video.float() / 255
    n_frames = pred_video.shape[0]
    sm = SequenceManager(sequence, cameras=TEST_CAMS)
    idx_offset = (len(sm) - n_frames) // 2
    assert len(sm) == n_frames + 2*idx_offset, f'{len(sm)} != {n_frames + 2*idx_offset}'
    background = background[None, None, :].repeat(pred_video.shape[1], pred_video.shape[2], 1)
    background = background.cpu()

    # Compute evaluation metrics
    loss_dict = None
    for i in tqdm(range(n_frames), desc='Computing Metrics'):
        cur_gt_frame, cur_alpha_mask = _load_gt_frame(sm, i, idx_offset, cut_lower_n_pixels)
        cur_gt_frame = cur_gt_frame*cur_alpha_mask + background * (1-cur_alpha_mask)
        cur_gt_frame = cur_gt_frame.to(device)
        cur_pred_frame = pred_video[i].to(device)
        cur_gt_frame = discretize_frames(cur_gt_frame)
        cur_pred_frame = discretize_frames(cur_pred_frame)
        loss_dict = loss_computer.forward(cur_gt_frame, cur_pred_frame, loss_dict)
        # sanity check, that top left pixel are the same
        assert torch.allclose(cur_gt_frame[0, 0], cur_pred_frame[0, 0])
    loss_dict = {key: value / n_frames for key, value in loss_dict.items()}

    # Compute Foveated Video Quality Prediction
    gt_video = sm.images[:, 0]  # (time, height, width, 3)
    gt_video = gt_video[idx_offset:-idx_offset, :-cut_lower_n_pixels]
    fov_metric, heatmap = fov_video_vdp(gt_video=gt_video, pred_video=pred_video)
    loss_dict['fov_video_vdp'] = fov_metric

    print(f'PSNR: {loss_dict["psnr"]}')
    print(f'SSIM: {loss_dict["ssim"]}')
    print(f'LPIPS (alex): {loss_dict["lpips_alex"]}')
    # print(f'LPIPS (vgg): {loss_dict["lpips_vgg"]}')
    print(f'L1: {loss_dict["l1"]}')
    print(f'Foveated Video Quality Prediction: {fov_metric}')

    # clear cuda memory
    torch.cuda.empty_cache()
    return loss_dict, n_frames


def evaluate(
    pred_dir: str,
    sequences: list[int] = list(range(80, 102)),
    device: torch.device | str = 'cuda',
    cut_lower_n_pixels: int = 350,
    background: Float[torch.Tensor, "3"] = torch.ones(3) * 0.66,
):  #-> None:
    """
    Args:
        pred_dir: Directory containing the predicted videos.
        sequences: List of sequences to evaluate. If None, all sequences in the directory are
            evaluated.
        device: Device to use for computation.
        cut_lower_n_pixels: Number of pixels to cut from the bottom of the video.
    """

    computer = _EvaluationComputer()
    computer.to(device)
    _background = background.to(device)

    # Evaluate each sequence
    results = {
        'psnr': [],
        'ssim': [],
        'lpips_alex': [],
        # 'lpips_vgg': [],
        'l1': [],
        'fov_video_vdp': [],
        'sequence_length': [],
        'sequence_name': [],
    }
    pred_data_dir = os.path.join(pred_dir, 'raw_data')

    for sequence in sequences:

        loss_dict, n_frames = _evaluate_single_sequence(
            sequence=sequence,
            pred_data_dir=pred_data_dir,
            device=device,
            cut_lower_n_pixels=cut_lower_n_pixels,
            background=_background,
        )
        results['psnr'].append(loss_dict['psnr'])
        results['ssim'].append(loss_dict['ssim'])
        results['lpips_alex'].append(loss_dict['lpips_alex'])
        # results['lpips_vgg'].append(loss_dict['lpips_vgg'])
        results['l1'].append(loss_dict['l1'])
        results['fov_video_vdp'].append(loss_dict['fov_video_vdp'])
        results['sequence_name'].append(sequence)
        results['sequence_length'].append(n_frames)

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
    output_path = os.path.join(pred_dir, 'evaluation_results.yaml')
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
            'range': [25, 33]
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
    ablations = [
        '0_no_flame_prior', '1_just_flame_prior', "2_with_per_gaussian", '3_with_color_mlp',
        '4_with_inner_mouth', '5_open_mouth_oversampling', '6_revised_densification',
        '7_markov_chain_monte_carlo'
    ]
    cur_ablation = f'{ablations[7]}_lpips_0.01'
    pred_dir = f'tmp/pred/ablations/{cur_ablation}/flame'
    evaluate(
        pred_dir=pred_dir,
        sequences=list(range(80, 102)),
        device='cuda',
    )
    for key, value in plot_all_visualizations(pred_dir).items():
        print(key)
        value.show()
