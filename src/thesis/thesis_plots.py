""" Plotting functions for the thesis pdf """

import os
import yaml
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import os
import yaml
import pandas as pd
import numpy as np


def load_experiment_results(
    experiment_names: list[str],
    sequences: list[int],
    base_path: str = 'ablations/ablations_final',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load results from specified experiments and sequences and compute weighted averages.
    
    Args:
        experiment_names (list): List of experiment names to process
        sequences (list): List of sequence numbers/names to process
        base_path (str): Base directory containing experiment folders
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame with raw experiment results
            - DataFrame with weighted averages per experiment
    """
    results = []
    metrics = ['fov_video_vdp', 'l1', 'lpips_alex', 'psnr', 'ssim']

    for experiment in experiment_names:
        for sequence in sequences:
            # Construct path to YAML file
            yaml_path = os.path.join(base_path, experiment, 'results', f'sequence_{sequence}.yaml')

            # Check if file exists
            if not os.path.exists(yaml_path):
                print(f"Warning: File not found - {yaml_path}")
                continue

            # Load YAML data
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                # Add experiment and sequence info to data
                data['experiment'] = experiment
                data['sequence'] = sequence
                results.append(data)
            except Exception as e:
                print(f"Error loading {yaml_path}: {str(e)}")

    # Return empty DataFrames if no results were found
    if not results:
        print("No results were found for the specified experiments and sequences")
        return pd.DataFrame(), pd.DataFrame()

    # Create DataFrame with all results
    df_results = pd.DataFrame(results)

    # Compute weighted averages for each experiment
    weighted_averages = []

    for experiment in experiment_names:
        exp_data = df_results[df_results['experiment'] == experiment]

        if exp_data.empty:
            continue

        weighted_metrics = {}
        weighted_metrics['experiment'] = experiment
        weighted_metrics['total_frames'] = exp_data['num_frames'].sum()

        # Calculate weighted average for each metric
        for metric in metrics:
            if metric in exp_data.columns:
                weighted_avg = np.average(exp_data[metric], weights=exp_data['num_frames'])
                weighted_metrics[f'{metric}_weighted_avg'] = weighted_avg

        weighted_averages.append(weighted_metrics)

    df_weighted = pd.DataFrame(weighted_averages)

    return df_results, df_weighted


def get_experiment_display_name(
    experiment: str,
    name_mapping: dict[str, str] = {
        'gaussian_avatars': 'GaussianAvatars',
        'with_color_mlp_2dgs': 'w/ Color MLP & 2DGS',
        'with_color_mlp': 'w/ Color MLP',
        'just_flame_prior': 'FLAME Prior',
        'just_flame_prior_inner_mouth': 'FLAME Prior++',
        'with_per_gaussian': 'w/ Per-Gaussian Adjustment',
    },
    remove_model_type: bool = True,
) -> str:
    """
    Get the display name for an experiment using the mapping.
    
    Args:
        experiment (str): Raw experiment name
        name_mapping (dict): Mapping of raw names to display names
    
    Returns:
        str: Display name for the experiment
    """
    parts = experiment.split('/')
    base_name = parts[0]
    model_type = parts[1] if len(parts) > 1 else None

    display_name = name_mapping.get(base_name, base_name)
    if model_type and not remove_model_type:
        if model_type.lower() == 'audio':
            display_name += ' (Audio driven)'
        elif model_type.lower() == 'flame':
            display_name += ' (FLAME driven)'
        else:
            display_name += f' ({model_type})'

    return display_name


def write_latex_table(data_frame: pd.DataFrame) -> None:
    """
    Write the DataFrame to a LaTeX table with custom formatting.
    - Adds ↑/↓ arrows to indicate if higher/lower values are better
    - Colors the top 3 values for each metric (green/yellow/orange)
    - Formats numbers with appropriate precision
    
    Args:
        data_frame (pd.DataFrame): DataFrame containing experiment results
    """
    # Define metrics where higher values are better
    higher_better = {'psnr', 'ssim', 'fov_video_vdp'}

    # Define metric name mappings
    metric_names = {
        'fov_video_vdp_weighted_avg': 'VDP',
        'l1_weighted_avg': 'L1 Loss',
        'lpips_alex_weighted_avg': 'LPIPS',
        'psnr_weighted_avg': 'PSNR',
        'ssim_weighted_avg': 'SSIM'
    }

    # Remove total_frames if present
    if 'total_frames' in data_frame.columns:
        data_frame = data_frame.drop(columns=['total_frames'])

    # Define color commands for top 3 results
    latex_colors = [
        '\\cellcolor{green!25}',  # First place
        '\\cellcolor{yellow!25}',  # Second place
        '\\cellcolor{orange!25}'  # Third place
    ]

    # Start LaTeX table
    latex_lines = [
        '\\begin{table}[htbp]', '\\centering', '\\begin{tabular}{l' + 'r' *
        (len(data_frame.columns) - 1) + '}', '\\toprule'
    ]

    # Generate header row
    headers = []
    for col in data_frame.columns:
        if col == 'experiment':
            headers.append('Method')
        else:
            # Add arrows to metric names
            is_metric = any(m in col for m in ['psnr', 'ssim', 'l1', 'lpips', 'fov_video_vdp'])
            if is_metric:
                # Get friendly metric name
                metric_name = metric_names.get(col, col)
                # Add arrow based on whether higher is better
                arrow = '\\ensuremath{\\uparrow}' if any(
                    hb in col for hb in higher_better) else '\\ensuremath{\\downarrow}'
                headers.append(f'{metric_name} {arrow}')
            else:
                headers.append(col)

    latex_lines.append(' & '.join(headers) + ' \\\\')
    latex_lines.append('\\midrule')

    # Process each metric column to find top 3 values
    top3_indices = {}
    for col in data_frame.columns:
        if col != 'experiment' and any(
                m in col for m in ['psnr', 'ssim', 'l1', 'lpips', 'fov_video_vdp']):
            values = data_frame[col].values
            if any(hb in col for hb in higher_better):
                # For metrics where higher is better
                top3_indices[col] = (-values).argsort()[:3]
            else:
                # For metrics where lower is better
                top3_indices[col] = values.argsort()[:3]

    # Generate data rows
    for idx, row in data_frame.iterrows():
        cells = []
        for col_idx, (col, value) in enumerate(row.items()):
            if col == 'experiment':
                # Escape special characters in experiment names
                # cells.append(value.replace('_', '\\_'))
                # use the renamed experiment name
                name = get_experiment_display_name(value)
                # escape & and _ in the name
                name = name.replace('_', '\\_')
                name = name.replace('&', '\\&')
                cells.append(name)

            else:
                # Format numbers with appropriate precision
                if isinstance(value, (int, float)):
                    formatted_value = f'{value:.3f}'
                else:
                    formatted_value = str(value)

                # Add color formatting if this is a top 3 value
                if col in top3_indices and idx in top3_indices[col]:
                    rank = list(top3_indices[col]).index(idx)
                    formatted_value = f'{latex_colors[rank]}{formatted_value}'

                cells.append(formatted_value)

        latex_lines.append(' & '.join(cells) + ' \\\\')

    # Close LaTeX table with properly escaped math mode
    latex_lines.extend([
        '\\bottomrule', '\\end{tabular}',
        '\\caption{Quantitative evaluation results. Arrows (\\ensuremath{\\uparrow}/\\ensuremath{\\downarrow}) indicate whether higher/lower values are better. ',
        'Top three results for each metric are highlighted in \\textcolor{green!75}{green}, ',
        '\\textcolor{yellow!75}{yellow}, and \\textcolor{orange!75}{orange}.}',
        '\\label{tab:quantitative_results}', '\\end{table}'
    ])

    # Write to file
    with open('plots/quantitative_results.tex', 'w') as f:
        f.write('\n'.join(latex_lines))


def plot_metric(
    data_frame: pd.DataFrame,
    weighted_average_df: pd.DataFrame,
    metric: Literal['l1', 'lpips_alex', 'psnr', 'ssim', 'fov_video_vdp'],
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    style: str = 'default',
    save_dir: str = 'plots',
    save_plot: bool = False,
) -> None:
    """
    Plot the specified metric for each experiment and sequence using matplotlib.
    
    Args:
        data_frame (pd.DataFrame): DataFrame with experiment, sequence, and metric data
        weighted_average_df (pd.DataFrame): DataFrame with weighted averages per experiment
        metric (str): Metric to plot
        title (Optional[str]): Custom title for the plot
        figsize (tuple[int, int]): Figure size (width, height)
        style (str): Matplotlib style to use (default, classic, bmh, etc.)
        save_path (Optional[str]): Path to save the plot (if None, display only)
    """
    # Try to set style, fall back to 'default' if invalid
    try:
        plt.style.use(style)
    except Exception as e:
        print(f"Warning: Style '{style}' not found, using 'default' style")
        plt.style.use('default')

    metric_names = {
        'l1': 'L1 Loss',
        'lpips_alex': 'LPIPS',
        'psnr': 'PSNR',
        'ssim': 'SSIM',
        'fov_video_vdp': 'VDP'
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Color cycle for better visibility
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_frame['experiment'].unique())))

    # Plot data for each experiment with improved styling
    for idx, experiment in enumerate(sorted(data_frame['experiment'].unique())):
        df = data_frame[data_frame['experiment'] == experiment]
        display_name = get_experiment_display_name(experiment)

        ax.plot(
            df['sequence'],
            df[metric],
            marker='o',
            label=display_name,
            color=colors[idx],
            linewidth=2,
            markersize=8,
            alpha=0.8)
        # Plot weighted average as a dotted line in the same color
        if experiment in weighted_average_df['experiment'].values:
            weighted_avg = weighted_average_df.loc[weighted_average_df['experiment'] == experiment,
                                                   f'{metric}_weighted_avg'].values[0]
            ax.plot(
                df['sequence'], [weighted_avg] * len(df['sequence']),
                linestyle=':',
                color=colors[idx],
                linewidth=2,
                alpha=0.8)

    # Customize axes
    ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_names[metric], fontsize=12, fontweight='bold')

    # Set title
    plot_title = title if title else f'{metric_names[metric]} across Test Sequences'
    # ax.set_title(plot_title, fontsize=14, pad=20)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Customize legend
    ax.legend(title='Experiments', title_fontsize=12, fontsize=10, loc='upper left')

    # Set integer ticks for sequence numbers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or display the plot
    if save_plot:
        # save as pdf
        save_path = os.path.join(save_dir, f'{metric}_plot.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()
