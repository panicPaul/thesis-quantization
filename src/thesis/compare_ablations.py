import os
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from glob import glob


def load_experiment_results(base_path, experiment_names, sequences):
    """
    Load results from specified experiments and sequences.
    
    Args:
        base_path (str): Base directory containing experiment folders
        experiment_names (list): List of experiment names to process
        sequences (list): List of sequence numbers/names to process
    
    Returns:
        pandas.DataFrame: DataFrame with experiment, sequence, and metric data
    """
    results = []

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
                data['experiment'] = experiment  #.split('/')[0]  # to remove flame / audio
                data['sequence'] = sequence
                results.append(data)
            except Exception as e:
                print(f"Error loading {yaml_path}: {str(e)}")

    # Return empty DataFrame if no results were found
    if not results:
        print("No results were found for the specified experiments and sequences")
        return pd.DataFrame()

    return pd.DataFrame(results)


def calculate_weighted_means(df, metrics):
    """
    Calculate weighted means for each metric based on num_frames.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the results
        metrics (list): List of metrics to calculate weighted means for
    
    Returns:
        pandas.DataFrame: DataFrame with weighted means for each experiment
    """
    weighted_means = []

    for experiment in df['experiment'].unique():
        exp_data = df[df['experiment'] == experiment]
        exp_means = {'experiment': experiment}

        for metric in metrics:
            weighted_mean = (exp_data[metric]
                             * exp_data['num_frames']).sum() / exp_data['num_frames'].sum()
            exp_means[f'{metric}_weighted_mean'] = weighted_mean

        weighted_means.append(exp_means)

    return pd.DataFrame(weighted_means)


def create_visualizations(df, save_visualization: bool = False) -> None:
    """
    Create various plotly visualizations for the experiment results.
    """
    # Define metrics to plot (excluding num_frames)
    metrics = ['psnr', 'ssim', 'l1', 'lpips_alex', 'fov_video_vdp']

    # Calculate weighted means
    weighted_means_df = calculate_weighted_means(df, metrics)

    # Create subplots
    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        subplot_titles=metrics,
    )

    # Color for each experiment
    experiments = df['experiment'].unique()
    colors = [f'hsl({i*360/len(experiments)},70%,50%)' for i in range(len(experiments))]

    # Create box plots for each metric
    for i, metric in enumerate(metrics, 1):
        for exp, color in zip(experiments, colors):
            exp_data = df[df['experiment'] == exp][metric]
            weighted_mean = weighted_means_df[weighted_means_df['experiment'] ==
                                              exp][f'{metric}_weighted_mean'].iloc[0]

            # Add box plot
            fig.add_trace(
                go.Box(
                    y=exp_data,
                    name=exp,
                    marker_color=color,
                    showlegend=i == 1  # Only show legend for first subplot
                ),
                row=i,
                col=1)

            # Add weighted mean as a line
            fig.add_trace(
                go.Scatter(
                    x=[exp],
                    y=[weighted_mean],
                    name=f'{exp} (weighted mean)',
                    mode='markers',
                    marker=dict(
                        symbol='star', size=12, color=color, line=dict(color='black', width=1)),
                    showlegend=i == 1),
                row=i,
                col=1)

    # Update layout
    fig.update_layout(
        title='Experiment Results Comparison (★ indicates weighted mean)',
        boxmode='group',
        height=250 * len(metrics),
        width=1000,
        showlegend=True,
        template='plotly_white')

    # Create sequence-wise line plots with weighted means
    for metric in metrics:
        metric_fig = go.Figure()

        for exp, color in zip(experiments, colors):
            exp_data = df[df['experiment'] == exp].sort_values('sequence')
            weighted_mean = weighted_means_df[weighted_means_df['experiment'] ==
                                              exp][f'{metric}_weighted_mean'].iloc[0]

            # Add sequence data
            metric_fig.add_trace(
                go.Scatter(
                    x=exp_data['sequence'],
                    y=exp_data[metric],
                    name=exp,
                    mode='lines+markers',
                    marker_color=color))

            # Add weighted mean as a horizontal line
            metric_fig.add_trace(
                go.Scatter(
                    x=[exp_data['sequence'].min(), exp_data['sequence'].max()],
                    y=[weighted_mean, weighted_mean],
                    name=f'{exp} (weighted mean)',
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    showlegend=True))

        metric_fig.update_layout(
            title=f'{metric} across Sequences (dashed lines show weighted means)',
            xaxis_title='Sequence',
            yaxis_title=metric,
            template='plotly_white',
            height=600,
            width=1000)

        # Save the figure
        if save_visualization:
            metric_fig.write_html(f'plots/{metric}_sequence_plot.html')
        else:
            metric_fig.show()

    # Save the main figure
    if save_visualization:
        fig.write_html('plots/experiment_comparison.html')
    else:
        fig.show()


def main():
    # Base path where experiment directories are located
    base_path = 'ablations/ablations_final'
    experiment_names = [
        'gaussian_avatars',  # baseline
        # 'no_flame_prior', # 0
        # 'just_flame_prior',  # 1
        'just_flame_prior_inner_mouth',  # 2
        'with_per_gaussian',  # 3
        'with_color_mlp',  # 4
        'with_color_mlp_2dgs',  # 5
        # 'oversample',  # 6
    ]
    type = 'audio'
    experiment_names_audio = [f'{name}/audio' for name in experiment_names]
    # experiment_names_flame = [f'{name}/flame' for name in experiment_names]
    experiment_names = experiment_names_audio  # + experiment_names_flame
    sequences = list(range(80, 102))

    # Load all results
    df = load_experiment_results(base_path, experiment_names, sequences)

    # Create visualizations
    create_visualizations(df)

    # Print summary statistics with weighted means
    print("\nSummary Statistics:")
    metrics = ['psnr', 'ssim', 'l1', 'lpips_alex', 'fov_video_vdp']

    weighted_means_df = calculate_weighted_means(df, metrics)

    for metric in metrics:
        print(f"\n{metric.upper()} Statistics by Experiment:")
        print(df.groupby('experiment')[metric].describe())
        print(f"\n{metric.upper()} Weighted Means by Experiment:")
        print(weighted_means_df[['experiment', f'{metric}_weighted_mean']])


if __name__ == "__main__":
    main()
