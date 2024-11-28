#!/bin/bash

# Define the ablations array
ablations=(
    "0_no_flame_prior"
    "1_just_flame_prior"
    "2_with_per_gaussian"
    "3_with_color_mlp"
    "4_flame_inner_mouth"
    "5_open_mouth_oversampling"
    "6_revised_densification"
    "7_markov_chain_monte_carlo"
)

# Base path for checkpoints
base_path="tb_logs/dynamic_gaussian_splatting/ablations"
checkpoint_file="epoch=3-step=400000.ckpt"

# Create a log directory for outputs
log_dir="render_logs"
mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/render_log_${timestamp}.txt"

# Function to log messages
log_message() {
    echo "$1"
    echo "$1" >> "$log_file"
}

# Function to check if render was successful
check_render() {
    if [ $? -eq 0 ]; then
        log_message "✓ Render completed successfully"
    else
        log_message "⨯ Error during rendering"
    fi
}

# Main processing loop
for ablation in "${ablations[@]}"; do
    log_message "Processing ablation: $ablation"

    # Construct full checkpoint path
    ablation_path="${base_path}/${ablation}/version_0/checkpoints/${checkpoint_file}"

    # Check if checkpoint exists
    if [ ! -f "$ablation_path" ]; then
        log_message "Warning: Checkpoint not found at $ablation_path"
        continue
    }

    # Process each sequence
    for seq in {80..101}; do
        log_message "Processing sequence $seq..."

        # Run the render script
        python src/thesis/render_video.py \
            --sequence=$seq \
            --gaussian_splats_checkpoint="$ablation_path" 2>> "$log_file"

        check_render

        log_message "--------------------------------------------"
    done

    log_message "\nCompleted ablation: $ablation"
    log_message "============================================\n"
done

log_message "All rendering jobs completed!"

# Print summary statistics from log file
echo "Summary:"
echo "Total ablations processed: ${#ablations[@]}"
echo "Sequences per ablation: 22 (80-101)"
echo "Log file: $log_file"
