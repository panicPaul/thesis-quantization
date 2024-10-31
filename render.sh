# Renders a video based on input

#!/bin/bash

# Default values
DEFAULT_AUDIO_PATH="tmp/audio_recording_cleaned_s3.ogg"
DEFAULT_GAUSSIAN_PATH="tb_logs/video/direct_pred/version_2/checkpoints/epoch=38-step=99000.ckpt"
DEFAULT_OUTPUT_PATH="tmp/output.mp4"
DEFAULT_AUDIO_TO_FLAME_PATH="tb_logs/audio2flame/my_model/version_6/checkpoints/epoch=29-step=10650.ckpt"
DEFAULT_LOAD_SEQUENCE="3"
DEFAULT_PREDICT_FLAME=false

# Help function to display usage information
show_help() {
    echo "Usage: ./run_render.sh [options]"
    echo
    echo "Render a video from audio with the following options:"
    echo "  -a,  --audio_path PATH          Path to the audio file"
    echo "                                  (default: $DEFAULT_AUDIO_PATH)"
    echo "  -gs, --gaussian_splats PATH     Path to the Gaussian splatting model checkpoint"
    echo "                                  (default: $DEFAULT_GAUSSIAN_PATH)"
    echo "  -o,  --output_path PATH         Path to save the output video"
    echo "                                  (default: $DEFAULT_OUTPUT_PATH)"
    echo "  -af, --audio_to_flame PATH      Path to the audio-to-flame model checkpoint"
    echo "                                  (default: $DEFAULT_AUDIO_TO_FLAME_PATH)"
    echo "  -s,  --load_sequence NUMBER     Load flame parameters from a sequence"
    echo "                                  (default: none)"
    echo "  -gt, --load_ground_truth        Load ground truth flame parameters instead"
    echo "                                  of predicting them from audio"
    echo "                                  (default: false)"
    echo "  -h,  --help                     Display this help message"
    echo
    echo "Example:"
    echo "  ./run_render.sh -a custom_input.wav -gs model.pth -o custom_output.mp4 -gt"
    echo
    echo "Using defaults:"
    echo "  ./run_render.sh"
}

# Initialize variables with default values
AUDIO_PATH="$DEFAULT_AUDIO_PATH"
GAUSSIAN_PATH="$DEFAULT_GAUSSIAN_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
AUDIO_TO_FLAME_PATH="$DEFAULT_AUDIO_TO_FLAME_PATH"
LOAD_SEQUENCE="$DEFAULT_LOAD_SEQUENCE"
LOAD_GROUND_TRUTH=$DEFAULT_LOAD_GROUND_TRUTH

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--audio_path)
            AUDIO_PATH="$2"
            shift 2
            ;;
        -gs|--gaussian_splats)
            GAUSSIAN_PATH="$2"
            shift 2
            ;;
        -o|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -af|--audio_to_flame)
            AUDIO_TO_FLAME_PATH="$2"
            shift 2
            ;;
        -s|--load_sequence)
            LOAD_SEQUENCE="$2"
            shift 2
            ;;
        -gt|--load_ground_truth)
            LOAD_GROUND_TRUTH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Verify file existence for input files
if [[ ! -f "$AUDIO_PATH" ]]; then
    echo "Error: Audio file not found: $AUDIO_PATH"
    exit 1
fi

if [[ ! -f "$GAUSSIAN_PATH" ]]; then
    echo "Error: Gaussian model checkpoint not found: $GAUSSIAN_PATH"
    exit 1
fi

if [[ -n "$AUDIO_TO_FLAME_PATH" ]] && [[ ! -f "$AUDIO_TO_FLAME_PATH" ]]; then
    echo "Error: FLAME model checkpoint not found: $AUDIO_TO_FLAME_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Construct the Python command
COMMAND="python3 src/thesis/render_video.py \
--audio_path \"$AUDIO_PATH\" \
--gaussian_splats_checkpoint_path \"$GAUSSIAN_PATH\" \
--output_path \"$OUTPUT_PATH\""

# Add optional arguments if provided
if [[ -n "$FLAME_PATH" ]]; then
    COMMAND+=" --audio_to_flame_checkpoint_path \"$FLAME_PATH\""
fi

if [[ -n "$LOAD_SEQUENCE" ]]; then
    COMMAND+=" --load_flame_from_sequence $LOAD_SEQUENCE"
fi

if [[ "$LOAD_GROUND_TRUTH" = true ]]; then
    COMMAND+=" --load_ground_truth"
fi

# Execute the command
echo "Running with configuration:"
echo "  Audio path: $AUDIO_PATH"
echo "  Gaussian model: $GAUSSIAN_PATH"
echo "  Output path: $OUTPUT_PATH"
echo "  Audio to FLAME checkpoint: $AUDIO_TO_FLAME_PATH"
echo "  Load sequence: $LOAD_SEQUENCE"
echo "  Load ground truth: $LOAD_GROUND_TRUTH"
echo
echo "Executing: $COMMAND"
echo
eval "$COMMAND"
