
# Loop through sequences 80 to and including 101
for seq in {80..101}; do
    echo "Processing sequence $seq..."
    python src/thesis/render_video.py --sequence=$seq
    echo "--------------------------------------------"
    echo ""
done

echo "Done."
