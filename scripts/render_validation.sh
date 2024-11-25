
# Loop through sequences [6 ... 10], + [85 ... 101) and render the videos
for seq in {6..10}; do
    echo "Processing sequence $seq..."
    python src/thesis/render_video.py --sequence=$seq
    echo "--------------------------------------------"
    echo ""
done
for seq in {85..101}; do
    echo "Processing sequence $seq..."
    python src/thesis/render_video.py --sequence=$seq
    echo "--------------------------------------------"
    echo ""
done

echo "Done."
