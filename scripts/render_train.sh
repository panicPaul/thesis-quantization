
# Loop through sequences [3 ... 5], + [11 ... 84] and render the videos
for seq in {3..5}; do
    echo "Processing sequence $seq..."
    python src/thesis/render_video.py --sequence=$seq
    echo "--------------------------------------------"
    echo ""
done
for seq in {11..84}; do
    echo "Processing sequence $seq..."
    python src/thesis/render_video.py --sequence=$seq
    echo "--------------------------------------------"
    echo ""
done

echo "Done."
