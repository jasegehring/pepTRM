#!/bin/bash
# Monitor Nine-Species dataset download and setup

DATA_DIR="data/nine_species"
ZIP_FILE="$DATA_DIR/nine-species-balanced.zip"

echo "Monitoring download progress..."
echo "File: $ZIP_FILE"
echo ""

# Wait for download to complete
while [ ! -f "$ZIP_FILE" ] || pgrep -f "wget.*nine-species-balanced" > /dev/null; do
    if [ -f "$ZIP_FILE" ]; then
        SIZE=$(du -h "$ZIP_FILE" 2>/dev/null | cut -f1)
        echo -ne "\rDownloading... Current size: $SIZE   "
    fi
    sleep 10
done

echo ""
echo "✓ Download complete!"
echo ""

# Check file size
FINAL_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "Final file size: $FINAL_SIZE"
echo ""

# Extract
echo "Extracting archive..."
cd "$DATA_DIR" || exit 1
unzip -q nine-species-balanced.zip

if [ $? -eq 0 ]; then
    echo "✓ Extraction complete!"
    echo ""

    # Run verification
    echo "Running verification..."
    cd ../.. || exit 1
    python scripts/setup_nine_species.py --balanced --verify
else
    echo "❌ Extraction failed!"
    exit 1
fi
