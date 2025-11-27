#!/bin/bash
# Simple shell script to run RoboSat to COCO pipeline using Docker
# This is a simpler alternative that uses Docker directly

set -e  # Exit on error

# Configuration
OSM_FILE="${1:-data.osm.pbf}"
OUTPUT_DIR="${2:-./data/coco_dataset}"
ZOOM="${3:-19}"
FEATURE_TYPE="${4:-building}"
MAPBOX_TOKEN="${MAPBOX_API_KEY:-}"

echo "============================================================"
echo "RoboSat to COCO - Simple Pipeline (Docker)"
echo "============================================================"
echo ""
echo "OSM File: $OSM_FILE"
echo "Output: $OUTPUT_DIR"
echo "Zoom: $ZOOM"
echo "Feature Type: $FEATURE_TYPE"
echo ""

# Create directories
WORK_DIR="$OUTPUT_DIR/robosat_work"
mkdir -p "$WORK_DIR"/{geojson,tiles,images,masks}

# Step 1: Extract features
echo "[Step 1/5] Extracting $FEATURE_TYPE features from OSM..."
docker run -it --rm \
    -v "$(pwd)/$(dirname $OSM_FILE):/data" \
    -v "$(pwd)/$WORK_DIR/geojson:/output" \
    robosat:latest-cpu \
    extract --type "$FEATURE_TYPE" \
    "/data/$(basename $OSM_FILE)" \
    "/output/$FEATURE_TYPE.geojson"

# Step 2: Generate tiles
echo ""
echo "[Step 2/5] Generating tiles..."
docker run -it --rm \
    -v "$(pwd)/$WORK_DIR/geojson:/data" \
    -v "$(pwd)/$WORK_DIR/tiles:/output" \
    robosat:latest-cpu \
    cover --zoom "$ZOOM" \
    "/data/$FEATURE_TYPE.geojson" \
    "/output/tiles.csv"

# Step 3: Download images
echo ""
echo "[Step 3/5] Downloading satellite images..."
if [ -n "$MAPBOX_TOKEN" ]; then
    URL="https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.webp?access_token=$MAPBOX_TOKEN"
else
    URL="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    echo "  Using OpenStreetMap tiles (no token needed)"
fi

docker run -it --rm \
    -v "$(pwd)/$WORK_DIR/tiles:/data" \
    -v "$(pwd)/$WORK_DIR/images:/output" \
    robosat:latest-cpu \
    download "$URL" --ext png --rate 10 \
    "/data/tiles.csv" \
    "/output"

# Step 4: Create dataset config
echo ""
echo "[Step 4/5] Creating dataset config..."
cat > "$WORK_DIR/dataset.toml" <<EOF
[common]
classes = ["background", "foreground"]
colors = ["denim", "orange"]
EOF

# Step 5: Rasterize masks
echo ""
echo "[Step 5/5] Rasterizing features to masks..."
docker run -it --rm \
    -v "$(pwd)/$WORK_DIR/geojson:/data/geojson" \
    -v "$(pwd)/$WORK_DIR/tiles:/data/tiles" \
    -v "$(pwd)/$WORK_DIR/masks:/output" \
    -v "$(pwd)/$WORK_DIR/dataset.toml:/config/dataset.toml" \
    robosat:latest-cpu \
    rasterize \
    "/data/geojson/$FEATURE_TYPE.geojson" \
    "/data/tiles/tiles.csv" \
    "/output" \
    --dataset "/config/dataset.toml" \
    --zoom "$ZOOM" \
    --size 512

echo ""
echo "============================================================"
echo "✅ RoboSat pipeline completed!"
echo "============================================================"
echo ""
echo "Next step: Convert to COCO format"
echo "Run: python scripts/convert_robosat_slippy_to_coco.py --images $WORK_DIR/images --masks $WORK_DIR/masks --output $OUTPUT_DIR/coco_dataset"
echo ""

