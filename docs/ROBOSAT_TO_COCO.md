# RoboSat to COCO Dataset Preparation

This guide explains how to use RoboSat tools to prepare a COCO format dataset for satellite images.

## Overview

The workflow consists of these steps:

1. **Extract** (`rs extract`) - Download OSM data and generate GeoJSON features
2. **Cover** (`rs cover`) - Generate tiles covering the GeoJSON features
3. **Download** (`rs download`) - Download satellite images for those tiles
4. **Rasterize** (`rs rasterize`) - Rasterize GeoJSON features to label masks
5. **Convert to COCO** - Convert slippy map format to COCO format

## Prerequisites

### Option 1: Install RoboSat

```bash
cd 3rdparty/robosat
pip install -e .
```

### Option 2: Use Docker (Recommended)

**Option A: Build locally (recommended for this project)**
```bash
cd 3rdparty/robosat
docker build -f docker/Dockerfile.cpu -t robosat:latest-cpu .
```

**Option B: Pull from Docker Hub**
```bash
docker pull mapbox/robosat:latest-cpu
```

**Note:** The scripts in this project use the locally built `robosat:latest-cpu` image by default.

## Quick Start

### Method 1: Full Python Pipeline

```bash
# Download OSM file first (example for Essex, UK)
wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf -O data/essex-latest.osm.pbf

# Run full pipeline
python scripts/robosat_to_coco.py \
    --osm-file data/essex-latest.osm.pbf \
    --output ./data/coco_dataset \
    --zoom 19 \
    --feature-types building \
    --mapbox-token YOUR_MAPBOX_TOKEN  # Optional, uses OSM tiles if not provided
```

### Method 2: Shell Script + Python Conversion

```bash
# Step 1: Run RoboSat pipeline using Docker
bash scripts/robosat_to_coco_simple.sh \
    data/essex-latest.osm.pbf \
    ./data/coco_dataset \
    19 \
    building

# Step 2: Convert to COCO format
python scripts/convert_robosat_slippy_to_coco.py \
    --images ./data/coco_dataset/robosat_work/images \
    --masks ./data/coco_dataset/robosat_work/masks \
    --output ./data/coco_dataset/coco_dataset
```

### Method 3: Manual Steps

If you want more control, you can run each step manually:

```bash
# 1. Extract features
docker run -it --rm \
    -v $PWD/data:/data \
    robosat:latest-cpu \
    extract --type building /data/essex-latest.osm.pbf /data/building.geojson

# 2. Generate tiles
docker run -it --rm \
    -v $PWD/data:/data \
    robosat:latest-cpu \
    cover --zoom 19 /data/building.geojson /data/tiles.csv

# 3. Download images
docker run -it --rm \
    -v $PWD/data:/data \
    robosat:latest-cpu \
    download "https://tile.openstreetmap.org/{z}/{x}/{y}.png" \
    --ext png --rate 10 \
    /data/tiles.csv \
    /data/images

# 4. Create dataset config
cat > data/dataset.toml <<EOF
[common]
classes = ["background", "foreground"]
colors = ["denim", "orange"]
EOF

# 5. Rasterize masks
docker run -it --rm \
    -v $PWD/data:/data \
    robosat:latest-cpu \
    rasterize \
    /data/building.geojson \
    /data/tiles.csv \
    /data/masks \
    --dataset /data/dataset.toml \
    --zoom 19 \
    --size 512

# 6. Convert to COCO
python scripts/convert_robosat_slippy_to_coco.py \
    --images ./data/images \
    --masks ./data/masks \
    --output ./data/coco_dataset
```

## Scripts

### `robosat_to_coco.py`

Full pipeline script that runs all steps automatically.

**Arguments:**
- `--osm-file`: Path to .osm.pbf file (required)
- `--output`: Output directory (required)
- `--zoom`: Zoom level for tiles (default: 19)
- `--tile-size`: Tile size in pixels (default: 512)
- `--feature-types`: Feature types to extract: building, road, parking (default: building)
- `--mapbox-token`: Mapbox API token (optional)
- `--url-template`: Custom URL template for tile download
- `--skip-download`: Skip image download step
- `--skip-rasterize`: Skip mask rasterization step
- `--no-split`: Don't split into train/val/test

**Example:**
```bash
python scripts/robosat_to_coco.py \
    --osm-file data/essex-latest.osm.pbf \
    --output ./data/coco_dataset \
    --zoom 19 \
    --feature-types building road
```

### `convert_robosat_slippy_to_coco.py`

Converts slippy map format (z/x/y.png) to COCO format.

**Arguments:**
- `--images`: Directory with images in slippy map format (required)
- `--masks`: Directory with masks in slippy map format (required)
- `--output`: Output directory for COCO dataset (required)
- `--no-split`: Don't split into train/val/test
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.2)
- `--test-ratio`: Test set ratio (default: 0.1)

**Example:**
```bash
python scripts/convert_robosat_slippy_to_coco.py \
    --images ./data/images \
    --masks ./data/masks \
    --output ./data/coco_dataset
```

### `test_robosat_coco.py`

Quick test script to verify the pipeline works.

**Usage:**
```bash
export TEST_OSM_FILE=/path/to/test.osm.pbf
python scripts/test_robosat_coco.py
```

## Output Structure

After running the pipeline, you'll get:

```
data/coco_dataset/
├── robosat_work/          # Intermediate files
│   ├── geojson/           # GeoJSON features
│   ├── tiles/             # Tile CSV files
│   ├── images/            # Images in slippy map format (z/x/y.png)
│   └── masks/             # Masks in slippy map format (z/x/y.png)
└── coco_dataset/          # Final COCO dataset
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    ├── test/
    │   ├── images/
    │   └── masks/
    ├── annotations_train.json
    ├── annotations_val.json
    └── annotations_test.json
```

## COCO Format

The output follows the COCO format specification:

- **Images**: Flat PNG files with descriptive names
- **Annotations**: JSON files with COCO format annotations
- **Categories**: 
  - Building (id: 1)
  - Road (id: 2)
  - Landuse (id: 3)
  - Water (id: 4)
  - Tree (id: 5)
  - Grass (id: 6)
  - Car (id: 7)

## Tips

1. **Zoom Level**: Higher zoom (19-20) gives more detail but more tiles. Lower zoom (16-18) is faster but less detail.

2. **Feature Types**: You can extract multiple types:
   ```bash
   --feature-types building road parking
   ```

3. **Mapbox Token**: Get a free token at https://account.mapbox.com/access-tokens/
   - Free tier: 50,000 requests/month
   - Alternative: Use OpenStreetMap tiles (no token needed, but slower)

4. **Large Areas**: For large areas, consider:
   - Using lower zoom levels
   - Processing in smaller chunks
   - Using `--skip-download` and `--skip-rasterize` to resume from intermediate steps

5. **Testing**: Use a small OSM file first to test the pipeline:
   ```bash
   wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf
   ```

## Troubleshooting

### Docker Issues

If Docker commands fail:
- Make sure Docker is running
- Check volume mount paths (use absolute paths on Windows)
- Try with `-it --rm` flags

### Missing Tiles

If some tiles are missing:
- Check internet connection
- Verify tile server URL
- Check rate limiting (reduce `--rate` parameter)

### Memory Issues

For large datasets:
- Process in smaller batches
- Use lower zoom levels
- Increase Docker memory limit

## References

- [RoboSat GitHub](https://github.com/mapbox/robosat)
- [COCO Format Specification](https://cocodataset.org/#format-data)
- [OpenStreetMap Downloads](https://download.geofabrik.de/)

