# RoboSat to COCO Scripts

Quick reference for using RoboSat to prepare COCO format datasets.

## Quick Start

```bash
# 1. Download OSM file (or use existing file in data/Essex/)
wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf -O data/essex-latest.osm.pbf

# 2. Run full pipeline
# If file is in data/Essex/:
# Option 1: Set environment variable (recommended)
export MAPBOX_API_KEY="your_token_here"  # Linux/Mac
# $env:MAPBOX_API_KEY="your_token_here"  # Windows PowerShell
python scripts/robosat_to_coco.py --osm-file data/Essex/essex-latest.osm.pbf --output ./data/coco_dataset --zoom 19

# Option 2: Pass token directly
python scripts/robosat_to_coco.py --osm-file data/Essex/england.osm.pbf --output ./data/coco_dataset/england --zoom 18 --test-tiles 20 --feature-types building road parking --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A

```

## Available Scripts

1. **robosat_to_coco.py** - Full pipeline (extract → cover → download → rasterize → COCO)
2. **convert_robosat_slippy_to_coco.py** - Convert existing slippy map to COCO
3. **test_robosat_coco.py** - Quick test script
4. **robosat_to_coco_simple.sh** - Shell script using Docker (Linux/Mac)
5. **robosat_to_coco_simple.ps1** - PowerShell script using Docker (Windows)

## Workflow

```
OSM File (.osm.pbf)
    ↓
[rs extract] → GeoJSON features
    ↓
[rs cover] → Tiles CSV
    ↓
[rs download] → Satellite images (slippy map: z/x/y.png)
    ↓
[rs rasterize] → Masks (slippy map: z/x/y.png)
    ↓
[convert_robosat_slippy_to_coco.py] → COCO format
```

## Examples

### Extract buildings only
```bash
python scripts/robosat_to_coco.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types building
```

### Extract multiple feature types
```bash
python scripts/robosat_to_coco.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types building road parking
```

### Use Mapbox satellite imagery
```bash
# Option 1: Set environment variable (automatically used)
export MAPBOX_API_KEY="your_token_here"  # Linux/Mac
# $env:MAPBOX_API_KEY="your_token_here"  # Windows PowerShell
python scripts/robosat_to_coco.py \
    --osm-file data.osm.pbf \
    --output ./output

# Option 2: Pass token directly
python scripts/robosat_to_coco.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --mapbox-token "your_token_here"
```

### Convert existing slippy map output
```bash
python scripts/convert_robosat_slippy_to_coco.py \
    --images ./images \
    --masks ./masks \
    --output ./coco_dataset
```

## Docker Alternative

If robosat is not installed, use Docker:

```bash
# Linux/Mac
bash scripts/robosat_to_coco_simple.sh data.osm.pbf ./output 19 building

# Windows PowerShell
.\scripts\robosat_to_coco_simple.ps1 -OsmFile data.osm.pbf -OutputDir ./output -Zoom 19
```

Then convert to COCO:
```bash
python scripts/convert_robosat_slippy_to_coco.py \
    --images ./output/robosat_work/images \
    --masks ./output/robosat_work/masks \
    --output ./output/coco_dataset
```

## See Also

- Full documentation: `docs/ROBOSAT_TO_COCO.md`
- RoboSat GitHub: https://github.com/mapbox/robosat

