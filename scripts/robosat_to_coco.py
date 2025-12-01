#!/usr/bin/env python
"""
RoboSat to COCO Dataset Preparation Script

This script uses robosat tools to:
1. Extract GeoJSON features from OSM data (rs extract)
2. Generate tiles covering the features (rs cover)
3. Download satellite images (rs download)
4. Rasterize features to masks (rs rasterize)
5. Convert to COCO format

Usage:
    python scripts/robosat_to_coco.py --osm-file data.osm.pbf --output ./data/coco_dataset --zoom 19
"""

import os
import sys
import argparse
import json
import csv
import shutil
import subprocess
import collections
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles

# Add robosat to path
robosat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(robosat_dir))

# Add geoseg src to path (for COCO exporter)
geoseg_dir = robosat_dir.parent.parent
sys.path.insert(0, str(geoseg_dir))

try:
    from robosat.tiles import tiles_from_csv, tiles_from_slippy_map
    from robosat.tools.extract import main as extract_main
    from robosat.tools.cover import main as cover_main
    from robosat.tools.download import main as download_main
    from robosat.tools.rasterize import main as rasterize_main
    try:
        from robosat.tools.rasterize import feature_to_mercator, burn
    except ImportError:
        # feature_to_mercator might not be directly importable, we'll define it ourselves
        feature_to_mercator = None
        burn = None
    from robosat.colors import make_palette, Mapbox
    ROBOSAT_AVAILABLE = True
except ImportError:
    ROBOSAT_AVAILABLE = False
    feature_to_mercator = None
    burn = None
    Mapbox = None
    print("Warning: robosat not available. Install robosat or use Docker.")
    print("Docker usage: docker run -it --rm -v $PWD:/data robosat:latest-cpu")
    
    # Fallback color map for make_palette
    COLOR_MAP = {
        "dark": (64, 64, 64),
        "gray": (238, 238, 238),
        "light": (248, 248, 248),
        "white": (255, 255, 255),
        "cyan": (59, 178, 208),
        "blue": (56, 135, 190),
        "bluedark": (34, 59, 83),
        "denim": (80, 102, 127),
        "navy": (40, 53, 61),
        "navydark": (34, 43, 48),
        "purple": (138, 138, 203),
        "teal": (65, 175, 165),
        "green": (86, 184, 129),
        "yellow": (241, 240, 117),
        "mustard": (251, 176, 59),
        "orange": (249, 136, 108),
        "red": (229, 94, 94),
        "pink": (237, 100, 152),
    }
    
    def make_palette(*colors):
        """Fallback make_palette implementation when robosat is not available."""
        rgbs = [COLOR_MAP.get(color, (128, 128, 128)) for color in colors]
        flattened = sum(rgbs, ())
        return list(flattened)

try:
    from src.dataset_builder.coco_exporter import COCOExporter
    from src.dataset_builder.splitter import split_dataset
    COCO_EXPORTER_AVAILABLE = True
except ImportError:
    COCO_EXPORTER_AVAILABLE = False
    print("Warning: COCO exporter not available. Install geoseg project dependencies.")


class RoboSatToCOCO:
    """Convert RoboSat workflow output to COCO format."""
    
    def __init__(
        self,
        output_dir: str,
        zoom: int = 19,
        tile_size: int = 512,
        feature_types: List[str] = None,
        test_tiles: Optional[int] = None
    ):
        """
        Initialize the converter.
        
        Args:
            output_dir: Base output directory
            zoom: Zoom level for tiles
            tile_size: Size of tiles in pixels
            feature_types: List of feature types to extract (building, road, parking)
            test_tiles: Limit number of tiles for testing (optional)
        """
        self.output_dir = Path(output_dir)
        self.zoom = zoom
        self.tile_size = tile_size
        # Handle "all" feature type
        if feature_types and "all" in feature_types:
            self.feature_types = ["building", "road", "parking"]
            self._use_all_handler = True
        else:
            self.feature_types = feature_types or ["building"]
            self._use_all_handler = False
        self.test_tiles = test_tiles
        
        # Create directory structure
        self.work_dir = self.output_dir / "robosat_work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.geojson_dir = self.work_dir / "geojson"
        self.tiles_dir = self.work_dir / "tiles"
        self.images_dir = self.work_dir / "images"
        self.masks_dir = self.work_dir / "masks"
        self.coco_dir = self.output_dir / "coco_dataset"
        
        for d in [self.geojson_dir, self.tiles_dir, self.images_dir, self.masks_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _validate_osm_file(self, osm_file: str) -> str:
        """Validate OSM file exists and return absolute path."""
        osm_file_abs = os.path.abspath(osm_file)
        if not os.path.exists(osm_file_abs):
            suggestions = []
            current_dir = Path.cwd()
            for alt_path in [
                current_dir / "data" / "Essex" / "essex-latest.osm.pbf",
                current_dir / "data" / "essex-latest.osm.pbf",
                current_dir.parent / "data" / "essex-latest.osm.pbf",
            ]:
                if alt_path.exists():
                    suggestions.append(str(alt_path.relative_to(current_dir)))
            
            error_msg = f"OSM file not found: {osm_file_abs}"
            if suggestions:
                error_msg += f"\n\nDid you mean one of these?\n  - " + "\n  - ".join(suggestions)
            else:
                error_msg += f"\n\nTo download a test file, run:\n  wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf -O data/essex-latest.osm.pbf"
            raise FileNotFoundError(error_msg)
        return osm_file_abs
    
    def extract_features_multi(self, osm_file: str, feature_types: List[str]) -> str:
        """Extract multiple feature types from OSM in a single pass."""
        if self._use_all_handler:
            print(f"\n[Step 1] Extracting all features (building, road, parking) from OSM...")
        else:
            print(f"\n[Step 1] Extracting {', '.join(feature_types)} features from OSM...")
        
        osm_file_abs = self._validate_osm_file(osm_file)
        geojson_file = self.geojson_dir / "merged_features.geojson"
        
        # Check if output file already exists
        if geojson_file.exists():
            print(f"  ⏭️  Skipping: GeoJSON file already exists ({geojson_file.stat().st_size:,} bytes)")
            return str(geojson_file)
        
        # Check for existing batch files (use first one if exists)
        batch_files = list(self.geojson_dir.glob("merged_features-*.geojson"))
        if batch_files:
            print(f"  ⏭️  Skipping: Found {len(batch_files)} batch file(s), using first one")
            return str(batch_files[0])
        
        # Extract using AllFeaturesHandler or Docker fallback
        if ROBOSAT_AVAILABLE:
            class Args:
                # Use "all" if we're extracting all types, otherwise use the list
                types = ["all"] if self._use_all_handler else feature_types
                batch = 100000
                map = osm_file_abs
                out = str(geojson_file)
            extract_main(Args())
        else:
            # Docker fallback: use AllFeaturesHandler to extract all types in one pass
            print("  Using Docker for extraction (multi-type, single pass)...")
            osm_dir = os.path.dirname(osm_file_abs)
            osm_filename = os.path.basename(osm_file_abs)
            
            # Build Docker command
            cmd = [
                "docker", "run", "-it", "--rm",
                "-v", f"{osm_dir}:/data",
                "-v", f"{os.path.abspath(self.geojson_dir)}:/output",
                "robosat:latest-cpu",
                "extract",
                f"/data/{osm_filename}",
                f"/output/merged_features.geojson"
            ]
            # Use "all" if extracting all types, otherwise add --type for each
            if self._use_all_handler:
                cmd.extend(["--type", "all"])
            else:
                for feature_type in feature_types:
                    cmd.extend(["--type", feature_type])
            
            subprocess.run(cmd, check=True)
        
        # Check for batch files created by FeatureStorage (use first one)
        batch_files = list(self.geojson_dir.glob("merged_features-*.geojson"))
        if batch_files:
            print(f"  ✅ Extracted features to {len(batch_files)} batch file(s)")
            return str(batch_files[0])
        
        if geojson_file.exists():
            return str(geojson_file)
        
        raise FileNotFoundError(f"GeoJSON file not found after extraction: {self.geojson_dir}")
    
    def extract_features(self, osm_file: str, feature_type: str) -> str:
        """Extract single feature type from OSM (uses extract_features_multi internally)."""
        return self.extract_features_multi(osm_file, [feature_type])
    
    def generate_tiles(self, geojson_file: str) -> str:
        """Generate tiles covering GeoJSON features."""
        print(f"\n[Step 2] Generating tiles covering features...")
        
        tiles_file = self.tiles_dir / "tiles.csv"
        
        if tiles_file.exists() and tiles_file.stat().st_size > 0:
            # Count tiles to show progress
            with open(tiles_file, 'r') as f:
                tile_count = sum(1 for _ in f)
            print(f"  ⏭️  Skipping: Tiles file already exists with {tile_count} tiles: {tiles_file}")
            return str(tiles_file)
        
        if ROBOSAT_AVAILABLE:
            class Args:
                zoom = self.zoom
                features = geojson_file
                out = str(tiles_file)
            
            args = Args()
            cover_main(args)
        else:
            # Fallback: use Docker
            print("  Using Docker for tile generation...")
            # Resolve geojson file to absolute path first
            geojson_file_abs = os.path.abspath(geojson_file)
            geojson_filename = os.path.basename(geojson_file_abs)
            
            cmd = [
                "docker", "run", "-it", "--rm",
                "-v", f"{os.path.abspath(self.geojson_dir)}:/data",
                "-v", f"{os.path.abspath(self.tiles_dir)}:/output",
                "robosat:latest-cpu",
                "cover",
                "--zoom", str(self.zoom),
                f"/data/{geojson_filename}",
                "/output/tiles.csv"
            ]
            subprocess.run(cmd, check=True)
        
        # Count tiles
        with open(tiles_file, 'r') as f:
            tile_count = sum(1 for _ in f)
        print(f"  ✅ Generated {tile_count} tiles")
        
        # Limit tiles if test_tiles is set
        if self.test_tiles is not None and tile_count > self.test_tiles:
            self._limit_tiles(tiles_file, self.test_tiles)
            print(f"  🔬 Limited to {self.test_tiles} tiles for testing")
        
        return str(tiles_file)
    
    def download_images(self, tiles_file: str, mapbox_token: Optional[str] = None, url_template: Optional[str] = None) -> str:
        """Download satellite images for tiles."""
        print(f"\n[Step 3] Downloading satellite images...")
        
        zoom_dir = self.images_dir / str(self.zoom)
        if zoom_dir.exists() and any(zoom_dir.iterdir()):
            # Count downloaded images
            image_count = sum(1 for _ in zoom_dir.rglob("*.png")) + sum(1 for _ in zoom_dir.rglob("*.webp"))
            print(f"  ⏭️  Skipping: Images already downloaded ({image_count} images) in {zoom_dir}")
            return str(self.images_dir)
        
        # Default to Mapbox if token provided, otherwise use OpenStreetMap
        if url_template is None:
            if mapbox_token:
                url_template = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.webp?access_token={mapbox_token}"
            else:
                # Use OpenStreetMap tile server (no token needed)
                url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        
        if ROBOSAT_AVAILABLE:
            class Args:
                url = url_template
                ext = "png"
                rate = 10
                tiles = tiles_file
                out = str(self.images_dir)
            
            args = Args()
            download_main(args)
        else:
            # Fallback: use Docker
            print("  Using Docker for image download...")
            cmd = [
                "docker", "run", "-it", "--rm",
                "-v", f"{os.path.abspath(self.tiles_dir)}:/data",
                "-v", f"{os.path.abspath(self.images_dir)}:/output",
                "robosat:latest-cpu",
                "download",
                url_template,
                "--ext", "png",
                "--rate", "10",
                "/data/tiles.csv",
                "/output"
            ]
            subprocess.run(cmd, check=True)
        
        print(f"  ✅ Downloaded images to {self.images_dir}")
        return str(self.images_dir)
    
    def _feature_to_mercator(self, feature):
        """Normalize feature and converts coords to 3857 (copied from robosat)."""
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3857)
        
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            xys = (zip(*part) for part in geometry["coordinates"])
            xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
            yield {"coordinates": list(xys), "type": "Polygon"}
        elif geometry["type"] == "MultiPolygon":
            for component in geometry["coordinates"]:
                xys = (zip(*part) for part in component)
                xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
                yield {"coordinates": list(xys), "type": "Polygon"}
    
    def _rasterize_multiclass(self, geojson_file: str, tiles_file: str, dataset_config: Path):
        """Custom multi-class rasterization with different pixel values per feature type."""
        from tqdm import tqdm
        
        # Load dataset config
        try:
            import toml
            with open(dataset_config, 'r') as f:
                dataset = toml.load(f)
        except ImportError:
            # Fallback: parse manually
            dataset = {"common": {"classes": ["background"] + self.feature_types}}
        
        classes = dataset["common"]["classes"]
        # Get colors from config, or generate them if missing
        if "colors" in dataset["common"]:
            colors = dataset["common"]["colors"]
        else:
            # Generate colors if missing
            colors = ["denim"]  # background
            for class_name in classes[1:]:  # skip background
                colors.append(self._get_color_for_feature_type(class_name))
        
        # Create mapping from feature type to pixel value (class index)
        feature_type_to_class = {ft: idx + 1 for idx, ft in enumerate(self.feature_types)}
        # Background is class 0
        
        # Load GeoJSON
        with open(geojson_file, 'r') as f:
            fc = json.load(f)
        
        # Group features by type and tile
        feature_map = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for feature in tqdm(fc["features"], ascii=True, unit="feature"):
            if feature["geometry"]["type"] not in ["Polygon", "MultiPolygon"]:
                continue
            
            props = feature.get("properties", {})
            # Support feature_types array (use first type) or fallback to feature_type
            feature_types = props.get("feature_types", None)
            if feature_types and isinstance(feature_types, list) and len(feature_types) > 0:
                # Use first type for rasterization (can be extended to support all types)
                feature_type = feature_types[0]
            else:
                # Fallback to singular feature_type for backward compatibility
                feature_type = props.get("feature_type", self.feature_types[0])
            
            if feature_type not in feature_type_to_class:
                continue
            
            try:
                for tile in burntiles.burn([feature], zoom=self.zoom):
                    tile_obj = mercantile.Tile(*tile)
                    feature_map[tile_obj][feature_type].append(feature)
            except ValueError:
                continue
        
        # Rasterize each tile
        os.makedirs(self.masks_dir, exist_ok=True)
        
        for tile in tqdm(list(tiles_from_csv(tiles_file)), ascii=True, unit="tile"):
            # Initialize mask with background (0)
            mask = np.zeros(shape=(self.tile_size, self.tile_size), dtype=np.uint8)
            
            if tile in feature_map:
                # Rasterize each feature type with its class value
                for feature_type, features in feature_map[tile].items():
                    class_value = feature_type_to_class[feature_type]
                    
                    # Convert features to mercator and rasterize
                    shapes = []
                    for feature in features:
                        for geometry in self._feature_to_mercator(feature):
                            shapes.append((geometry, class_value))
                    
                    if shapes:
                        bounds = mercantile.xy_bounds(tile)
                        transform = from_bounds(*bounds, self.tile_size, self.tile_size)
                        rasterized = rasterize(shapes, out_shape=(self.tile_size, self.tile_size), transform=transform)
                        # Use maximum to handle overlapping features (later features overwrite)
                        mask = np.maximum(mask, rasterized.astype(np.uint8))
            
            # Save mask
            out_dir = self.masks_dir / str(tile.z) / str(tile.x)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tile.y}.png"
            
            # Convert to PIL Image with palette
            img = Image.fromarray(mask, mode="P")
            palette = make_palette(*colors)
            img.putpalette(palette)
            img.save(out_path, optimize=True)
    
    def rasterize_masks(self, geojson_file: str, tiles_file: str, dataset_config: Optional[str] = None, use_multiclass: bool = True) -> str:
        """Rasterize GeoJSON features to masks with multi-class support."""
        print(f"\n[Step 4] Rasterizing features to masks...")
        
        zoom_mask_dir = self.masks_dir / str(self.zoom)
        # Check if masks actually exist (not just empty directories)
        mask_count = 0
        if zoom_mask_dir.exists():
            mask_count = sum(1 for _ in zoom_mask_dir.rglob("*.png"))
        
        if mask_count > 0:
            print(f"  ⏭️  Skipping: Masks already generated ({mask_count} masks) in {zoom_mask_dir}")
            return str(self.masks_dir)
        
        # Create dataset config
        if dataset_config is None:
            dataset_config = self.work_dir / "dataset.toml"
            self._create_dataset_config(dataset_config)
        else:
            dataset_config = Path(dataset_config)
        
        # Use robosat rasterize (now supports multi-class after our fix)
        if ROBOSAT_AVAILABLE:
            if len(self.feature_types) > 1:
                print(f"  Using multi-class rasterization for {len(self.feature_types)} feature types...")
            class Args:
                features = geojson_file
                tiles = tiles_file
                out = str(self.masks_dir)
                dataset = str(dataset_config)
                zoom = self.zoom
                size = self.tile_size
            
            args = Args()
            rasterize_main(args)
        else:
            # Fallback: use Docker (Docker image may have old code, so use custom rasterization)
            if len(self.feature_types) > 1:
                print("  Warning: Docker image may not support multi-class. Using custom rasterization...")
                self._rasterize_multiclass(geojson_file, tiles_file, dataset_config)
            else:
                # Binary mode with Docker
                print("  Using Docker for mask rasterization...")
                geojson_file_abs = os.path.abspath(geojson_file)
                geojson_filename = os.path.basename(geojson_file_abs)
                
                cmd = [
                    "docker", "run", "-it", "--rm",
                    "-v", f"{os.path.abspath(self.geojson_dir)}:/data/geojson",
                    "-v", f"{os.path.abspath(self.tiles_dir)}:/data/tiles",
                    "-v", f"{os.path.abspath(self.masks_dir)}:/output",
                    "-v", f"{os.path.abspath(dataset_config)}:/config/dataset.toml",
                    "robosat:latest-cpu",
                    "rasterize",
                    f"/data/geojson/{geojson_filename}",
                    "/data/tiles/tiles.csv",
                    "/output",
                    "--dataset", "/config/dataset.toml",
                    "--zoom", str(self.zoom),
                    "--size", str(self.tile_size)
                ]
                subprocess.run(cmd, check=True)
        
        print(f"  ✅ Generated masks in {self.masks_dir}")
        return str(self.masks_dir)
    
    def _limit_tiles(self, tiles_file: Path, limit: int):
        """Limit tiles CSV to first N tiles."""
        with open(tiles_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > limit:
            with open(tiles_file, 'w') as f:
                f.writelines(lines[:limit])
    
    def _get_color_for_feature_type(self, feature_type: str) -> str:
        """Get a color name for a feature type."""
        color_map = {
            "building": "orange",
            "road": "cyan",
            "parking": "yellow",
            "water": "blue",
            "forest": "green"
        }
        return color_map.get(feature_type, "red")
    
    def _create_dataset_config(self, config_path: Path):
        """Create dataset config with classes and colors for feature types."""
        classes = ["background"]
        colors = ["denim"]
        
        for feature_type in self.feature_types:
            classes.append(feature_type)
            colors.append(self._get_color_for_feature_type(feature_type))
        
        try:
            import toml
            config = {
                "common": {
                    "classes": classes,
                    "colors": colors
                }
            }
            with open(config_path, 'w') as f:
                toml.dump(config, f)
        except ImportError:
            # Fallback: write TOML manually
            classes_str = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
            colors_str = "[" + ", ".join(f'"{c}"' for c in colors) + "]"
            with open(config_path, 'w') as f:
                f.write(f"""[common]
classes = {classes_str}
colors = {colors_str}
""")
    
    def convert_to_coco(self, images_dir: str, masks_dir: str, split: bool = True, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Convert slippy map format to COCO format."""
        if not COCO_EXPORTER_AVAILABLE:
            print("Error: COCO exporter not available. Cannot convert to COCO format.")
            print("Please ensure geoseg project is available.")
            return
        
        print(f"\n[Step 5] Converting to COCO format...")
        
        # Check if COCO dataset already exists
        if self.coco_dir.exists():
            # Check if annotation files exist
            annotation_files = list(self.coco_dir.glob("annotations*.json"))
            if annotation_files:
                total_images = 0
                for split_name in ["train", "val", "test"]:
                    split_images_dir = self.coco_dir / split_name / "images"
                    if split_images_dir.exists():
                        total_images += sum(1 for _ in split_images_dir.glob("*.png"))
                
                if total_images > 0:
                    print(f"  ⏭️  Skipping: COCO dataset already exists ({len(annotation_files)} annotation files, {total_images} images) in {self.coco_dir}")
                    return
        
        # First, convert slippy map to flat structure
        flat_images_dir = self.work_dir / "flat_images"
        flat_masks_dir = self.work_dir / "flat_masks"
        flat_images_dir.mkdir(exist_ok=True)
        flat_masks_dir.mkdir(exist_ok=True)
        
        print("  Converting slippy map to flat structure...")
        
        # Get all tiles from images
        image_tiles = list(tiles_from_slippy_map(images_dir))
        mask_tiles = list(tiles_from_slippy_map(masks_dir))
        
        # Create mapping from tile to paths
        mask_dict = {tile: path for tile, path in mask_tiles}
        
        copied = 0
        for tile, image_path in image_tiles:
            if tile not in mask_dict:
                continue
            
            # Create flat filenames: z_x_y.png
            flat_name = f"{tile.z}_{tile.x}_{tile.y}.png"
            flat_image_path = flat_images_dir / flat_name
            flat_mask_path = flat_masks_dir / flat_name
            
            # Copy files
            shutil.copy2(image_path, flat_image_path)
            shutil.copy2(mask_dict[tile], flat_mask_path)
            copied += 1
        
        print(f"  ✅ Copied {copied} image-mask pairs")
        
        # Create temporary directory with images/ and masks/ structure for splitter
        temp_flat_dir = self.work_dir / "flat"
        temp_flat_images = temp_flat_dir / "images"
        temp_flat_masks = temp_flat_dir / "masks"
        temp_flat_images.mkdir(parents=True, exist_ok=True)
        temp_flat_masks.mkdir(parents=True, exist_ok=True)
        
        # Move files to temp directory
        for f in flat_images_dir.glob("*.png"):
            shutil.copy2(f, temp_flat_images / f.name)
        for f in flat_masks_dir.glob("*.png"):
            shutil.copy2(f, temp_flat_masks / f.name)
        
        if split:
            print("  Splitting dataset...")
            split_dataset(
                input_dir=str(temp_flat_dir),
                output_dir=str(self.coco_dir),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        else:
            # No split, just copy to coco_dir
            coco_images_dir = self.coco_dir / "images"
            coco_masks_dir = self.coco_dir / "masks"
            coco_images_dir.mkdir(parents=True, exist_ok=True)
            coco_masks_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copytree(temp_flat_images, coco_images_dir, dirs_exist_ok=True)
            shutil.copytree(temp_flat_masks, coco_masks_dir, dirs_exist_ok=True)
        
        # Export to COCO format
        exporter = COCOExporter()
        
        if split:
            for split_name in ["train", "val", "test"]:
                split_dir = self.coco_dir / split_name
                if split_dir.exists():
                    output_json = self.coco_dir / f"annotations_{split_name}.json"
                    exporter.export_split(str(split_dir), str(output_json), split_name)
        else:
            output_json = self.coco_dir / "annotations.json"
            exporter.export_split(str(self.coco_dir), str(output_json), split_name="all")
        
        print(f"  ✅ COCO dataset created in {self.coco_dir}")
    
    def run_full_pipeline(self, osm_file: str, mapbox_token: Optional[str] = None, url_template: Optional[str] = None, skip_download: bool = False, skip_rasterize: bool = False):
        """Run full pipeline from OSM to COCO."""
        print("=" * 60)
        print("RoboSat to COCO Dataset Preparation")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Work directory: {self.work_dir}")
        print(f"Feature types: {', '.join(self.feature_types)}")
        print(f"Zoom level: {self.zoom}")
        if self.test_tiles is not None:
            print(f"Test mode: Limited to {self.test_tiles} tiles")
        print()
        
        # Track what was processed vs skipped
        processed_steps = []
        skipped_steps = []
        
        # Step 1: Extract features
        if len(self.feature_types) > 1:
            # Use multi-type extraction (more efficient - single pass)
            existing_before = list(self.geojson_dir.glob("merged_features-*.geojson"))
            merged_geojson = self.extract_features_multi(osm_file, self.feature_types)
            existing_after = list(self.geojson_dir.glob("merged_features-*.geojson"))
            if existing_before and len(existing_after) == len(existing_before):
                skipped_steps.append(f"Extract {', '.join(self.feature_types)} (multi-type)")
            else:
                processed_steps.append(f"Extract {', '.join(self.feature_types)} (multi-type)")
            geojson_for_tiles = merged_geojson
            geojson_for_rasterize = merged_geojson
        else:
            # Single type extraction
            feature_type = self.feature_types[0]
            existing_before = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            geojson_file = self.extract_features(osm_file, feature_type)
            existing_after = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            if existing_before and len(existing_after) == len(existing_before):
                skipped_steps.append(f"Extract {feature_type}")
            else:
                processed_steps.append(f"Extract {feature_type}")
            geojson_for_tiles = geojson_file
            geojson_for_rasterize = geojson_file
        
        # Step 2: Generate tiles (use merged geojson to cover all features)
        tiles_file_before = self.tiles_dir / "tiles.csv"
        tiles_file = self.generate_tiles(geojson_for_tiles)
        if tiles_file_before.exists() and tiles_file_before.stat().st_size > 0:
            skipped_steps.append("Generate tiles")
        else:
            processed_steps.append("Generate tiles")
        
        # Step 3: Download images
        if not skip_download:
            zoom_dir_before = self.images_dir / str(self.zoom)
            had_images = zoom_dir_before.exists() and any(zoom_dir_before.iterdir())
            self.download_images(tiles_file, mapbox_token, url_template)
            if had_images:
                skipped_steps.append("Download images")
            else:
                processed_steps.append("Download images")
        else:
            skipped_steps.append("Download images (skipped by flag)")
        
        # Step 4: Rasterize masks (with multi-class support)
        if not skip_rasterize:
            zoom_mask_dir_before = self.masks_dir / str(self.zoom)
            had_masks = zoom_mask_dir_before.exists() and any(zoom_mask_dir_before.iterdir())
            # Use merged geojson for multi-class rasterization
            self.rasterize_masks(geojson_for_rasterize, tiles_file, use_multiclass=len(self.feature_types) > 1)
            if had_masks:
                skipped_steps.append("Rasterize masks")
            else:
                processed_steps.append("Rasterize masks")
        else:
            skipped_steps.append("Rasterize masks (skipped by flag)")
        
        # Step 5: Convert to COCO
        coco_existed = self.coco_dir.exists() and any(self.coco_dir.glob("annotations*.json"))
        self.convert_to_coco(str(self.images_dir), str(self.masks_dir))
        if coco_existed:
            skipped_steps.append("Convert to COCO")
        else:
            processed_steps.append("Convert to COCO")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ Pipeline completed!")
        print("=" * 60)
        if processed_steps:
            print(f"\n📝 Processed steps ({len(processed_steps)}):")
            for step in processed_steps:
                print(f"   ✓ {step}")
        if skipped_steps:
            print(f"\n⏭️  Skipped steps ({len(skipped_steps)}):")
            for step in skipped_steps:
                print(f"   ⊘ {step}")
        print(f"\nOutput directory: {self.coco_dir}")
        print(f"Work directory: {self.work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboSat workflow to COCO format"
    )
    parser.add_argument(
        "--osm-file",
        type=str,
        required=True,
        help="Path to .osm.pbf file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Zoom level for tiles (default: 19)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)"
    )
    parser.add_argument(
        "--feature-types",
        dest="feature_types",
        type=str,
        nargs="+",
        default=["building"],
        choices=["building", "road", "parking", "all"],
        help="Feature types to extract (default: building). Use 'all' to extract all types. Can use --feature-types or --feature_types"
    )
    parser.add_argument(
        "--feature_types",
        dest="feature_types",
        type=str,
        nargs="+",
        choices=["building", "road", "parking", "all"],
        help=argparse.SUPPRESS  # Hide from help since it's an alias
    )
    parser.add_argument(
        "--mapbox-token",
        type=str,
        default=os.environ.get("MAPBOX_API_KEY"),
        help="Mapbox API token (optional, uses OSM tiles if not provided). Can also be set via MAPBOX_API_KEY environment variable."
    )
    parser.add_argument(
        "--url-template",
        type=str,
        default=None,
        help="Custom URL template for tile download"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip image download step"
    )
    parser.add_argument(
        "--skip-rasterize",
        action="store_true",
        help="Skip mask rasterization step"
    )
    parser.add_argument(
        "--no-split",
        dest="split",
        action="store_false",
        help="Don't split into train/val/test"
    )
    parser.add_argument(
        "--test-tiles",
        type=int,
        default=None,
        help="Limit number of tiles for quick testing (e.g., --test-tiles 20)"
    )
    
    args = parser.parse_args()
    
    converter = RoboSatToCOCO(
        output_dir=args.output,
        zoom=args.zoom,
        tile_size=args.tile_size,
        feature_types=args.feature_types,
        test_tiles=args.test_tiles
    )
    
    converter.run_full_pipeline(
        osm_file=args.osm_file,
        mapbox_token=args.mapbox_token,
        url_template=args.url_template,
        skip_download=args.skip_download,
        skip_rasterize=args.skip_rasterize
    )


if __name__ == "__main__":
    main()

