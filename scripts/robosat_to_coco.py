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
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
    ROBOSAT_AVAILABLE = True
except ImportError:
    ROBOSAT_AVAILABLE = False
    print("Warning: robosat not available. Install robosat or use Docker.")
    print("Docker usage: docker run -it --rm -v $PWD:/data robosat:latest-cpu")

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
        feature_types: List[str] = None
    ):
        """
        Initialize the converter.
        
        Args:
            output_dir: Base output directory
            zoom: Zoom level for tiles
            tile_size: Size of tiles in pixels
            feature_types: List of feature types to extract (building, road, parking)
        """
        self.output_dir = Path(output_dir)
        self.zoom = zoom
        self.tile_size = tile_size
        self.feature_types = feature_types or ["building"]
        
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
    
    def extract_features(self, osm_file: str, feature_type: str) -> str:
        """
        Extract GeoJSON features from OSM file using rs extract.
        
        Args:
            osm_file: Path to .osm.pbf file
            feature_type: Type of feature (building, road, parking)
            
        Returns:
            Path to output GeoJSON file
        """
        print(f"\n[Step 1] Extracting {feature_type} features from OSM...")
        
        # Resolve and check OSM file exists
        osm_file_abs = os.path.abspath(osm_file)
        if not os.path.exists(osm_file_abs):
            # Try to suggest alternative paths
            suggestions = []
            current_dir = Path.cwd()
            # Check common locations
            common_paths = [
                current_dir / "data" / "Essex" / "essex-latest.osm.pbf",
                current_dir / "data" / "essex-latest.osm.pbf",
                current_dir.parent / "data" / "essex-latest.osm.pbf",
            ]
            for alt_path in common_paths:
                if alt_path.exists():
                    suggestions.append(str(alt_path.relative_to(current_dir)))
            
            error_msg = f"OSM file not found: {osm_file_abs}"
            if suggestions:
                error_msg += f"\n\nDid you mean one of these?\n  - " + "\n  - ".join(suggestions)
            else:
                error_msg += f"\n\nTo download a test file, run:\n  wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf -O data/essex-latest.osm.pbf"
            raise FileNotFoundError(error_msg)
        
        geojson_file = self.geojson_dir / f"{feature_type}.geojson"
        
        # Check if any geojson files already exist (robosat creates files with UUIDs)
        existing_files = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
        if existing_files:
            # Use the most recently created file
            geojson_file = max(existing_files, key=lambda p: p.stat().st_mtime)
            file_size = geojson_file.stat().st_size
            print(f"  ⏭️  Skipping: GeoJSON file already exists ({file_size:,} bytes): {geojson_file.name}")
            return str(geojson_file)
        
        # Use robosat extract tool
        if ROBOSAT_AVAILABLE:
            # Create argparse namespace for extract tool
            class Args:
                type = feature_type
                batch = 100000
                map = osm_file_abs  # Use resolved absolute path
                out = str(geojson_file)
            
            args = Args()
            extract_main(args)
        else:
            # Fallback: use Docker
            print("  Using Docker for extraction...")
            # OSM file already resolved above
            osm_dir = os.path.dirname(osm_file_abs)
            osm_filename = os.path.basename(osm_file_abs)
            
            cmd = [
                "docker", "run", "-it", "--rm",
                "-v", f"{osm_dir}:/data",
                "-v", f"{os.path.abspath(self.geojson_dir)}:/output",
                "robosat:latest-cpu",
                "extract",
                "--type", feature_type,
                f"/data/{osm_filename}",
                f"/output/{feature_type}.geojson"
            ]
            subprocess.run(cmd, check=True)
        
        # RoboSat creates files with UUIDs, so find the actual file(s) created
        created_files = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
        if created_files:
            # Use the most recently created file
            geojson_file = max(created_files, key=lambda p: p.stat().st_mtime)
            print(f"  ✅ Extracted features to {geojson_file}")
            return str(geojson_file)
        else:
            # Fallback: check if the exact filename exists
            if geojson_file.exists():
                print(f"  ✅ Extracted features to {geojson_file}")
                return str(geojson_file)
            else:
                raise FileNotFoundError(f"GeoJSON file not found after extraction. Expected in {self.geojson_dir}")
    
    def generate_tiles(self, geojson_file: str) -> str:
        """
        Generate tiles covering GeoJSON features using rs cover.
        
        Args:
            geojson_file: Path to GeoJSON file
            
        Returns:
            Path to tiles CSV file
        """
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
        return str(tiles_file)
    
    def download_images(
        self,
        tiles_file: str,
        mapbox_token: Optional[str] = None,
        url_template: Optional[str] = None
        ) -> str:
        """
        Download satellite images using rs download.
        
        Args:
            tiles_file: Path to tiles CSV file
            mapbox_token: Mapbox API token (optional)
            url_template: Custom URL template (optional)
            
        Returns:
            Path to images directory
        """
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
    
    def rasterize_masks(
        self,
        geojson_file: str,
        tiles_file: str,
        dataset_config: Optional[str] = None
        ) -> str:
        """
        Rasterize GeoJSON features to masks using rs rasterize.
        
        Args:
            geojson_file: Path to GeoJSON file
            tiles_file: Path to tiles CSV file
            dataset_config: Path to dataset config file (optional)
            
        Returns:
            Path to masks directory
        """
        print(f"\n[Step 4] Rasterizing features to masks...")
        
        zoom_mask_dir = self.masks_dir / str(self.zoom)
        if zoom_mask_dir.exists() and any(zoom_mask_dir.iterdir()):
            # Count generated masks
            mask_count = sum(1 for _ in zoom_mask_dir.rglob("*.png"))
            print(f"  ⏭️  Skipping: Masks already generated ({mask_count} masks) in {zoom_mask_dir}")
            return str(self.masks_dir)
        
        # Create a simple dataset config if not provided
        if dataset_config is None:
            dataset_config = self.work_dir / "dataset.toml"
            self._create_dataset_config(dataset_config)
        
        if ROBOSAT_AVAILABLE:
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
            # Fallback: use Docker
            print("  Using Docker for mask rasterization...")
            # Resolve geojson file to absolute path first
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
    
    def _create_dataset_config(self, config_path: Path):
        """Create a simple dataset config for robosat."""
        try:
            import toml
            config = {
                "common": {
                    "classes": ["background", "foreground"],
                    "colors": ["denim", "orange"]
                }
            }
            with open(config_path, 'w') as f:
                toml.dump(config, f)
        except ImportError:
            # Fallback: write TOML manually
            with open(config_path, 'w') as f:
                f.write("""[common]
classes = ["background", "foreground"]
colors = ["denim", "orange"]
""")
    
    def convert_to_coco(
        self,
        images_dir: str,
        masks_dir: str,
        split: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """
        Convert slippy map format to COCO format.
        
        Args:
            images_dir: Directory with images in slippy map format (z/x/y.png)
            masks_dir: Directory with masks in slippy map format (z/x/y.png)
            split: Whether to split into train/val/test
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
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
    
    def run_full_pipeline(
        self,
        osm_file: str,
        mapbox_token: Optional[str] = None,
        url_template: Optional[str] = None,
        skip_download: bool = False,
        skip_rasterize: bool = False
    ):
        """
        Run the full pipeline from OSM to COCO.
        
        Args:
            osm_file: Path to .osm.pbf file
            mapbox_token: Mapbox API token (optional)
            url_template: Custom URL template (optional)
            skip_download: Skip image download step
            skip_rasterize: Skip mask rasterization step
        """
        print("=" * 60)
        print("RoboSat to COCO Dataset Preparation")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Work directory: {self.work_dir}")
        print(f"Feature types: {', '.join(self.feature_types)}")
        print(f"Zoom level: {self.zoom}")
        print()
        
        # Track what was processed vs skipped
        processed_steps = []
        skipped_steps = []
        
        # Step 1: Extract features
        geojson_files = []
        for feature_type in self.feature_types:
            # Check if file already exists before extraction
            existing_before = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            geojson_file = self.extract_features(osm_file, feature_type)
            geojson_files.append(geojson_file)
            # Check if it was skipped (file existed before)
            existing_after = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            if existing_before and len(existing_after) == len(existing_before):
                skipped_steps.append(f"Extract {feature_type}")
            else:
                processed_steps.append(f"Extract {feature_type}")
        
        # Step 2: Generate tiles (use first geojson for now)
        tiles_file_before = self.tiles_dir / "tiles.csv"
        tiles_file = self.generate_tiles(geojson_files[0])
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
        
        # Step 4: Rasterize masks
        if not skip_rasterize:
            zoom_mask_dir_before = self.masks_dir / str(self.zoom)
            had_masks = zoom_mask_dir_before.exists() and any(zoom_mask_dir_before.iterdir())
            # For now, rasterize first feature type
            # TODO: Support multiple feature types
            self.rasterize_masks(geojson_files[0], tiles_file)
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
        type=str,
        nargs="+",
        default=["building"],
        choices=["building", "road", "parking"],
        help="Feature types to extract (default: building)"
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
    
    args = parser.parse_args()
    
    converter = RoboSatToCOCO(
        output_dir=args.output,
        zoom=args.zoom,
        tile_size=args.tile_size,
        feature_types=args.feature_types
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

