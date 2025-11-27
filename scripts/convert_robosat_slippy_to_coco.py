#!/usr/bin/env python
"""
Convert RoboSat slippy map output to COCO format.

This script converts the slippy map format (z/x/y.png) from RoboSat
to a flat COCO format dataset.

Usage:
    python scripts/convert_robosat_slippy_to_coco.py --images ./images --masks ./masks --output ./coco_dataset
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Add robosat to path
robosat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(robosat_dir))

# Add geoseg src to path (for COCO exporter)
geoseg_dir = robosat_dir.parent.parent
sys.path.insert(0, str(geoseg_dir))

try:
    from robosat.tiles import tiles_from_slippy_map
    ROBOSAT_AVAILABLE = True
except ImportError:
    ROBOSAT_AVAILABLE = False
    print("Warning: robosat.tiles not available. Install robosat or use Docker.")

try:
    from src.dataset_builder.splitter import split_dataset
    from src.dataset_builder.coco_exporter import COCOExporter
    COCO_EXPORTER_AVAILABLE = True
except ImportError:
    COCO_EXPORTER_AVAILABLE = False
    print("Warning: COCO exporter not available. Install geoseg project dependencies.")


def convert_slippy_to_coco(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
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
        output_dir: Output directory for COCO dataset
        split: Whether to split into train/val/test
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    if not ROBOSAT_AVAILABLE:
        print("Error: robosat.tiles not available. Cannot read slippy map format.")
        print("Please install robosat: pip install robosat")
        return
    
    if not COCO_EXPORTER_AVAILABLE:
        print("Error: COCO exporter not available. Cannot convert to COCO format.")
        print("Please ensure geoseg project is available.")
        return
    
    print("=" * 60)
    print("Converting RoboSat Slippy Map to COCO Format")
    print("=" * 60)
    
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    
    # Create temporary flat structure
    temp_dir = output_dir.parent / f"{output_dir.name}_temp"
    flat_images_dir = temp_dir / "images"
    flat_masks_dir = temp_dir / "masks"
    flat_images_dir.mkdir(parents=True, exist_ok=True)
    flat_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 1/3] Converting slippy map to flat structure...")
    
    # Get all tiles from images
    print("  Reading image tiles...")
    image_tiles = list(tiles_from_slippy_map(str(images_dir)))
    print(f"  Found {len(image_tiles)} image tiles")
    
    # Get all tiles from masks
    print("  Reading mask tiles...")
    mask_tiles = list(tiles_from_slippy_map(str(masks_dir)))
    print(f"  Found {len(mask_tiles)} mask tiles")
    
    # Create mapping from tile to paths
    mask_dict = {tile: path for tile, path in mask_tiles}
    
    # Copy matching pairs
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
        
        if copied % 100 == 0:
            print(f"  Copied {copied} pairs...")
    
    print(f"  ✅ Copied {copied} image-mask pairs")
    
    if copied == 0:
        print("  ⚠️  No matching image-mask pairs found!")
        return
    
    # Split dataset
    if split:
        print(f"\n[Step 2/3] Splitting dataset...")
        split_dataset(
            input_dir=str(temp_dir),
            output_dir=str(output_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    else:
        # No split, just copy to output
        output_images_dir = output_dir / "images"
        output_masks_dir = output_dir / "masks"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_masks_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(flat_images_dir, output_images_dir, dirs_exist_ok=True)
        shutil.copytree(flat_masks_dir, output_masks_dir, dirs_exist_ok=True)
    
    # Export to COCO format
    print(f"\n[Step 3/3] Exporting to COCO format...")
    exporter = COCOExporter()
    
    if split:
        for split_name in ["train", "val", "test"]:
            split_dir = output_dir / split_name
            if split_dir.exists() and (split_dir / "images").exists():
                output_json = output_dir / f"annotations_{split_name}.json"
                exporter.export_split(str(split_dir), str(output_json), split_name)
    else:
        output_json = output_dir / "annotations.json"
        exporter.export_split(str(output_dir), str(output_json), split_name="all")
    
    # Cleanup temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("✅ COCO Conversion Completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    if split:
        print("COCO annotation files:")
        for split_name in ["train", "val", "test"]:
            json_path = output_dir / f"annotations_{split_name}.json"
            if json_path.exists():
                print(f"  - {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboSat slippy map output to COCO format"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory with images in slippy map format (z/x/y.png)"
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Directory with masks in slippy map format (z/x/y.png)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--no-split",
        dest="split",
        action="store_false",
        help="Don't split into train/val/test"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    
    args = parser.parse_args()
    
    convert_slippy_to_coco(
        images_dir=args.images,
        masks_dir=args.masks,
        output_dir=args.output,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()

