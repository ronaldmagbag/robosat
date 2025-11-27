#!/usr/bin/env python
"""
Quick test script for RoboSat to COCO pipeline.

This script tests the pipeline with a small area to verify everything works.

Usage:
    python scripts/test_robosat_coco.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add robosat to path
robosat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(robosat_dir))

from scripts.robosat_to_coco import RoboSatToCOCO


def test_robosat_coco():
    """Test the RoboSat to COCO pipeline."""
    
    print("=" * 60)
    print("RoboSat to COCO - Quick Test")
    print("=" * 60)
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="robosat_test_"))
    print(f"\nTest directory: {test_dir}")
    
    try:
        # Check if we have a test OSM file
        # You can download a small OSM file for testing:
        # wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf
        # Or use any small .osm.pbf file
        
        test_osm_file = os.environ.get("TEST_OSM_FILE")
        if not test_osm_file or not os.path.exists(test_osm_file):
            print("\n⚠️  No test OSM file found.")
            print("Set TEST_OSM_FILE environment variable to point to a .osm.pbf file")
            print("Example:")
            print("  export TEST_OSM_FILE=/path/to/essex-latest.osm.pbf")
            print("\nOr download a test file:")
            print("  wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf")
            return
        
        print(f"\nUsing OSM file: {test_osm_file}")
        
        # Initialize converter
        converter = RoboSatToCOCO(
            output_dir=str(test_dir / "output"),
            zoom=18,  # Lower zoom for faster testing
            tile_size=512,
            feature_types=["building"]
        )
        
        # Run pipeline
        print("\nRunning pipeline...")
        converter.run_full_pipeline(
            osm_file=test_osm_file,
            mapbox_token=os.environ.get("MAPBOX_API_KEY"),
            skip_download=False,  # Set to True if you want to skip download
            skip_rasterize=False
        )
        
        # Check output
        coco_dir = converter.coco_dir
        print(f"\n✅ Test completed!")
        print(f"\nOutput structure:")
        print(f"  {coco_dir}")
        
        # List files
        if coco_dir.exists():
            for item in coco_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(coco_dir)
                    size = item.stat().st_size
                    print(f"    {rel_path} ({size:,} bytes)")
        
        # Check for COCO JSON files
        json_files = list(coco_dir.glob("annotations*.json"))
        if json_files:
            print(f"\n✅ Found {len(json_files)} COCO annotation file(s):")
            for json_file in json_files:
                print(f"    {json_file}")
        else:
            print("\n⚠️  No COCO annotation files found")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup option
        cleanup = os.environ.get("KEEP_TEST_DIR", "false").lower() != "true"
        if cleanup:
            print(f"\nCleaning up test directory: {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)
        else:
            print(f"\nTest directory kept: {test_dir}")
            print("Set KEEP_TEST_DIR=true to keep the directory")


if __name__ == "__main__":
    test_robosat_coco()

