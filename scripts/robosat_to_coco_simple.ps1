# Simple PowerShell script to run RoboSat to COCO pipeline using Docker
# This is a simpler alternative that uses Docker directly

param(
    [string]$OsmFile = "data.osm.pbf",
    [string]$OutputDir = "./data/coco_dataset",
    [int]$Zoom = 19,
    [string]$FeatureType = "building",
    [string]$MapboxToken = $env:MAPBOX_API_KEY
)

Write-Host "============================================================"
Write-Host "RoboSat to COCO - Simple Pipeline (Docker)"
Write-Host "============================================================"
Write-Host ""
Write-Host "OSM File: $OsmFile"
Write-Host "Output: $OutputDir"
Write-Host "Zoom: $Zoom"
Write-Host "Feature Type: $FeatureType"
Write-Host ""

# Create directories
$WorkDir = Join-Path $OutputDir "robosat_work"
$GeoJsonDir = Join-Path $WorkDir "geojson"
$TilesDir = Join-Path $WorkDir "tiles"
$ImagesDir = Join-Path $WorkDir "images"
$MasksDir = Join-Path $WorkDir "masks"

New-Item -ItemType Directory -Force -Path $GeoJsonDir | Out-Null
New-Item -ItemType Directory -Force -Path $TilesDir | Out-Null
New-Item -ItemType Directory -Force -Path $ImagesDir | Out-Null
New-Item -ItemType Directory -Force -Path $MasksDir | Out-Null

# Get absolute paths for Docker volume mounts
$OsmFileAbs = (Resolve-Path $OsmFile).Path
$OsmDir = Split-Path $OsmFileAbs -Parent
$OsmFileName = Split-Path $OsmFileAbs -Leaf

$GeoJsonDirAbs = (Resolve-Path $GeoJsonDir).Path
$TilesDirAbs = (Resolve-Path $TilesDir).Path
$ImagesDirAbs = (Resolve-Path $ImagesDir).Path
$MasksDirAbs = (Resolve-Path $MasksDir).Path

# Step 1: Extract features
Write-Host "[Step 1/5] Extracting $FeatureType features from OSM..."
docker run -it --rm `
    -v "${OsmDir}:/data" `
    -v "${GeoJsonDirAbs}:/output" `
    robosat:latest-cpu `
    extract --type $FeatureType `
    "/data/$OsmFileName" `
    "/output/$FeatureType.geojson"

# Step 2: Generate tiles
Write-Host ""
Write-Host "[Step 2/5] Generating tiles..."
docker run -it --rm `
    -v "${GeoJsonDirAbs}:/data" `
    -v "${TilesDirAbs}:/output" `
    robosat:latest-cpu `
    cover --zoom $Zoom `
    "/data/$FeatureType.geojson" `
    "/output/tiles.csv"

# Step 3: Download images
Write-Host ""
Write-Host "[Step 3/5] Downloading satellite images..."
if ($MapboxToken) {
    $Url = "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.webp?access_token=$MapboxToken"
} else {
    $Url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    Write-Host "  Using OpenStreetMap tiles (no token needed)"
}

docker run -it --rm `
    -v "${TilesDirAbs}:/data" `
    -v "${ImagesDirAbs}:/output" `
    robosat:latest-cpu `
    download $Url --ext png --rate 10 `
    "/data/tiles.csv" `
    "/output"

# Step 4: Create dataset config
Write-Host ""
Write-Host "[Step 4/5] Creating dataset config..."
$ConfigFile = Join-Path $WorkDir "dataset.toml"
@"
[common]
classes = ["background", "foreground"]
colors = ["denim", "orange"]
"@ | Out-File -FilePath $ConfigFile -Encoding utf8

# Step 5: Rasterize masks
Write-Host ""
Write-Host "[Step 5/5] Rasterizing features to masks..."
$ConfigFileAbs = (Resolve-Path $ConfigFile).Path

docker run -it --rm `
    -v "${GeoJsonDirAbs}:/data/geojson" `
    -v "${TilesDirAbs}:/data/tiles" `
    -v "${MasksDirAbs}:/output" `
    -v "${ConfigFileAbs}:/config/dataset.toml" `
    robosat:latest-cpu `
    rasterize `
    "/data/geojson/$FeatureType.geojson" `
    "/data/tiles/tiles.csv" `
    "/output" `
    --dataset "/config/dataset.toml" `
    --zoom $Zoom `
    --size 512

Write-Host ""
Write-Host "============================================================"
Write-Host "✅ RoboSat pipeline completed!"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next step: Convert to COCO format"
Write-Host "Run: python scripts/convert_robosat_slippy_to_coco.py --images $ImagesDirAbs --masks $MasksDirAbs --output $(Join-Path $OutputDir 'coco_dataset')"
Write-Host ""

