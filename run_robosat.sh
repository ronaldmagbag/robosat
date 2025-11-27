git clone https://github.com/mapbox/robosat.git
cd robosat/
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
mkdir data
mkdir data/Essex
cd data/Essex/
wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf
# Invoke-WebRequest -Uri "https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf" -OutFile "essex-latest.osm.pbf"
# curl -L -o essex-latest.osm.pbf https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf

cd ../..
docker run -it --rm -v ./data:/data --ipc=host --network=host mapbox/robosat:latest-cpu extract --type building /data/Essex/essex-latest.osm.pbf /data/Essex/essex.geojson

docker run -it --rm -v ./data:/data --ipc=host --network=host mapbox/robosat:latest-cpu cover --zoom 19 /data/Essex/essex-6eb025d8b1404c4b8c9e8ff70225a2a9.geojson /data/Essex/essex_building_tiles.txt

docker run -it --rm -v ./data:/data --ipc=host --network=host mapbox/robosat:latest-gpu cover --zoom 19 /data/Essex/essex-7c1a0ed5352547448f1aa58d8a1d99ac.geojson /data/Essex/essex_building_tiles.txt