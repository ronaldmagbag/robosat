import sys
import math

import osmium
import geojson
import shapely.geometry

from robosat.osm.core import FeatureStorage, is_polygon


class AllFeaturesHandler(osmium.SimpleHandler):
    """Unified handler that extracts building, parking, and road features in a single pass.
    
    This handler merges the logic from BuildingHandler, ParkingHandler, and RoadHandler
    to extract all feature types at once, with each feature tagged with its type.
    """

    # Building handler constants
    building_filter = set(
        ["construction", "houseboat", "static_caravan", "stadium", "conservatory", "digester", "greenhouse", "ruins"]
    )
    location_filter = set(["underground", "underwater"])

    # Parking handler constants
    parking_filter = set(["underground", "sheds", "carports", "garage_boxes"])

    # Road handler constants
    highway_attributes = {
        "motorway": {
            "lanes": 4,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.0,
        },
        "trunk": {"lanes": 3, "lane_width": 3.75, "left_hard_shoulder_width": 0.75, "right_hard_shoulder_width": 3.0},
        "primary": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "secondary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "unclassified": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "residential": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "service": {
            "lanes": 1,
            "lane_width": 3.00,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "motorway_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.00,
        },
        "trunk_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "primary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "secondary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
    }

    road_filter = set(highway_attributes.keys())
    EARTH_MEAN_RADIUS = 6371004.0

    def __init__(self, out, batch, feature_types=None):
        """Initialize the handler.
        
        Args:
            out: Output file path
            batch: Batch size for feature storage
            feature_types: List of feature types to extract (None or empty list means all types)
                          Options: ["building", "parking", "road"]
        """
        super().__init__()
        self.storage = FeatureStorage(out, batch)
        # If feature_types is None or empty, extract all types
        self.feature_types = set(feature_types) if feature_types else {"building", "parking", "road"}

    def way(self, w):
        """Process each way and extract all matching feature types in one pass.
        
        Collects all matching feature types for a single way and creates one feature
        with feature_types array, avoiding duplicate geometries.
        """
        matched_types = []
        geometry = None
        shape = None
        
        # Check for building features
        is_building = False
        if "building" in self.feature_types and is_polygon(w) and "building" in w.tags:
            if w.tags["building"] not in self.building_filter:
                if "location" not in w.tags or w.tags["location"] not in self.location_filter:
                    is_building = True
                    matched_types.append("building")
                    # Use polygon geometry for buildings
                    if geometry is None:
                        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
                        shape = shapely.geometry.shape(geometry)
        
        # Check for parking features
        is_parking = False
        if "parking" in self.feature_types and is_polygon(w) and "amenity" in w.tags and w.tags["amenity"] == "parking":
            if "parking" not in w.tags or w.tags["parking"] not in self.parking_filter:
                is_parking = True
                matched_types.append("parking")
                # Use polygon geometry for parking (same as building if both match)
                if geometry is None:
                    geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
                    shape = shapely.geometry.shape(geometry)
        
        # Check for road features
        is_road = False
        if "road" in self.feature_types and "highway" in w.tags and w.tags["highway"] in self.road_filter:
            is_road = True
            matched_types.append("road")
            # Calculate road geometry (buffered LineString)
            left_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["left_hard_shoulder_width"]
            lane_width = self.highway_attributes[w.tags["highway"]]["lane_width"]
            lanes = self.highway_attributes[w.tags["highway"]]["lanes"]
            right_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["right_hard_shoulder_width"]
            
            if "oneway" not in w.tags:
                lanes = lanes * 2
            elif w.tags["oneway"] == "no":
                lanes = lanes * 2
            
            if "lanes" in w.tags:
                try:
                    lanes = max(int(w.tags["lanes"]), 1)
                except ValueError:
                    print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)
            
            road_width = left_hard_shoulder_width + lane_width * lanes + right_hard_shoulder_width
            
            if "width" in w.tags:
                try:
                    road_width = max(float(w.tags["width"]), 1.0)
                except ValueError:
                    print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)
            
            # Road geometry is buffered LineString
            if geometry is None:
                line_geometry = geojson.LineString([(n.lon, n.lat) for n in w.nodes])
                line_shape = shapely.geometry.shape(line_geometry)
                if line_shape.is_valid:
                    geometry_buffer = line_shape.buffer(math.degrees(road_width / 2.0 / self.EARTH_MEAN_RADIUS))
                    geometry = shapely.geometry.mapping(geometry_buffer)
                    shape = shapely.geometry.shape(geometry)
                else:
                    print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)
                    return
        
        # Create single feature with all matched types
        if matched_types and geometry is not None and shape is not None and shape.is_valid:
            # Use feature_types array to support multiple types per feature
            feature = geojson.Feature(
                geometry=geometry,
                properties={"feature_types": matched_types}
            )
            self.storage.add(feature)
        elif matched_types:
            # Some types matched but geometry was invalid
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def flush(self):
        self.storage.flush()

