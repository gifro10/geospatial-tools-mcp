"""
Geospatial Tools MCP Server
A comprehensive suite of geospatial utilities for AI agents.

Tools:
  - Coordinate transformation (CRS/EPSG)
  - Spatial data validation (topology, duplicates, CRS detection)
  - Elevation/raster value extraction
  - Viewshed / line-of-sight analysis
  - Drone flight planning (GSD, overlap, coverage)
"""

import math
import json
from mcp.server.fastmcp import FastMCP
from pyproj import Transformer, CRS
from pyproj.exceptions import CRSError
from shapely.geometry import shape, mapping
from shapely.validation import explain_validity
from shapely import wkt as shapely_wkt
import httpx

# Initialize the MCP server
mcp = FastMCP("Geospatial Tools")


# ============================================================
# COORDINATE TRANSFORMATION TOOLS
# ============================================================

@mcp.tool()
def transform_coordinates(
    coordinates: list[float],
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857"
) -> dict:
    """
    Transform coordinates between coordinate reference systems (EPSG codes).

    Args:
        coordinates: [x, y] or [[x1,y1], [x2,y2], ...] coordinate pairs
        source_crs: Source CRS (e.g., "EPSG:4326" for WGS84 GPS coordinates)
        target_crs: Target CRS (e.g., "EPSG:3857" for Web Mercator, "EPSG:26910" for UTM 10N)

    Returns:
        Transformed coordinates with source/target CRS metadata
    """
    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        if len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
            x, y = coordinates
            tx, ty = transformer.transform(x, y)
            return {
                "success": True,
                "source_crs": source_crs,
                "target_crs": target_crs,
                "input": coordinates,
                "output": [round(tx, 6), round(ty, 6)]
            }

        results = []
        for coord in coordinates:
            if len(coord) >= 2:
                tx, ty = transformer.transform(coord[0], coord[1])
                results.append([round(tx, 6), round(ty, 6)])

        return {
            "success": True,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "input_count": len(results),
            "output": results
        }

    except CRSError as e:
        return {"success": False, "error": f"Invalid CRS: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def batch_transform_coordinates(
    x_coords: list[float],
    y_coords: list[float],
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857"
) -> dict:
    """
    Efficiently transform large batches of coordinates using vectorized operations.
    Optimized for thousands of points. Provide X and Y as separate arrays.

    Args:
        x_coords: List of X values (longitude if geographic CRS)
        y_coords: List of Y values (latitude if geographic CRS)
        source_crs: Source CRS (e.g., "EPSG:4326")
        target_crs: Target CRS (e.g., "EPSG:3857")

    Returns:
        Transformed X/Y arrays with bounds and failure count
    """
    try:
        if len(x_coords) != len(y_coords):
            return {"success": False, "error": f"Array length mismatch: x={len(x_coords)}, y={len(y_coords)}"}

        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        tx, ty = transformer.transform(x_coords, y_coords)

        tx_list = [round(x, 6) if x == x else None for x in tx]
        ty_list = [round(y, 6) if y == y else None for y in ty]
        failed = sum(1 for x in tx_list if x is None)

        valid_x = [x for x in tx_list if x is not None]
        valid_y = [y for y in ty_list if y is not None]

        return {
            "success": True,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "count": len(x_coords),
            "failed_count": failed,
            "x_transformed": tx_list,
            "y_transformed": ty_list,
            "bounds": {
                "min_x": min(valid_x) if valid_x else None,
                "max_x": max(valid_x) if valid_x else None,
                "min_y": min(valid_y) if valid_y else None,
                "max_y": max(valid_y) if valid_y else None,
            }
        }

    except CRSError as e:
        return {"success": False, "error": f"Invalid CRS: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def transform_geojson(
    geojson: dict,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857"
) -> dict:
    """
    Transform a GeoJSON geometry (Point, LineString, Polygon, etc.) to a different CRS.

    Args:
        geojson: GeoJSON geometry object with "type" and "coordinates"
        source_crs: Source CRS
        target_crs: Target CRS

    Returns:
        Transformed GeoJSON geometry
    """
    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        def transform_coords(coords):
            if isinstance(coords[0], (int, float)):
                tx, ty = transformer.transform(coords[0], coords[1])
                return [round(tx, 6), round(ty, 6)]
            return [transform_coords(c) for c in coords]

        return {
            "success": True,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "geometry": {
                "type": geojson.get("type", ""),
                "coordinates": transform_coords(geojson.get("coordinates", []))
            }
        }

    except CRSError as e:
        return {"success": False, "error": f"Invalid CRS: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_crs_info(crs_code: str) -> dict:
    """
    Get detailed information about a coordinate reference system by EPSG code.

    Args:
        crs_code: EPSG code (e.g., "EPSG:4326", "EPSG:26910", "EPSG:3005")

    Returns:
        CRS name, type, datum, axis info, and geographic area of use
    """
    try:
        crs = CRS.from_user_input(crs_code)
        area = crs.area_of_use

        return {
            "success": True,
            "code": crs_code,
            "name": crs.name,
            "type": crs.type_name,
            "is_geographic": crs.is_geographic,
            "is_projected": crs.is_projected,
            "axis_info": [
                {"name": ax.name, "unit": ax.unit_name, "direction": ax.direction}
                for ax in crs.axis_info
            ],
            "area_of_use": {
                "name": area.name if area else None,
                "bounds": [area.west, area.south, area.east, area.north] if area else None
            } if area else None,
            "datum": crs.datum.name if crs.datum else None
        }

    except CRSError as e:
        return {"success": False, "error": f"Invalid CRS: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_common_crs() -> dict:
    """
    List commonly used coordinate reference systems for quick reference.
    Includes GPS (WGS84), Web Mercator, NAD83, UTM zones, and regional CRS.

    Returns:
        List of common EPSG codes with names and descriptions
    """
    return {
        "success": True,
        "crs_list": [
            {"code": "EPSG:4326", "name": "WGS 84", "description": "GPS coordinates (lat/lon)", "type": "Geographic"},
            {"code": "EPSG:3857", "name": "Web Mercator", "description": "Google Maps, OpenStreetMap", "type": "Projected"},
            {"code": "EPSG:4269", "name": "NAD83", "description": "North American Datum 1983", "type": "Geographic"},
            {"code": "EPSG:26910", "name": "NAD83 / UTM zone 10N", "description": "BC Coast, Washington", "type": "Projected"},
            {"code": "EPSG:26911", "name": "NAD83 / UTM zone 11N", "description": "BC Interior, Alberta", "type": "Projected"},
            {"code": "EPSG:32610", "name": "WGS 84 / UTM zone 10N", "description": "West Coast (WGS84)", "type": "Projected"},
            {"code": "EPSG:32611", "name": "WGS 84 / UTM zone 11N", "description": "Interior (WGS84)", "type": "Projected"},
            {"code": "EPSG:3005", "name": "NAD83 / BC Albers", "description": "British Columbia standard", "type": "Projected"},
            {"code": "EPSG:3347", "name": "NAD83(CSRS) / Statistics Canada Lambert", "description": "Canada Lambert", "type": "Projected"},
            {"code": "EPSG:2154", "name": "RGF93 / Lambert-93", "description": "France", "type": "Projected"},
            {"code": "EPSG:27700", "name": "OSGB 1936 / British National Grid", "description": "United Kingdom", "type": "Projected"},
        ]
    }


# ============================================================
# SPATIAL DATA VALIDATION TOOLS
# ============================================================

@mcp.tool()
def validate_geojson(geojson: dict) -> dict:
    """
    Validate a GeoJSON geometry for topology errors, self-intersections,
    and structural issues. Reports fixable problems and can auto-repair.

    Args:
        geojson: A GeoJSON geometry object (Point, LineString, Polygon, MultiPolygon, etc.)

    Returns:
        Validation result with is_valid flag, error details, and auto-fixed geometry if applicable
    """
    try:
        geom = shape(geojson)
        is_valid = geom.is_valid
        validity_reason = explain_validity(geom)

        result = {
            "success": True,
            "is_valid": is_valid,
            "geometry_type": geom.geom_type,
            "reason": validity_reason if not is_valid else "Valid Geometry",
            "is_empty": geom.is_empty,
            "has_z": geom.has_z,
            "area": round(geom.area, 6) if hasattr(geom, 'area') else None,
            "length": round(geom.length, 6) if hasattr(geom, 'length') else None,
            "num_coordinates": len(geom.exterior.coords) if hasattr(geom, 'exterior') else (len(list(geom.coords)) if hasattr(geom, 'coords') else None),
            "bounds": list(geom.bounds) if not geom.is_empty else None,
        }

        # Auto-fix if invalid
        if not is_valid:
            fixed = geom.buffer(0)
            if fixed.is_valid:
                result["auto_fixed"] = True
                result["fixed_geometry"] = mapping(fixed)
            else:
                result["auto_fixed"] = False

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def detect_crs_from_coordinates(
    x: float,
    y: float
) -> dict:
    """
    Detect the likely coordinate reference system from a coordinate pair
    by analyzing value ranges and patterns.

    Args:
        x: X coordinate (could be longitude, easting, etc.)
        y: Y coordinate (could be latitude, northing, etc.)

    Returns:
        List of likely CRS matches with confidence levels
    """
    candidates = []

    # WGS84 / Geographic
    if -180 <= x <= 180 and -90 <= y <= 90:
        candidates.append({
            "crs": "EPSG:4326",
            "name": "WGS 84 (Geographic)",
            "confidence": "high",
            "reason": f"Values within lon/lat range"
        })

    # Web Mercator
    if -20037508 <= x <= 20037508 and -20037508 <= y <= 20037508:
        if abs(x) > 180 or abs(y) > 90:
            candidates.append({
                "crs": "EPSG:3857",
                "name": "Web Mercator",
                "confidence": "medium",
                "reason": "Values within Web Mercator bounds"
            })

    # UTM (typical easting 100000-900000, northing 0-10000000)
    if 100000 <= x <= 900000 and 0 <= y <= 10000000:
        # Estimate UTM zone from northing
        hemisphere = "N" if y < 5000000 or y > 5000000 else "unknown"
        candidates.append({
            "crs": "UTM (zone unknown)",
            "name": "UTM Projected",
            "confidence": "medium",
            "reason": f"Easting/Northing pattern matches UTM. Provide longitude to determine zone.",
            "hint": "Use get_utm_zone with a known longitude to identify the exact zone"
        })

    # State Plane / large projected coords
    if abs(x) > 900000 or abs(y) > 10000000:
        if abs(x) < 20037508:
            candidates.append({
                "crs": "Unknown projected CRS",
                "name": "State Plane or regional projection",
                "confidence": "low",
                "reason": "Large values suggest a projected CRS but zone cannot be determined"
            })

    if not candidates:
        candidates.append({
            "crs": "Unknown",
            "name": "Cannot determine",
            "confidence": "none",
            "reason": f"Values x={x}, y={y} don't match common CRS patterns"
        })

    return {
        "success": True,
        "input": {"x": x, "y": y},
        "candidates": candidates
    }


@mcp.tool()
def find_duplicate_points(
    coordinates: list[list[float]],
    tolerance: float = 0.001
) -> dict:
    """
    Find duplicate or near-duplicate points within a coordinate list.
    Useful for cleaning survey data, GPS tracks, or point clouds.

    Args:
        coordinates: List of [x, y] coordinate pairs
        tolerance: Distance threshold to consider points as duplicates (in same units as coordinates)

    Returns:
        Duplicate groups with indices and distances
    """
    try:
        duplicates = []
        seen = set()

        for i in range(len(coordinates)):
            if i in seen:
                continue
            group = [i]
            for j in range(i + 1, len(coordinates)):
                if j in seen:
                    continue
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= tolerance:
                    group.append(j)
                    seen.add(j)

            if len(group) > 1:
                seen.add(i)
                duplicates.append({
                    "indices": group,
                    "coordinate": coordinates[i],
                    "count": len(group)
                })

        return {
            "success": True,
            "total_points": len(coordinates),
            "duplicate_groups": len(duplicates),
            "total_duplicates": sum(g["count"] - 1 for g in duplicates),
            "unique_points": len(coordinates) - sum(g["count"] - 1 for g in duplicates),
            "tolerance": tolerance,
            "groups": duplicates
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_utm_zone(longitude: float, latitude: float) -> dict:
    """
    Determine the correct UTM zone and EPSG code for a given longitude/latitude.

    Args:
        longitude: Longitude in decimal degrees (-180 to 180)
        latitude: Latitude in decimal degrees (-90 to 90)

    Returns:
        UTM zone number, hemisphere, and EPSG code
    """
    try:
        zone = int((longitude + 180) / 6) + 1
        hemisphere = "N" if latitude >= 0 else "S"

        # WGS84 UTM EPSG codes: North = 32600 + zone, South = 32700 + zone
        epsg = (32600 if hemisphere == "N" else 32700) + zone

        # NAD83 UTM (North America only, zones 1-23)
        nad83_epsg = None
        if hemisphere == "N" and 1 <= zone <= 23:
            nad83_epsg = 26900 + zone

        return {
            "success": True,
            "longitude": longitude,
            "latitude": latitude,
            "utm_zone": zone,
            "hemisphere": hemisphere,
            "wgs84_epsg": f"EPSG:{epsg}",
            "nad83_epsg": f"EPSG:{nad83_epsg}" if nad83_epsg else None,
            "zone_label": f"{zone}{hemisphere}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# ELEVATION / RASTER VALUE EXTRACTION TOOLS
# ============================================================

@mcp.tool()
def get_elevation(
    latitude: float,
    longitude: float
) -> dict:
    """
    Get the ground elevation at a specific latitude/longitude using the Open-Elevation API.
    Returns elevation in meters above sea level (AMSL).

    Args:
        latitude: Latitude in decimal degrees (WGS84)
        longitude: Longitude in decimal degrees (WGS84)

    Returns:
        Elevation in meters, data source info
    """
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
        response = httpx.get(url, timeout=15.0)
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            elev = data["results"][0]["elevation"]
            return {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elev,
                "elevation_ft": round(elev * 3.28084, 1),
                "source": "SRTM via Open-Elevation API"
            }
        else:
            return {"success": False, "error": "No elevation data available for this location"}

    except httpx.TimeoutException:
        return {"success": False, "error": "Elevation API timed out. Try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_elevation_profile(
    coordinates: list[list[float]]
) -> dict:
    """
    Get elevation values along a path defined by lat/lon coordinate pairs.
    Useful for terrain profiles, pipeline routes, and flight path planning.

    Args:
        coordinates: List of [latitude, longitude] pairs defining the path

    Returns:
        Elevation profile with min/max/avg stats, total distance, and elevation change
    """
    try:
        locations = "|".join(f"{c[0]},{c[1]}" for c in coordinates)
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        response = httpx.get(url, timeout=30.0)
        data = response.json()

        if "results" not in data:
            return {"success": False, "error": "No elevation data returned"}

        elevations = [r["elevation"] for r in data["results"]]

        # Calculate distances between points (haversine)
        distances = [0.0]
        total_dist = 0.0
        for i in range(1, len(coordinates)):
            d = _haversine_distance(
                coordinates[i-1][0], coordinates[i-1][1],
                coordinates[i][0], coordinates[i][1]
            )
            total_dist += d
            distances.append(round(total_dist, 1))

        # Calculate elevation gain/loss
        gain = 0.0
        loss = 0.0
        for i in range(1, len(elevations)):
            diff = elevations[i] - elevations[i-1]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)

        return {
            "success": True,
            "point_count": len(elevations),
            "profile": [
                {
                    "lat": coordinates[i][0],
                    "lon": coordinates[i][1],
                    "elevation_m": elevations[i],
                    "distance_m": distances[i]
                }
                for i in range(len(elevations))
            ],
            "stats": {
                "min_elevation_m": min(elevations),
                "max_elevation_m": max(elevations),
                "avg_elevation_m": round(sum(elevations) / len(elevations), 1),
                "total_elevation_gain_m": round(gain, 1),
                "total_elevation_loss_m": round(loss, 1),
                "total_distance_m": round(total_dist, 1)
            },
            "source": "SRTM via Open-Elevation API"
        }

    except httpx.TimeoutException:
        return {"success": False, "error": "Elevation API timed out. Try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# VIEWSHED / LINE-OF-SIGHT TOOLS
# ============================================================

@mcp.tool()
def line_of_sight(
    observer_lat: float,
    observer_lon: float,
    observer_height_m: float,
    target_lat: float,
    target_lon: float,
    target_height_m: float = 0.0,
    num_samples: int = 20
) -> dict:
    """
    Calculate line-of-sight visibility between an observer and target point.
    Uses terrain elevation data to determine if the view is obstructed.
    Useful for tower placement, antenna coverage, and drone operations.

    Args:
        observer_lat: Observer latitude (WGS84)
        observer_lon: Observer longitude (WGS84)
        observer_height_m: Observer height above ground (meters)
        target_lat: Target latitude (WGS84)
        target_lon: Target longitude (WGS84)
        target_height_m: Target height above ground (meters, default 0)
        num_samples: Number of terrain sample points along the path (default 20)

    Returns:
        Visibility result (visible/obstructed), obstruction details, and terrain profile
    """
    try:
        # Generate sample points along the path
        sample_coords = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            lat = observer_lat + t * (target_lat - observer_lat)
            lon = observer_lon + t * (target_lon - observer_lon)
            sample_coords.append([lat, lon])

        # Get elevation profile
        locations = "|".join(f"{c[0]},{c[1]}" for c in sample_coords)
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        response = httpx.get(url, timeout=30.0)
        data = response.json()

        if "results" not in data:
            return {"success": False, "error": "Could not fetch elevation data"}

        elevations = [r["elevation"] for r in data["results"]]

        # Observer and target absolute elevations
        obs_elev = elevations[0] + observer_height_m
        tgt_elev = elevations[-1] + target_height_m

        # Check line of sight
        total_dist = _haversine_distance(observer_lat, observer_lon, target_lat, target_lon)
        obstructions = []
        is_visible = True

        for i in range(1, num_samples - 1):
            t = i / (num_samples - 1)
            # Expected elevation on the sight line at this point
            los_elev = obs_elev + t * (tgt_elev - obs_elev)
            terrain_elev = elevations[i]

            if terrain_elev > los_elev:
                is_visible = False
                point_dist = total_dist * t
                obstructions.append({
                    "sample_index": i,
                    "lat": round(sample_coords[i][0], 6),
                    "lon": round(sample_coords[i][1], 6),
                    "terrain_elevation_m": terrain_elev,
                    "sight_line_elevation_m": round(los_elev, 1),
                    "obstruction_height_m": round(terrain_elev - los_elev, 1),
                    "distance_from_observer_m": round(point_dist, 1)
                })

        return {
            "success": True,
            "is_visible": is_visible,
            "observer": {
                "lat": observer_lat, "lon": observer_lon,
                "ground_elev_m": elevations[0],
                "total_elev_m": obs_elev
            },
            "target": {
                "lat": target_lat, "lon": target_lon,
                "ground_elev_m": elevations[-1],
                "total_elev_m": tgt_elev
            },
            "distance_m": round(total_dist, 1),
            "obstruction_count": len(obstructions),
            "obstructions": obstructions,
            "terrain_samples": len(elevations)
        }

    except httpx.TimeoutException:
        return {"success": False, "error": "Elevation API timed out. Try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# DRONE FLIGHT PLANNING TOOLS
# ============================================================

@mcp.tool()
def calculate_gsd(
    flight_altitude_m: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width_px: int,
    image_height_px: int,
    focal_length_mm: float
) -> dict:
    """
    Calculate Ground Sample Distance (GSD) for drone photogrammetry.
    GSD is the distance between adjacent pixel centers on the ground.

    Args:
        flight_altitude_m: Flight altitude above ground level (meters)
        sensor_width_mm: Camera sensor width (mm)
        sensor_height_mm: Camera sensor height (mm)
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        focal_length_mm: Lens focal length (mm)

    Returns:
        GSD in cm/px, ground footprint dimensions, and coverage area
    """
    try:
        # GSD = (sensor_dim * altitude) / (focal_length * image_dim)
        gsd_w = (sensor_width_mm * flight_altitude_m) / (focal_length_mm * image_width_px) * 100  # cm/px
        gsd_h = (sensor_height_mm * flight_altitude_m) / (focal_length_mm * image_height_px) * 100

        # Ground footprint: (mm * m) / mm = meters
        footprint_w = (sensor_width_mm * flight_altitude_m) / focal_length_mm
        footprint_h = (sensor_height_mm * flight_altitude_m) / focal_length_mm

        return {
            "success": True,
            "gsd_cm_per_px": round(max(gsd_w, gsd_h), 2),
            "gsd_width_cm_per_px": round(gsd_w, 2),
            "gsd_height_cm_per_px": round(gsd_h, 2),
            "footprint_width_m": round(footprint_w, 1),
            "footprint_height_m": round(footprint_h, 1),
            "footprint_area_sq_m": round(footprint_w * footprint_h, 1),
            "footprint_area_hectares": round(footprint_w * footprint_h / 10000, 3),
            "flight_altitude_m": flight_altitude_m,
            "quality_tier": _gsd_quality_tier(max(gsd_w, gsd_h))
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def plan_drone_grid(
    area_width_m: float,
    area_height_m: float,
    flight_altitude_m: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width_px: int,
    image_height_px: int,
    focal_length_mm: float,
    front_overlap_pct: float = 80.0,
    side_overlap_pct: float = 70.0,
    flight_speed_ms: float = 5.0,
    battery_minutes: float = 25.0
) -> dict:
    """
    Plan a drone photogrammetry grid mission. Calculates flight lines, photo count,
    total flight time, and battery requirements for a rectangular survey area.

    Args:
        area_width_m: Survey area width (meters)
        area_height_m: Survey area height (meters)
        flight_altitude_m: Flight altitude AGL (meters)
        sensor_width_mm: Camera sensor width (mm)
        sensor_height_mm: Camera sensor height (mm)
        image_width_px: Image width (pixels)
        image_height_px: Image height (pixels)
        focal_length_mm: Focal length (mm)
        front_overlap_pct: Forward overlap percentage (default 80%)
        side_overlap_pct: Side overlap percentage (default 70%)
        flight_speed_ms: Flight speed in m/s (default 5)
        battery_minutes: Available battery time in minutes (default 25)

    Returns:
        Mission plan with GSD, flight lines, photo count, time estimate, and battery assessment
    """
    try:
        # Calculate footprint: (mm * m) / mm = meters
        footprint_w = (sensor_width_mm * flight_altitude_m) / focal_length_mm
        footprint_h = (sensor_height_mm * flight_altitude_m) / focal_length_mm

        # GSD
        gsd_w = (sensor_width_mm * flight_altitude_m) / (focal_length_mm * image_width_px) * 100
        gsd_h = (sensor_height_mm * flight_altitude_m) / (focal_length_mm * image_height_px) * 100
        gsd = max(gsd_w, gsd_h)

        # Spacing between photos (forward) and between flight lines (side)
        photo_spacing = footprint_h * (1 - front_overlap_pct / 100)
        line_spacing = footprint_w * (1 - side_overlap_pct / 100)

        # Number of flight lines and photos per line
        num_lines = math.ceil(area_width_m / line_spacing) + 1
        photos_per_line = math.ceil(area_height_m / photo_spacing) + 1
        total_photos = num_lines * photos_per_line

        # Photo interval
        photo_interval_s = photo_spacing / flight_speed_ms

        # Flight distance and time
        flight_distance = num_lines * area_height_m + (num_lines - 1) * line_spacing
        flight_time_s = flight_distance / flight_speed_ms
        flight_time_min = flight_time_s / 60

        # Battery check
        batteries_needed = math.ceil(flight_time_min / battery_minutes)

        return {
            "success": True,
            "gsd_cm_per_px": round(gsd, 2),
            "quality_tier": _gsd_quality_tier(gsd),
            "footprint": {
                "width_m": round(footprint_w, 1),
                "height_m": round(footprint_h, 1),
            },
            "spacing": {
                "photo_spacing_m": round(photo_spacing, 1),
                "line_spacing_m": round(line_spacing, 1),
                "photo_interval_s": round(photo_interval_s, 1),
            },
            "mission": {
                "flight_lines": num_lines,
                "photos_per_line": photos_per_line,
                "total_photos": total_photos,
                "flight_distance_m": round(flight_distance, 0),
                "flight_time_min": round(flight_time_min, 1),
                "batteries_needed": batteries_needed,
            },
            "area": {
                "width_m": area_width_m,
                "height_m": area_height_m,
                "area_sq_m": round(area_width_m * area_height_m, 0),
                "area_hectares": round(area_width_m * area_height_m / 10000, 2),
            },
            "overlaps": {
                "front_pct": front_overlap_pct,
                "side_pct": side_overlap_pct
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def recommend_flight_altitude(
    target_gsd_cm: float,
    sensor_width_mm: float,
    image_width_px: int,
    focal_length_mm: float
) -> dict:
    """
    Calculate the required flight altitude to achieve a target GSD.
    Useful for planning drone surveys to meet accuracy specifications.

    Args:
        target_gsd_cm: Desired ground sample distance (cm/pixel)
        sensor_width_mm: Camera sensor width (mm)
        image_width_px: Image width (pixels)
        focal_length_mm: Lens focal length (mm)

    Returns:
        Required flight altitude and related parameters
    """
    try:
        # altitude = (GSD * focal_length * image_width) / (sensor_width * 100)
        altitude_m = (target_gsd_cm * focal_length_mm * image_width_px) / (sensor_width_mm * 100)

        return {
            "success": True,
            "target_gsd_cm": target_gsd_cm,
            "required_altitude_m": round(altitude_m, 1),
            "required_altitude_ft": round(altitude_m * 3.28084, 0),
            "quality_tier": _gsd_quality_tier(target_gsd_cm),
            "common_drones": _suggest_drone_for_altitude(altitude_m)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _gsd_quality_tier(gsd_cm):
    """Classify GSD into survey quality tiers."""
    if gsd_cm <= 1.0:
        return "Ultra-high resolution (sub-cm) — engineering surveys, structural inspection"
    elif gsd_cm <= 2.5:
        return "High resolution — topographic surveys, detailed mapping"
    elif gsd_cm <= 5.0:
        return "Standard resolution — site planning, volumetrics"
    elif gsd_cm <= 10.0:
        return "Moderate resolution — agriculture, large area mapping"
    else:
        return "Low resolution — reconnaissance, corridor overview"


def _suggest_drone_for_altitude(altitude_m):
    """Suggest common drones suitable for the altitude."""
    suggestions = []
    if altitude_m <= 120:
        suggestions.append("DJI Mavic 3 Enterprise (max 120m AGL)")
    if altitude_m <= 150:
        suggestions.append("DJI Matrice 350 RTK (max 150m AGL)")
    if altitude_m <= 500:
        suggestions.append("Fixed-wing (e.g., senseFly eBee X, max ~500m AGL)")
    if altitude_m > 150:
        suggestions.append("Note: Altitudes above 120m AGL may require special airspace authorization (BVLOS/SFOC)")
    return suggestions


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
