# Geospatial Tools MCP Server

A comprehensive Model Context Protocol (MCP) server providing 15 geospatial utility tools for AI agents. Built for surveyors, GIS analysts, drone operators, and developers working with spatial data.

## Tools (15)

### Coordinate Transformation (5)
| Tool | Description |
|------|-------------|
| `transform_coordinates` | Transform single or batch coordinates between CRS (EPSG codes) |
| `batch_transform_coordinates` | High-volume vectorized coordinate transforms |
| `transform_geojson` | Transform GeoJSON geometries to different CRS |
| `get_crs_info` | Get details about any CRS by EPSG code |
| `list_common_crs` | Quick reference list of common EPSG codes |

### Spatial Data Validation (4)
| Tool | Description |
|------|-------------|
| `validate_geojson` | Check topology, self-intersections, and auto-repair geometries |
| `detect_crs_from_coordinates` | Detect likely CRS from coordinate value patterns |
| `find_duplicate_points` | Find near-duplicate points within a coordinate list |
| `get_utm_zone` | Determine correct UTM zone and EPSG for any lon/lat |

### Elevation & Terrain (2)
| Tool | Description |
|------|-------------|
| `get_elevation` | Get ground elevation (meters) at a lat/lon point |
| `get_elevation_profile` | Elevation profile along a path with gain/loss stats |

### Viewshed Analysis (1)
| Tool | Description |
|------|-------------|
| `line_of_sight` | Check visibility between observer and target using terrain data |

### Drone Flight Planning (3)
| Tool | Description |
|------|-------------|
| `calculate_gsd` | Calculate Ground Sample Distance from flight parameters |
| `plan_drone_grid` | Full photogrammetry grid mission plan with photo count, time, batteries |
| `recommend_flight_altitude` | Calculate required altitude for a target GSD |

## Installation

```bash
cd mcp-servers/geospatial-tools
pip install mcp pyproj shapely httpx
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "geospatial-tools": {
      "command": "python",
      "args": ["/full/path/to/server.py"]
    }
  }
}
```

## License

MIT
