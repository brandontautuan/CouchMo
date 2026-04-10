#!/usr/bin/env python3
"""
generate_flc_map.py
===================
Downloads OpenStreetMap building / path / parking data for
Folsom Lake College and rasterizes it into a Nav2-ready
occupancy-grid (.pgm + .yaml).

Dependencies:
    pip install requests Pillow numpy

Output (relative to repo root):
    src/couchmo_nav/maps/flc_campus.pgm
    src/couchmo_nav/maps/flc_campus.yaml

Usage:
    cd /path/to/couchmo
    python3 scripts/generate_flc_map.py
"""

import json
import math
import os
import time

import numpy as np
import requests
from PIL import Image, ImageDraw

# ── Campus bounding box (decimal degrees) ───────────────────────────────────
# Tightly fitted to FLC main campus + all parking lots.
# Excludes surrounding Folsom neighborhoods.
# Source: OpenStreetMap / flc.losrios.edu campus map
SOUTH =  38.6762
NORTH =  38.6838
WEST  = -121.1525
EAST  = -121.1428

# Map resolution.
# 0.10 m/px → ~50 MB on disk  (practical default for Docker)
# 0.05 m/px → ~200 MB on disk (pass --full-res to enable)
import sys
RESOLUTION = 0.05 if "--full-res" in sys.argv else 0.10   # metres per pixel

# ── PGM pixel values (Nav2 / costmap_2d convention) ─────────────────────────
#   pixel / 255 ≥ occupied_thresh  →  lethal obstacle
#   pixel / 255 ≤ free_thresh      →  free
#   in between                      →  unknown
FREE     = 254
UNKNOWN  = 205
OCCUPIED =   0

# ── Real-world widths for path/road rendering ────────────────────────────────
FOOTWAY_W_M  = 3.0   # pedestrian path / steps
ROAD_W_M     = 6.0   # campus service / residential road
BARRIER_W_M  = 0.4   # wall / fence

# ── Coordinate maths ─────────────────────────────────────────────────────────
_mid_lat   = (NORTH + SOUTH) / 2
LAT_M      = 111_139.0                              # metres per degree latitude
LON_M      = LAT_M * math.cos(math.radians(_mid_lat))  # metres per degree longitude

MAP_W_M    = (EAST  - WEST ) * LON_M
MAP_H_M    = (NORTH - SOUTH) * LAT_M
MAP_W_PX   = int(math.ceil(MAP_W_M / RESOLUTION))
MAP_H_PX   = int(math.ceil(MAP_H_M / RESOLUTION))

print(f"Bounding box:  {MAP_W_M:.0f} m × {MAP_H_M:.0f} m")
print(f"Image size:    {MAP_W_PX} × {MAP_H_PX} px  "
      f"({MAP_W_PX * MAP_H_PX / 1e6:.1f} Mpx,  "
      f"≈{MAP_W_PX * MAP_H_PX / 1e6:.0f} MB on disk)")


def ll_to_px(lat: float, lon: float) -> tuple[int, int]:
    """Lat/lon → (col, row) in the PGM image.
    Image row 0 = north edge; Y increases downward (image convention).
    Map YAML origin = SW corner (bottom-left), so row is flipped.
    """
    x_m = (lon - WEST ) * LON_M
    y_m = (lat - SOUTH) * LAT_M
    col = int(x_m / RESOLUTION)
    row = MAP_H_PX - 1 - int(y_m / RESOLUTION)
    return col, row


def m_to_px(metres: float) -> int:
    return max(1, int(round(metres / RESOLUTION)))


# ── Overpass API query ────────────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# bbox in Overpass is (south, west, north, east)
_BB = f"{SOUTH},{WEST},{NORTH},{EAST}"

QUERY = f"""
[out:json][timeout:90];
(
  way["building"]({_BB});
  way["amenity"="parking"]({_BB});
  way["parking"~"surface|multi-storey|lane"]({_BB});
  way["leisure"~"park|garden|recreation_ground"]({_BB});
  way["landuse"~"grass|meadow|recreation_ground|education|university"]({_BB});
  way["highway"~"footway|path|pedestrian|steps|cycleway|service|residential|primary|secondary|tertiary|unclassified|living_street|track"]({_BB});
  way["barrier"~"wall|fence|retaining_wall|guard_rail"]({_BB});
  way["man_made"="wall"]({_BB});
);
out body;
>;
out skel qt;
"""


def fetch_osm(cache_path: str) -> dict:
    """Download OSM data; cache to avoid repeated Overpass calls."""
    if os.path.exists(cache_path):
        print(f"  Using cached OSM data: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print("  Querying Overpass API …")
    for attempt in range(3):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": QUERY},
                timeout=120,
                headers={"User-Agent": "couchmo-map-generator/1.0"},
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate-limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            print(f"  Cached → {cache_path}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    raise RuntimeError("Failed to download OSM data after 3 attempts")


# ── Parse OSM elements ────────────────────────────────────────────────────────
def parse_ways(data: dict) -> list[tuple[dict, list[tuple[int, int]]]]:
    """Return list of (tags, pixel_coords) for every way in the response."""
    nodes: dict[int, tuple[float, float]] = {}
    for el in data["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])

    ways: list[tuple[dict, list[tuple[int, int]]]] = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        coords = [
            ll_to_px(nodes[nid][0], nodes[nid][1])
            for nid in el.get("nodes", [])
            if nid in nodes
        ]
        if len(coords) >= 2:
            ways.append((el.get("tags", {}), coords))
    return ways


# ── Rendering ─────────────────────────────────────────────────────────────────
def render(ways: list[tuple[dict, list[tuple[int, int]]]]) -> Image.Image:
    img  = Image.new("L", (MAP_W_PX, MAP_H_PX), UNKNOWN)
    draw = ImageDraw.Draw(img)

    # ── Pass 1: free-space areas (painted before obstacles so buildings win) ──

    for tags, coords in ways:
        hw   = tags.get("highway", "")
        lnd  = tags.get("landuse", "")
        lei  = tags.get("leisure", "")
        amen = tags.get("amenity", "")
        park = tags.get("parking", "")

        # Campuses / educational grounds → assume freely navigable
        if lnd in ("education", "university", "grass", "meadow",
                   "recreation_ground"):
            if len(coords) >= 3:
                draw.polygon(coords, fill=FREE)

        if lei in ("park", "garden", "recreation_ground"):
            if len(coords) >= 3:
                draw.polygon(coords, fill=FREE)

        # Parking lots
        if amen == "parking" or park in ("surface", "multi-storey", "lane"):
            if len(coords) >= 3:
                draw.polygon(coords, fill=FREE)

        # Pedestrian paths / footways
        if hw in ("footway", "path", "pedestrian", "steps", "cycleway"):
            draw.line(coords, fill=FREE, width=m_to_px(FOOTWAY_W_M))

        # Campus roads / service drives
        if hw in ("service", "residential", "living_street", "track",
                  "unclassified", "tertiary", "secondary", "primary"):
            draw.line(coords, fill=FREE, width=m_to_px(ROAD_W_M))

    # ── Pass 2: obstacles (painted on top of free space) ─────────────────────

    for tags, coords in ways:
        # Buildings — filled polygon + outline to close any rendering gaps
        if tags.get("building"):
            if len(coords) >= 3:
                draw.polygon(coords, fill=OCCUPIED)
                draw.line(coords + [coords[0]], fill=OCCUPIED, width=2)

        # Walls / fences
        if tags.get("barrier") in ("wall", "fence", "retaining_wall", "guard_rail"):
            draw.line(coords, fill=OCCUPIED, width=m_to_px(BARRIER_W_M))
        if tags.get("man_made") == "wall":
            draw.line(coords, fill=OCCUPIED, width=m_to_px(BARRIER_W_M))

    return img


# ── File writers ──────────────────────────────────────────────────────────────
def save_pgm(img: Image.Image, path: str) -> None:
    """Write binary P5 PGM — more compact and universally readable than PIL default."""
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    header = f"P5\n{w} {h}\n255\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path}  ({size_mb:.1f} MB)")


def save_yaml(pgm_filename: str, yaml_path: str) -> None:
    """Write the Nav2 map_server .yaml sidecar.

    origin: pose of the bottom-left pixel in the ROS 'map' frame.
    Setting to [0, 0, 0] means the SW campus corner = map frame origin.
    Adjust x/y if your Gazebo world uses a different reference point.
    """
    content = (
        f"image: {pgm_filename}\n"
        f"resolution: {RESOLUTION}\n"
        f"origin: [0.0, 0.0, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  Saved {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    maps_dir   = os.path.join(repo_root, "src", "couchmo_nav", "maps")
    # Use separate cache for each resolution so bbox changes don't poison it
    cache_path = os.path.join(maps_dir, "_osm_cache.json")
    pgm_path   = os.path.join(maps_dir, "flc_campus.pgm")
    yaml_path  = os.path.join(maps_dir, "flc_campus.yaml")

    os.makedirs(maps_dir, exist_ok=True)

    # ── 1. Download OSM data ───────────────────────────────────────────────
    print("\n[1/3] Fetching OSM data …")
    data = fetch_osm(cache_path)
    n_elements = len(data.get("elements", []))
    print(f"  {n_elements} OSM elements received")
    if n_elements < 10:
        print("  WARNING: very few OSM elements — campus may not be mapped in OSM yet.")
        print("  The map will be mostly UNKNOWN; buildings from OSM will still appear.")

    # ── 2. Rasterize ──────────────────────────────────────────────────────
    print("\n[2/3] Rasterizing occupancy grid …")
    ways = parse_ways(data)
    print(f"  {len(ways)} ways to render")

    n_buildings  = sum(1 for t, _ in ways if t.get("building"))
    n_paths      = sum(1 for t, _ in ways if t.get("highway"))
    n_parking    = sum(1 for t, _ in ways
                       if t.get("amenity") == "parking" or t.get("parking"))
    print(f"  buildings={n_buildings}, paths={n_paths}, parking={n_parking}")

    img = render(ways)

    # ── 3. Save ────────────────────────────────────────────────────────────
    print("\n[3/3] Saving map files …")
    save_pgm(img, pgm_path)
    save_yaml("flc_campus.pgm", yaml_path)

    # ── Summary ────────────────────────────────────────────────────────────
    container_yaml = ("/ros2_ws/install/couchmo_nav/share/couchmo_nav"
                      "/maps/flc_campus.yaml")
    print(f"""
Done!
─────────────────────────────────────────────────────────
Map:  {MAP_W_M:.0f} m × {MAP_H_M:.0f} m  at {RESOLUTION} m/px
SW corner (map origin): {SOUTH}°N, {WEST}°E

To activate in Nav2, set in nav2_params.yaml line 214:
  yaml_filename: "{container_yaml}"

Or pass at launch time:
  ros2 launch nav2_bringup localization_launch.py \\
      map:={container_yaml} \\
      use_sim_time:=true

To use as a SLAM prior (keeps online mapping active):
  Add to slam_toolbox section of nav2_params.yaml:
    map_file_name: "{container_yaml}"
    map_start_at_dock: false
─────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
