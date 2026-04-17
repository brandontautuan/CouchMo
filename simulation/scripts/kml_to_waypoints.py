#!/usr/bin/env python3
"""Convert Google Earth KML/KMZ sidewalk routes to local ENU waypoints.

Part of the CouchMo Campus Sim pipeline. Accepts a KML (or KMZ) file
containing one or more ``<LineString>`` placemarks traced in Google Earth
and emits a CSV of ``(x_m, y_m, s_m)`` rows in a local east/north tangent
plane. That CSV feeds:

* the world generator (produces a Gazebo ``.world`` with sidewalk-ribbon
  geometry from the waypoints), and
* the pure-pursuit waypoint expert used for imitation-learning demos.

This module intentionally depends only on the Python standard library so
it runs identically on Windows, macOS and Linux without a virtual env.

Typical CLI usage::

    python kml_to_waypoints.py \\
        --in  simulation/scripts/fixtures/synthetic_campus.kml \\
        --out simulation/waypoints/synthetic_campus.csv \\
        --origin-lat 37.0 --origin-lon -120.0

If ``--origin-lat`` / ``--origin-lon`` are omitted, the first parsed
coordinate is used as the origin and is reported on stderr.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

WGS84_EQUATORIAL_RADIUS_M = 6378137.0
KML_NS = "http://www.opengis.net/kml/2.2"


def _read_kml_bytes(path: Path) -> bytes:
    """Return raw KML XML bytes, transparently unwrapping a KMZ archive.

    KMZ files are plain zip archives. Per the KML 2.2 spec the "main"
    document is the first ``.kml`` entry in the archive (by convention
    ``doc.kml``). We honour that convention: the first member whose
    name ends in ``.kml`` wins.
    """
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        with zipfile.ZipFile(path, "r") as zf:
            kml_members = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_members:
                raise ValueError(f"KMZ archive {path} contains no .kml member")
            with zf.open(kml_members[0], "r") as fh:
                return fh.read()
    if suffix == ".kml":
        return path.read_bytes()
    raise ValueError(
        f"Unsupported input extension {suffix!r} for {path}; expected .kml or .kmz"
    )


def _iter_linestring_elements(root: ET.Element):
    """Yield every ``<LineString>`` element regardless of namespace.

    Matches on the local tag name so KML 2.2, KML 2.3, and no-namespace
    hand-authored fixtures all parse. Real Google Earth 7.x exports
    sometimes drift between 2.2 and 2.3, and we want both to work.
    """
    for elem in root.iter():
        tag = elem.tag
        if tag.endswith("}LineString") or tag == "LineString":
            yield elem


def _find_coordinates_child(linestring: ET.Element) -> ET.Element | None:
    for child in linestring:
        tag = child.tag
        if tag.endswith("}coordinates") or tag == "coordinates":
            return child
    return None


def _parse_coordinate_block(text: str) -> list[tuple[float, float, float]]:
    """Parse a KML ``<coordinates>`` blob.

    KML stores triples as ``lon,lat[,alt]`` separated by any whitespace
    (spaces, tabs, newlines). Altitude is optional; when absent we
    default to ``0.0`` so callers always receive a three-tuple.
    """
    out: list[tuple[float, float, float]] = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) < 2:
            raise ValueError(
                f"KML <coordinates> token {token!r} has fewer than 2 "
                f"comma-separated fields; expected 'lon,lat[,alt]'"
            )
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else 0.0
        out.append((lon, lat, alt))
    return out


def parse_kml(path: Path) -> list[tuple[float, float, float]]:
    """Parse a KML or KMZ file into a list of ``(lon, lat, alt)`` triples.

    All ``<LineString><coordinates>`` blocks are concatenated in
    document order. For v1 of the pipeline this gives Google-Earth
    traced sidewalks a single contiguous waypoint list per export.
    Altitudes are preserved but are not consumed by the downstream ENU
    projection (the tangent plane is 2D).

    Parameters
    ----------
    path:
        Filesystem path to a ``.kml`` or ``.kmz`` file.

    Returns
    -------
    list[tuple[float, float, float]]
        Ordered ``(lon_deg, lat_deg, alt_m)`` triples.
    """
    data = _read_kml_bytes(path)
    root = ET.fromstring(data)
    waypoints: list[tuple[float, float, float]] = []
    for linestring in _iter_linestring_elements(root):
        coords_elem = _find_coordinates_child(linestring)
        if coords_elem is None or coords_elem.text is None:
            continue
        waypoints.extend(_parse_coordinate_block(coords_elem.text))
    return waypoints


def latlon_to_enu(
    lat: float,
    lon: float,
    *,
    origin_lat: float,
    origin_lon: float,
) -> tuple[float, float]:
    """Project WGS84 lat/lon to local east/north metres (tangent plane).

    Uses the equirectangular / small-angle tangent-plane approximation::

        x_east  = R * cos(origin_lat) * (lon - origin_lon)
        y_north = R * (lat - origin_lat)

    with ``R`` the WGS84 equatorial radius. Accurate to well under a
    metre for campus-scale extents (<5 km from the origin) which is all
    we need for sim world generation and waypoint following. For
    regional-scale routes (>10 km) prefer a proper UTM or geodesic
    projection (e.g. ``pyproj``).
    """
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x_east = WGS84_EQUATORIAL_RADIUS_M * math.cos(origin_lat_rad) * (
        lon_rad - origin_lon_rad
    )
    y_north = WGS84_EQUATORIAL_RADIUS_M * (lat_rad - origin_lat_rad)
    return (x_east, y_north)


def _triples_to_enu_rows(
    triples: list[tuple[float, float, float]],
    *,
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float, float]]:
    """Project parsed (lon, lat, alt) triples to (x_m, y_m, s_m) rows.

    Consecutive duplicate coordinates (which would contribute a
    zero-length segment) are dropped so the cumulative arc-length
    ``s_m`` is strictly monotonically increasing after the first
    waypoint. Pure projection — no I/O.
    """
    rows: list[tuple[float, float, float]] = []
    prev_xy: tuple[float, float] | None = None
    cumulative = 0.0
    for lon, lat, _alt in triples:
        x, y = latlon_to_enu(
            lat, lon, origin_lat=origin_lat, origin_lon=origin_lon
        )
        if prev_xy is None:
            rows.append((x, y, 0.0))
            prev_xy = (x, y)
            continue
        dx = x - prev_xy[0]
        dy = y - prev_xy[1]
        step = math.hypot(dx, dy)
        if step == 0.0:
            continue
        cumulative += step
        rows.append((x, y, cumulative))
        prev_xy = (x, y)
    return rows


def convert(
    kml_path: Path,
    *,
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float, float]]:
    """Parse a KML/KMZ file and project it to ``(x_m, y_m, s_m)`` rows.

    Consecutive duplicate coordinates (which would contribute a
    zero-length segment) are dropped so the cumulative arc-length
    ``s_m`` is strictly monotonically increasing after the first
    waypoint.

    Parameters
    ----------
    kml_path:
        Path to the ``.kml`` or ``.kmz`` file.
    origin_lat, origin_lon:
        Tangent-plane origin in WGS84 degrees. The returned ``(x, y)``
        is zero at this origin.

    Returns
    -------
    list[tuple[float, float, float]]
        One row per waypoint: ``(x_m_east, y_m_north, s_m_cumulative)``.
    """
    triples = parse_kml(kml_path)
    return _triples_to_enu_rows(
        triples, origin_lat=origin_lat, origin_lon=origin_lon
    )


def _write_csv(rows: list[tuple[float, float, float]], out_path: Path) -> None:
    """Write waypoint rows to CSV with a ``x_m,y_m,s_m`` header.

    Uses ``newline=""`` so Python's csv module owns line termination;
    this prevents the blank-line-every-row bug on Windows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x_m", "y_m", "s_m"])
        for x, y, s in rows:
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{s:.6f}"])


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert KML/KMZ routes to local ENU waypoint CSV.",
    )
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        required=True,
        help="Input .kml or .kmz file.",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Output CSV path (parent dirs are created).",
    )
    p.add_argument(
        "--origin-lat",
        type=float,
        default=None,
        help="Tangent-plane origin latitude (deg). Defaults to first KML coord.",
    )
    p.add_argument(
        "--origin-lon",
        type=float,
        default=None,
        help="Tangent-plane origin longitude (deg). Defaults to first KML coord.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    in_path: Path = args.in_path
    out_path: Path = args.out_path

    triples = parse_kml(in_path)
    if not triples:
        print(
            f"error: {in_path} produced 0 waypoints — no <LineString> with "
            f"<coordinates> found (check your Google Earth export)",
            file=sys.stderr,
        )
        return 2

    origin_lat = args.origin_lat
    origin_lon = args.origin_lon
    if origin_lat is None or origin_lon is None:
        first_lon, first_lat, _ = triples[0]
        if origin_lat is None:
            origin_lat = first_lat
        if origin_lon is None:
            origin_lon = first_lon
        print(
            f"kml_to_waypoints: using origin from first coord: "
            f"lat={origin_lat}, lon={origin_lon}",
            file=sys.stderr,
        )

    rows = _triples_to_enu_rows(
        triples, origin_lat=origin_lat, origin_lon=origin_lon
    )
    _write_csv(rows, out_path)
    print(
        f"kml_to_waypoints: wrote {len(rows)} waypoints to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
