#!/usr/bin/env python3
"""Rebuild the pinned Mount Tamalpais scenario from AWS Terrain Tiles.

This development-only command downloads two Terrarium PNGs, verifies their
bytes, decodes elevation, and writes deterministic JSON plus a hash sidecar.
Ordinary package use reads the checked-in JSON and performs no network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import urllib.request
import zlib
from pathlib import Path

ZOOM = 13
TILE_X = 1306
TILES = (
    (
        3161,
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/13/1306/3161.png",
        "f4c9f7170e8f247f096febb316ca405c959183538e4dc3582fd145c87b9dc4db",
    ),
    (
        3162,
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/13/1306/3162.png",
        "414a9017e59684c4dd686c529d7b1574293a40957d85f014a534a19cfe9c8d9f",
    ),
)
REGISTRY_URL = "https://registry.opendata.aws/terrain-tiles/"
ATTRIBUTION_URL = "https://github.com/tilezen/joerd/blob/master/docs/attribution.md"
OUTPUT = Path(__file__).parents[1] / "src/path_planning_ode/data/mount_tamalpais.json"
MANIFEST = OUTPUT.with_suffix(".provenance.json")
WEB_MERCATOR_RADIUS_M = 6_378_137.0
WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_rgb(data: bytes) -> list[list[tuple[int, int, int]]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("input is not PNG")
    pos, compressed = 8, bytearray()
    width = height = colour = depth = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind, chunk = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, colour, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if (depth, colour, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("expected non-interlaced, 8-bit RGB PNG")
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG has no IHDR")
    raw = zlib.decompress(bytes(compressed))
    stride, rows, previous = width * 3, [], bytearray(width * 3)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        scan = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        for i in range(stride):
            left = scan[i - 3] if i >= 3 else 0
            up = previous[i]
            upper_left = previous[i - 3] if i >= 3 else 0
            if filter_type == 1:
                scan[i] = (scan[i] + left) & 255
            elif filter_type == 2:
                scan[i] = (scan[i] + up) & 255
            elif filter_type == 3:
                scan[i] = (scan[i] + ((left + up) // 2)) & 255
            elif filter_type == 4:
                p = left + up - upper_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upper_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else upper_left
                scan[i] = (scan[i] + predictor) & 255
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}")
        rows.append([tuple(scan[i : i + 3]) for i in range(0, stride, 3)])
        previous = scan
    return rows


def _canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode()


def build(downloads: dict[int, bytes]) -> dict:
    decoded = {tile_y: _png_rgb(data) for tile_y, data in downloads.items()}
    mosaic = decoded[3161] + decoded[3162]
    # 129 raw pixels centred on the summit, sampled every two pixels to 65 x 65.
    pixel_rows = range(217, 346, 2)
    pixel_cols = range(0, 129, 2)
    north_to_south = [
        [
            mosaic[row][col][0] * 256 + mosaic[row][col][1] + mosaic[row][col][2] / 256 - 32768
            for col in pixel_cols
        ]
        for row in pixel_rows
    ]
    elevation = list(reversed(north_to_south))
    n = 2**ZOOM
    world_pixels = 256 * n
    x_pixels = [TILE_X * 256 + col + 0.5 for col in pixel_cols]
    y_pixels_north_to_south = [3161 * 256 + row + 0.5 for row in pixel_rows]
    longitudes = [2 * math.pi * pixel / world_pixels - math.pi for pixel in x_pixels]
    latitudes_north_to_south = [
        math.atan(math.sinh(math.pi - 2 * math.pi * pixel / world_pixels))
        for pixel in y_pixels_north_to_south
    ]
    latitudes = list(reversed(latitudes_north_to_south))
    centre_lat = (latitudes[0] + latitudes[-1]) / 2
    centre_lon = (longitudes[0] + longitudes[-1]) / 2
    centre_denominator = 1 - WGS84_ECCENTRICITY_SQUARED * math.sin(centre_lat) ** 2
    centre_prime_vertical_radius = WEB_MERCATOR_RADIUS_M / math.sqrt(centre_denominator)
    centre_meridional_radius = (
        WEB_MERCATOR_RADIUS_M * (1 - WGS84_ECCENTRICITY_SQUARED) / centre_denominator**1.5
    )
    x_axis = [
        round(centre_prime_vertical_radius * math.cos(centre_lat) * (lon - longitudes[0]), 9)
        for lon in longitudes
    ]
    y_axis = [round(centre_meridional_radius * (lat - latitudes[0]), 9) for lat in latitudes]
    scale_errors = []
    for latitude in latitudes:
        denominator = 1 - WGS84_ECCENTRICITY_SQUARED * math.sin(latitude) ** 2
        prime_vertical_radius = WEB_MERCATOR_RADIUS_M / math.sqrt(denominator)
        meridional_radius = (
            WEB_MERCATOR_RADIUS_M * (1 - WGS84_ECCENTRICITY_SQUARED) / denominator**1.5
        )
        scale_errors.extend(
            [
                abs(
                    centre_prime_vertical_radius
                    * math.cos(centre_lat)
                    / (prime_vertical_radius * math.cos(latitude))
                    - 1
                ),
                abs(centre_meridional_radius / meridional_radius - 1),
            ]
        )
    # Isotropic illustrative travel cost based on local slope magnitude.
    log_s = []
    for row in range(65):
        values = []
        for col in range(65):
            left, right = elevation[row][max(0, col - 1)], elevation[row][min(64, col + 1)]
            down, up = elevation[max(0, row - 1)][col], elevation[min(64, row + 1)][col]
            dx = x_axis[min(64, col + 1)] - x_axis[max(0, col - 1)]
            dy = y_axis[min(64, row + 1)] - y_axis[max(0, row - 1)]
            slope = math.hypot((right - left) / dx, (up - down) / dy)
            values.append(round(math.log(0.8) + 2.4 * slope * slope, 12))
        log_s.append(values)
    world = 2 * math.pi * WEB_MERCATOR_RADIUS_M
    west_global_px = x_pixels[0]
    east_global_px = x_pixels[-1]
    north_global_px = y_pixels_north_to_south[0]
    south_global_px = y_pixels_north_to_south[-1]
    mercator_bounds = [
        west_global_px / (256 * n) * world - world / 2,
        world / 2 - south_global_px / (256 * n) * world,
        east_global_px / (256 * n) * world - world / 2,
        world / 2 - north_global_px / (256 * n) * world,
    ]
    raw_sources = [
        {"url": url, "sha256": _sha256(downloads[tile_y]), "tile": [ZOOM, TILE_X, tile_y]}
        for tile_y, url, _ in TILES
    ]
    payload = {
        "version": 2,
        "name": "Mount Tamalpais (AWS Terrain Tiles crop)",
        "bounds_m": [0.0, 0.0, x_axis[-1], y_axis[-1]],
        "start_m": [x_axis[4], y_axis[6]],
        "goal_m": [x_axis[60], y_axis[57]],
        "field_x_m": x_axis,
        "field_y_m": y_axis,
        "elevation_m": elevation,
        "model_base_log_slowness": log_s,
        "barriers_geojson": [],
        "illustrative_barriers_geojson": [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [round(0.47 * x_axis[-1], 9), round(0.39 * y_axis[-1], 9)],
                        [round(0.53 * x_axis[-1], 9), round(0.39 * y_axis[-1], 9)],
                        [round(0.53 * x_axis[-1], 9), round(0.61 * y_axis[-1], 9)],
                        [round(0.47 * x_axis[-1], 9), round(0.61 * y_axis[-1], 9)],
                        [round(0.47 * x_axis[-1], 9), round(0.39 * y_axis[-1], 9)],
                    ]
                ],
            }
        ],
        "provenance": {
            "dataset": "Mapzen Terrain Tiles on AWS Open Data",
            "registry_url": REGISTRY_URL,
            "raw_sources": raw_sources,
            "accessed": "2026-09-15",
            "source_format": "Terrarium PNG; elevation_m = R*256 + G + B/256 - 32768",
            "source_crs": "EPSG:3857 (Web Mercator metres)",
            "source_crop_bounds_epsg3857_m": [round(v, 6) for v in mercator_bounds],
            "local_projection": (
                "local WGS84 ellipsoidal equirectangular approximation; "
                "x=N(latitude_center)*cos(latitude_center)*(longitude-longitude_west), "
                "y=M(latitude_center)*(latitude-latitude_south), axes east/north in metres"
            ),
            "local_projection_center_lon_lat_deg": [
                round(math.degrees(centre_lon), 9),
                round(math.degrees(centre_lat), 9),
            ],
            "local_projection_max_horizontal_scale_error_fraction": round(max(scale_errors), 12),
            "local_projection_scale_error_reference": "WGS84 ellipsoid",
            "processing": {
                "raw_crop_pixels": [129, 129],
                "pixel_stride": 2,
                "output_samples": [65, 65],
                "resampling": "point sample; no interpolation",
            },
            "attribution": (
                "Mapzen; United States 3DEP and global GMTED2010/SRTM terrain data "
                "courtesy of the U.S. Geological Survey."
            ),
            "attribution_requirements_url": ATTRIBUTION_URL,
        },
        "metadata": {
            "kind": "observed_elevation_with_illustrative_cost_model",
            "source_resolution": [65, 65],
            "cost_units": "seconds per horizontal metre",
            "cost_model": (
                "static isotropic cost exp(bicubic(log_slowness)); log(c)=log(0.8)+2.4*slope^2"
            ),
            "model_assumptions": (
                "Travel cost and optional closure polygons are illustrative assumptions, "
                "separate from observed elevation."
            ),
            "contrast": 1.0,
            "barriers_enabled": False,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify", action="store_true", help="verify checked-in outputs instead of rewriting"
    )
    parser.add_argument("--raw-dir", type=Path, help="read pinned PNG names from a local directory")
    args = parser.parse_args()
    downloads = {}
    for tile_y, url, expected in TILES:
        name = f"{ZOOM}-{TILE_X}-{tile_y}.png"
        data = (
            (args.raw_dir / name).read_bytes()
            if args.raw_dir
            else urllib.request.urlopen(url, timeout=30).read()
        )
        if expected and _sha256(data) != expected:
            raise SystemExit(f"raw SHA-256 mismatch for {url}")
        downloads[tile_y] = data
    payload = build(downloads)
    output_bytes = _canonical_bytes(payload)
    manifest = {
        "processed_file": OUTPUT.name,
        "processed_sha256": _sha256(output_bytes),
        "raw_sha256": {str(tile_y): _sha256(downloads[tile_y]) for tile_y, _, _ in TILES},
    }
    manifest_bytes = _canonical_bytes(manifest)
    if args.verify:
        if OUTPUT.read_bytes() != output_bytes or MANIFEST.read_bytes() != manifest_bytes:
            raise SystemExit("checked-in terrain data does not match deterministic preprocessing")
        print(f"verified {OUTPUT} ({manifest['processed_sha256']})")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(output_bytes)
        MANIFEST.write_bytes(manifest_bytes)
        print(f"wrote {OUTPUT} ({manifest['processed_sha256']})")


if __name__ == "__main__":
    main()
