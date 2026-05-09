"""
Data loading utilities for UAV fire coverage environments.

Supported file formats:
  - Circle center: CSV with columns [circle_id, center_x, center_y, radius_m, diameter_m]
      where center_x = longitude (°E) and center_y = latitude (°N).
      ``radius_m`` may be omitted when ``diameter_m`` is present (radius = diameter / 2).
      Legacy format [latitude, longitude, radius_m] is also accepted.
  - Fire points: ESRI Shapefile (.shp) **or** CSV with columns [latitude, longitude]
  - Elevation map: GeoTIFF (.tif) — requires rasterio (optional)

If the real data files are unavailable, call the ``generate_sample_*`` helpers
to obtain synthetic datasets that match the expected format.
"""

import os
import csv
import math
import warnings
import numpy as np
import zipfile
import tempfile

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def latlon_to_local(lat, lon, lat_center, lon_center):
    """Convert WGS-84 (lat, lon) to a local East-North metric frame (metres).

    Parameters
    ----------
    lat, lon : float  — point to convert
    lat_center, lon_center : float  — origin of the local frame

    Returns
    -------
    (east_m, north_m) : float, float
    """
    R = 111_000.0  # metres per degree of latitude (approximate)
    north_m = (lat - lat_center) * R
    east_m  = (lon - lon_center) * R * math.cos(math.radians(lat_center))
    return float(east_m), float(north_m)


def local_to_latlon(east_m, north_m, lat_center, lon_center):
    """Inverse of :func:`latlon_to_local`."""
    R = 111_000.0
    lat = lat_center + north_m / R
    lon = lon_center + east_m / (R * math.cos(math.radians(lat_center)))
    return float(lat), float(lon)


def infer_utm_epsg(lon_center, lat_center):
    """Infer UTM EPSG code from WGS84 center coordinates."""
    zone = int((float(lon_center) + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if float(lat_center) >= 0 else 32700
    return base + zone


def local_offsets_from_projected_xy(xy, lat_center, lon_center):
    """Convert projected absolute XY points to local XY (meters) around center.

    The center is transformed from WGS84 into an inferred local UTM CRS and then
    subtracted from all projected XY points.
    """
    arr = np.asarray(xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        from pyproj import Transformer
        epsg = infer_utm_epsg(lon_center, lat_center)
        to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        cx, cy = to_proj.transform(float(lon_center), float(lat_center))
        return (arr - np.array([cx, cy], dtype=np.float64)[None, :]).astype(np.float32)
    except Exception:
        # Last-resort fallback to keep points numerically local if CRS toolchain
        # is unavailable; center-relative path above remains preferred.
        warnings.warn(
            "Falling back to median-centered projected coordinates because center CRS "
            "transformation failed; local origin may not match circle center exactly. "
            "Install pyproj for accurate coordinate transformation: pip install pyproj",
            UserWarning,
        )
        centre = np.array([np.nanmedian(arr[:, 0]), np.nanmedian(arr[:, 1])], dtype=np.float64)
        return (arr - centre[None, :]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_circle_center_csv(filepath, circle_id=None):
    """Load circle centre from a CSV file.

    Accepts two column-name conventions (header names are case-insensitive and
    stripped of whitespace):

    **New format** (output of the greedy-circle clustering script):
        circle_id, center_x, center_y, radius_m, diameter_m

        ``center_x`` is the longitude (°E) and ``center_y`` is the latitude (°N).
        ``radius_m`` may be omitted when ``diameter_m`` is present; in that case
        ``radius_m`` is computed as ``diameter_m / 2``.
        A bare ``radius`` column (no ``_m`` suffix) is also accepted as an alias
        for ``radius_m``.

    **Legacy format**:
        latitude, longitude, radius_m

    If none of the above radius columns are present, a default of 5 000 m is
    assumed.  Pass an explicit ``radius_m`` column to avoid this silent default.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    circle_id : int or str, optional
        When the CSV contains multiple rows (one per circle), use this value to
        select a specific row by its ``circle_id`` column.  If *None*, the
        first row is used.

    Returns
    -------
    (lat_center, lon_center, radius_m) : tuple[float, float, float]
    """
    # Try common encodings so the file works on both Windows (GBK) and Linux
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            with open(filepath, newline='', encoding=encoding) as f:
                rows = list(csv.DictReader(f))
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"Cannot decode CSV with any supported encoding: {filepath}")

    if not rows:
        raise ValueError(f"Empty CSV: {filepath}")

    # Normalise column names (strip whitespace, lowercase)
    norm_rows = [{k.strip().lower(): v.strip() for k, v in r.items()} for r in rows]

    # Select the target row
    if circle_id is not None:
        target_id = str(circle_id).strip()
        matched = [r for r in norm_rows if r.get('circle_id', '') == target_id]
        if not matched:
            available = [r.get('circle_id', '') for r in norm_rows]
            raise ValueError(
                f"circle_id '{circle_id}' not found in {filepath}. "
                f"Available IDs: {available}"
            )
        row = matched[0]
    else:
        row = norm_rows[0]

    # ── Detect column-name convention ────────────────────────────────────────
    if 'center_x' in row and 'center_y' in row:
        # New format: center_x = longitude, center_y = latitude
        lon = float(row['center_x'])
        lat = float(row['center_y'])
    elif 'latitude' in row and 'longitude' in row:
        # Legacy format
        lat = float(row['latitude'])
        lon = float(row['longitude'])
    else:
        raise ValueError(
            f"CSV {filepath} has unrecognised columns: {list(row.keys())}.\n"
            "Expected either 'center_x'/'center_y' (new format) "
            "or 'latitude'/'longitude' (legacy format)."
        )

    # ── Resolve radius ────────────────────────────────────────────────────────
    if 'radius_m' in row and row['radius_m'] != '':
        radius = float(row['radius_m'])
    elif 'diameter_m' in row and row['diameter_m'] != '':
        radius = float(row['diameter_m']) / 2.0
    elif 'radius' in row and row['radius'] != '':
        radius = float(row['radius'])
    else:
        radius = 5000.0  # default fallback

    return lat, lon, radius


def load_fire_points_csv(filepath, lat_center, lon_center):
    """Load fire points from a CSV file and return local-metric coordinates.

    The CSV must contain columns ``latitude`` and ``longitude``
    (case-insensitive).

    Returns
    -------
    points : np.ndarray, shape (N, 2), dtype float32
        Each row is (east_m, north_m) relative to (lat_center, lon_center).
    """
    points = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_n = {k.strip().lower(): v for k, v in row.items()}
            lat = float(row_n['latitude'])
            lon = float(row_n['longitude'])
            e, n = latlon_to_local(lat, lon, lat_center, lon_center)
            points.append([e, n])
    return np.array(points, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Shapefile loader (optional — requires pyshp / shapefile)
# ──────────────────────────────────────────────────────────────────────────────

def load_fire_points_shp(filepath, lat_center, lon_center):
    """Load fire points from an ESRI Shapefile.

    Requires the ``shapefile`` package (``pip install pyshp``).
    Falls back to a CSV with the same basename if the import fails.

    Returns
    -------
    points : np.ndarray, shape (N, 2), dtype float32
    """
    # Preferred path: geopandas can read CRS and safely transform to WGS84.
    try:
        import geopandas as gpd
        from pyproj import Transformer
        gdf = gpd.read_file(filepath)
        if gdf.empty:
            return np.zeros((0, 2), dtype=np.float32)
        gdf = gdf[gdf.geometry.notnull()].copy()
        if gdf.empty:
            return np.zeros((0, 2), dtype=np.float32)
        if gdf.crs is None:
            # If values look projected (meters), convert center into inferred UTM
            # and subtract center to build local coordinates.
            xy = np.array([(geom.x, geom.y) for geom in gdf.geometry], dtype=np.float64)
            x_abs = np.nanmedian(np.abs(xy[:, 0]))
            y_abs = np.nanmedian(np.abs(xy[:, 1]))
            if x_abs > 1e4 or y_abs > 1e4:
                return local_offsets_from_projected_xy(xy, lat_center, lon_center)
            # Otherwise assume lon/lat
            lon = xy[:, 0]
            lat = xy[:, 1]
            out = [latlon_to_local(la, lo, lat_center, lon_center) for la, lo in zip(lat, lon)]
            return np.asarray(out, dtype=np.float32)
        if gdf.crs.is_geographic:
            lon = gdf.geometry.x.to_numpy(dtype=np.float64)
            lat = gdf.geometry.y.to_numpy(dtype=np.float64)
            out = [latlon_to_local(la, lo, lat_center, lon_center) for la, lo in zip(lat, lon)]
            return np.asarray(out, dtype=np.float32)
        # Projected CRS (meter): transform center lon/lat into this CRS and compute local meter offsets.
        to_proj = Transformer.from_crs("EPSG:4326", gdf.crs, always_xy=True)
        cx, cy = to_proj.transform(float(lon_center), float(lat_center))
        x = gdf.geometry.x.to_numpy(dtype=np.float64)
        y = gdf.geometry.y.to_numpy(dtype=np.float64)
        pts = np.column_stack([x - cx, y - cy]).astype(np.float32)
        return pts
    except Exception:
        pass

    try:
        import shapefile  # pyshp
    except ImportError:
        # Try companion CSV instead
        csv_path = os.path.splitext(filepath)[0] + '.csv'
        if os.path.exists(csv_path):
            print(f"[data_utils] pyshp not available; loading {csv_path} instead.")
            return load_fire_points_csv(csv_path, lat_center, lon_center)
        raise ImportError(
            "geopandas/pyproj or pyshp is required to read .shp files. "
            "Install one of:\n"
            "  pip install geopandas pyproj\n"
            "  pip install pyshp\n"
            f"Alternatively, place a CSV at {csv_path}."
        )

    xy = []
    with shapefile.Reader(filepath) as sf:
        for shape in sf.shapes():
            x, y = shape.points[0]
            xy.append([x, y])
    if not xy:
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float64)
    x_abs = np.nanmedian(np.abs(xy[:, 0]))
    y_abs = np.nanmedian(np.abs(xy[:, 1]))
    if x_abs > 1e4 or y_abs > 1e4:
        # Projected-meter legacy SHP without CRS metadata.
        return local_offsets_from_projected_xy(xy, lat_center, lon_center)
    # Assume lon/lat legacy input.
    out = [latlon_to_local(y, x, lat_center, lon_center) for x, y in xy]
    return np.asarray(out, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# GeoTIFF elevation loader (optional — requires rasterio)
# ──────────────────────────────────────────────────────────────────────────────

def find_default_elevation_source(prepare_dir):
    """Find default DEM file under prepare dir.

    Search order:
      1) prepare/elevation.zip, prepare/elevation.tif, prepare/elevation.tiff
      2) any tif/tiff/zip in prepare/
      3) any tif/tiff/zip in prepare/elevation/
    """
    if not prepare_dir:
        return ''
    root = os.path.abspath(prepare_dir)
    if not os.path.isdir(root):
        return ''

    preferred = [
        os.path.join(root, 'elevation.zip'),
        os.path.join(root, 'elevation.tif'),
        os.path.join(root, 'elevation.tiff'),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path

    def _pick_from_dir(path):
        if not os.path.isdir(path):
            return ''
        files = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith(('.tif', '.tiff', '.zip'))
        )
        if files:
            return os.path.join(path, files[0])
        return ''

    from_root = _pick_from_dir(root)
    if from_root:
        return from_root
    return _pick_from_dir(os.path.join(root, 'elevation'))


def resolve_elevation_raster_path(path):
    """Resolve DEM source path; supports .tif/.tiff and .zip containing tif."""
    if not path:
        raise ValueError("Empty elevation path")
    src = os.path.abspath(path)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Elevation source not found: {src}")
    lower = src.lower()
    if lower.endswith(('.tif', '.tiff')):
        return src
    if not lower.endswith('.zip'):
        raise ValueError(f"Unsupported elevation file type: {src}")
    cache_dir = os.path.join(tempfile.gettempdir(), 'dreamerv3_dem_cache')
    os.makedirs(cache_dir, exist_ok=True)
    with zipfile.ZipFile(src, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(('.tif', '.tiff'))]
        if not names:
            raise ValueError(f"No tif/tiff found in zip: {src}")
        pick = sorted(names)[0]
        out = os.path.join(cache_dir, os.path.basename(pick))
        if not os.path.exists(out):
            zf.extract(pick, cache_dir)
            extracted = os.path.join(cache_dir, pick)
            if extracted != out:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                os.replace(extracted, out)
    return out


def build_dem_query_metadata(tif_filepath):
    """Load full DEM and transforms for step-time elevation query."""
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.transform import rowcol
    except ImportError:
        return None
    raster_path = resolve_elevation_raster_path(tif_filepath)
    with rasterio.open(raster_path) as src:
        elevation = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        height, width = src.height, src.width
    to_raster = Transformer.from_crs("EPSG:4326", crs, always_xy=True) if crs else None
    return {
        'elevation': elevation,
        'transform': transform,
        'to_raster': to_raster,
        'width': int(width),
        'height': int(height),
        'nodata': nodata,
        'source_path': raster_path,
        'rowcol': rowcol,
    }


def load_elevation_obstacle_map(tif_filepath, lat_center, lon_center,
                                 region_radius_m, elevation_threshold=2000.0,
                                 target_resolution_m=50.0, return_metadata=False,
                                 mask_outside_circle=True):
    """Load a GeoTIFF elevation map and build a binary obstacle grid.

    Pixels with elevation ≥ *elevation_threshold* metres are marked as
    obstacles (``True``).  The returned grid covers a square bounding box of
    side 2 × *region_radius_m* centred on (*lat_center*, *lon_center*).

    Parameters
    ----------
    tif_filepath : str
    lat_center, lon_center : float
    region_radius_m : float
    elevation_threshold : float  (default 2000 m)
    target_resolution_m : float  (default 50 m/pixel)
    mask_outside_circle : bool (default True)
        If True, clear obstacle pixels whose cell centres fall outside the
        mission circle radius.

    Returns
    -------
    obstacle_map : np.ndarray, shape (H, W), dtype bool
        ``True`` where an obstacle is present.
    resolution_m : float
        Side length of each pixel in metres.

    Raises
    ------
    ImportError   if ``rasterio`` is not installed.
    FileNotFoundError   if the TIF file does not exist.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required to read GeoTIFF files. "
            "Install it with:  pip install rasterio"
        )

    raster_path = resolve_elevation_raster_path(tif_filepath)

    with rasterio.open(raster_path) as src:
        # Build a bounding box in geographic coordinates
        delta_lat = region_radius_m / 111_000.0
        delta_lon = region_radius_m / (111_000.0 * math.cos(math.radians(lat_center)))

        min_lon = lon_center - delta_lon
        max_lon = lon_center + delta_lon
        min_lat = lat_center - delta_lat
        max_lat = lat_center + delta_lat

        if src.crs and str(src.crs).upper() not in ('EPSG:4326', 'OGC:CRS84'):
            try:
                min_x, min_y, max_x, max_y = transform_bounds(
                    'EPSG:4326', src.crs,
                    min_lon, min_lat, max_lon, max_lat,
                    densify_pts=21,
                )
            except Exception:
                min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat
        else:
            min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat

        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)

        # Compute output size at target resolution
        size_pix = int(2 * region_radius_m / target_resolution_m)
        size_pix = max(size_pix, 2)

        elevation = src.read(
            1,
            window=window,
            out_shape=(size_pix, size_pix),
            resampling=Resampling.bilinear
        )

    obstacle_map = (elevation >= elevation_threshold)
    if mask_outside_circle:
        y = (np.arange(size_pix, dtype=np.float32) + 0.5 - size_pix / 2.0) * float(target_resolution_m)
        x = (np.arange(size_pix, dtype=np.float32) + 0.5 - size_pix / 2.0) * float(target_resolution_m)
        xx, yy = np.meshgrid(x, y)
        inside_circle = (xx * xx + yy * yy) <= (float(region_radius_m) ** 2)
        obstacle_map = np.logical_and(obstacle_map, inside_circle)
    if not return_metadata:
        return obstacle_map.astype(bool), float(target_resolution_m)
    return obstacle_map.astype(bool), float(target_resolution_m), build_dem_query_metadata(raster_path)


# ──────────────────────────────────────────────────────────────────────────────
# High-level loader — auto-detects file type
# ──────────────────────────────────────────────────────────────────────────────

def load_circle_data(center_csv, points_file, circle_id=None):
    """Convenience wrapper that loads centre + fire points for one circle.

    *center_csv* — path to the circle-centre CSV file.
    *points_file* — path to fire-points file (.shp or .csv).
    *circle_id*   — optional row selector (see :func:`load_circle_center_csv`).

    Returns
    -------
    lat_c, lon_c : float   — circle centre in WGS-84
    radius_m : float        — circle radius (metres)
    points : np.ndarray, shape (N, 2)  — fire points in local metric coords
    """
    lat_c, lon_c, radius_m = load_circle_center_csv(center_csv, circle_id=circle_id)

    ext = os.path.splitext(points_file)[1].lower()
    if ext == '.shp':
        points = load_fire_points_shp(points_file, lat_c, lon_c)
    elif ext == '.csv':
        points = load_fire_points_csv(points_file, lat_c, lon_c)
    else:
        raise ValueError(f"Unsupported fire-points file type: {ext}  (use .shp or .csv)")

    return lat_c, lon_c, radius_m, points


# ──────────────────────────────────────────────────────────────────────────────
# Sample data generators (used when real data is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def generate_sample_circle1_data(n_points=60, radius=5000.0, seed=42):
    """Synthesise sample data for Circle 1 (fire-point coverage, no obstacles).

    Returns
    -------
    center : (lat, lon, radius_m)   — representative centre in SW China
    points : np.ndarray (N, 2)      — fire points in local metric coordinates
    """
    rng = np.random.default_rng(seed)
    # Scatter fire points non-uniformly within the circle
    angles = rng.uniform(0, 2 * np.pi, n_points)
    # Use beta distribution to cluster points more toward the boundary
    radii  = rng.beta(1.5, 1.0, n_points) * radius * 0.95
    points = np.column_stack([radii * np.cos(angles),
                               radii * np.sin(angles)]).astype(np.float32)
    center = (25.5, 101.2, radius)   # somewhere in Yunnan, China
    return center, points


def generate_sample_circle8_data(n_points=35, radius=3000.0, seed=99):
    """Synthesise sample data for Circle 8 (fire-point coverage + obstacles).

    Returns
    -------
    center : (lat, lon, radius_m)
    points : np.ndarray (N, 2)  — fire points
    obstacle_map : np.ndarray (H, W) bool  — synthetic elevation obstacle grid
    resolution_m : float
    """
    rng = np.random.default_rng(seed)

    # Fire points — biased toward the lower half of the circle (avoiding mountains)
    angles = rng.uniform(-np.pi, 0, n_points)
    radii  = rng.beta(1.2, 1.0, n_points) * radius * 0.9
    points = np.column_stack([radii * np.cos(angles),
                               radii * np.sin(angles)]).astype(np.float32)

    # Obstacle map: upper half of the circle has obstacles (simulates high terrain)
    resolution_m = 50.0
    grid_size = int(2 * radius / resolution_m)
    obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
    cx = cy = grid_size // 2
    for i in range(grid_size):
        for j in range(grid_size):
            # Local coords of this pixel
            x = (j - cx) * resolution_m
            y = (cy - i) * resolution_m   # row 0 = top = north
            # Mountains in the upper half (y > 0)
            if y > radius * 0.2 and rng.random() < 0.65:
                obstacle_map[i, j] = True
            elif rng.random() < 0.05:
                obstacle_map[i, j] = True   # scattered obstacles elsewhere

    center = (26.1, 101.8, radius)
    return center, points, obstacle_map, resolution_m


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer helpers (used to save sample data for inspection)
# ──────────────────────────────────────────────────────────────────────────────

def save_sample_center_csv(filepath, lat, lon, radius_m, circle_id=1):
    """Write a sample circle-centre CSV file in the new format.

    Columns: circle_id, center_x (longitude), center_y (latitude),
             radius_m, diameter_m.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['circle_id', 'center_x', 'center_y', 'radius_m', 'diameter_m'])
        writer.writerow([circle_id, lon, lat, radius_m, radius_m * 2])


def save_sample_points_csv(filepath, points_local, lat_center, lon_center):
    """Write fire points as a lat/lon CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['latitude', 'longitude'])
        for e, n in points_local:
            lat, lon = local_to_latlon(e, n, lat_center, lon_center)
            writer.writerow([lat, lon])
