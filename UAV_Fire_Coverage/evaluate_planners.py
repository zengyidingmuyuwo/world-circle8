import argparse
import math
import os
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_utils import (
    load_circle_data,
    load_elevation_obstacle_map,
    generate_sample_circle1_data,
    generate_sample_circle8_data,
)
from planners.dubins_utils import dubins_like_connect, path_length, simple_turning_connect, stitch_paths
from planners.rrt_star_pdubins import pdubins_rrt_star_connect
from planners.tsp_heuristics import (
    nearest_neighbor_order,
    route_cost_dubins,
    turning_aware_order,
    two_opt_improve,
)


UAV_SPEED = 20.0
MAX_TURN_RATE = 0.25
DT = 1.0
TURN_RADIUS = UAV_SPEED / MAX_TURN_RATE
BIRD_RADIUS = 85.0
N_CIRCLE1_CLUSTERS = 3
BASELINE3_SEED_OFFSET = 123
BIRD_MIN_RADIUS_FACTOR = 0.15
BIRD_MAX_RADIUS_FACTOR = 0.85
BIRD_MIN_SPEED = 8.0
BIRD_MAX_SPEED = 16.0
# Keep reflected birds slightly inside boundary to avoid repeated boundary hits.
BOUNDARY_REFLECTION_OFFSET = 1.0
EPS = 1e-9


@dataclass
class EvalResult:
    method: str
    path: np.ndarray
    planning_time: float
    exec_length: float
    plan_length: float
    total_time: float
    collisions: int
    min_distance: float
    obstacle_hits: int
    infeasible_segments: int
    visited_count: int
    total_points: int


@dataclass
class PlannerOutput:
    path: np.ndarray
    order: List[int]
    plan_length: float
    infeasible_segments: int
    visited_count: int


def _resolve_prepare_paths(task: str, center_csv: str = '', points_file: str = '') -> Tuple[str, str]:
    if center_csv and points_file:
        return center_csv, points_file
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prepare = os.path.join(root, 'prepare')
    center = os.path.join(prepare, f'circle_{1 if task == "circle1" else 8}_center.csv')
    shp = os.path.join(prepare, f'circle_{1 if task == "circle1" else 8}_points.shp')
    csv = os.path.join(prepare, f'circle_{1 if task == "circle1" else 8}_points.csv')
    points = csv if os.path.exists(csv) else shp
    return center, points


def _resolve_elevation_path(task: str, elevation_tif: str = '') -> str:
    if task != 'circle8':
        return ''
    if elevation_tif:
        return elevation_tif
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prepare = os.path.join(root, 'prepare')
    direct_zip = os.path.join(prepare, 'elevation.zip')
    if os.path.exists(direct_zip):
        return direct_zip
    elev_dir = os.path.join(prepare, 'elevation')
    if os.path.isdir(elev_dir):
        tif_candidates = sorted(
            f for f in os.listdir(elev_dir) if f.lower().endswith(('.tif', '.tiff', '.zip'))
        )
        if tif_candidates:
            return os.path.join(elev_dir, tif_candidates[0])
    return ''


def _log_elevation_source(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    lower = path.lower()
    if lower.endswith('.zip'):
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
                tif_candidates = [n for n in names if n.lower().endswith(('.tif', '.tiff'))]
                if tif_candidates:
                    pick = sorted(tif_candidates)[0]
                    ext = os.path.splitext(pick)[1].lower()
                    print(f'[Data] elevation_zip_member={pick} (format={ext})')
                else:
                    print('[WARNING] elevation.zip contains no tif/tiff files.')
        except Exception as e:
            print(f'[WARNING] Could not inspect elevation.zip: {e}')
    else:
        ext = os.path.splitext(path)[1].lower()
        print(f'[Data] elevation_raster={os.path.basename(path)} (format={ext})')


def _load_circle_dataset(
    task: str,
    seed: int,
    center_csv: str,
    points_file: str,
    circle_id: Optional[int],
    allow_sample: bool,
) -> Tuple[float, float, float, np.ndarray, dict, bool]:
    center_csv, points_file = _resolve_prepare_paths(task, center_csv, points_file)
    if os.path.exists(center_csv) and os.path.exists(points_file):
        lat_c, lon_c, radius, fire_points, meta = load_circle_data(
            center_csv,
            points_file,
            circle_id=circle_id,
            return_meta=True,
        )
        return float(lat_c), float(lon_c), float(radius), np.asarray(fire_points, dtype=np.float32), meta, False
    if not allow_sample:
        raise FileNotFoundError(
            f'Missing data files for {task}: center_csv={center_csv}, points_file={points_file}'
        )
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    sample_center = os.path.join(sample_dir, f'circle_{1 if task == "circle1" else 8}_center.csv')
    sample_points = os.path.join(sample_dir, f'circle_{1 if task == "circle1" else 8}_points.csv')
    if os.path.exists(sample_center) and os.path.exists(sample_points):
        lat_c, lon_c, radius, fire_points, meta = load_circle_data(
            sample_center,
            sample_points,
            return_meta=True,
        )
        meta['sample_data'] = True
        return float(lat_c), float(lon_c), float(radius), np.asarray(fire_points, dtype=np.float32), meta, True
    if task == 'circle1':
        (lat_c, lon_c, radius), fire_points = generate_sample_circle1_data(seed=seed)
        return float(lat_c), float(lon_c), float(radius), np.asarray(fire_points, dtype=np.float32), {'sample_data': True}, True
    (lat_c, lon_c, radius), fire_points, _, _ = generate_sample_circle8_data(seed=seed)
    return float(lat_c), float(lon_c), float(radius), np.asarray(fire_points, dtype=np.float32), {'sample_data': True}, True


def _split_circle1_clusters(points: np.ndarray, seed: int) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=N_CIRCLE1_CLUSTERS, random_state=seed, n_init='auto')
        labels = km.fit_predict(pts)
    except Exception:
        rng = np.random.default_rng(seed)
        labels = np.arange(len(pts), dtype=np.int32)
        rng.shuffle(labels)
        labels = labels % N_CIRCLE1_CLUSTERS
    clusters = [pts[labels == k] for k in range(N_CIRCLE1_CLUSTERS)]
    return [c for c in clusters if len(c) > 0]


def _heading_to(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return float(math.atan2(float(d[1]), float(d[0])))


def _plan_length_euclidean(points: np.ndarray, order: List[int]) -> float:
    if not order:
        return 0.0
    cur = np.asarray([0.0, 0.0], dtype=np.float32)
    total = 0.0
    for idx in order:
        total += float(np.linalg.norm(points[idx] - cur))
        cur = points[idx]
    return float(total)


def _segment_has_obstacle(path: np.ndarray, obstacle_map: Optional[np.ndarray], resolution_m: float) -> bool:
    if obstacle_map is None:
        return False
    return _count_obstacle_hits(path, obstacle_map, resolution_m) > 0


def _count_infeasible_segments(
    segments: List[np.ndarray],
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
) -> int:
    if obstacle_map is None:
        return 0
    count = 0
    for seg in segments:
        if len(seg) < 2:
            continue
        if _segment_has_obstacle(seg, obstacle_map, resolution_m):
            count += 1
    return count


def _build_path_simple(points: np.ndarray, order: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
    pos = np.asarray([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    for idx in order:
        seg, heading = simple_turning_connect(pos, heading, points[idx], UAV_SPEED, MAX_TURN_RATE, DT)
        segments.append(seg)
        pos = points[idx]
    return stitch_paths(segments), segments


def _build_path_dubins(points: np.ndarray, order: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
    pos = np.asarray([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    for k, idx in enumerate(order):
        goal = points[idx]
        goal_h = heading
        if k + 1 < len(order):
            goal_h = _heading_to(goal, points[order[k + 1]])
        seg, heading = dubins_like_connect(
            pos,
            heading,
            goal,
            goal_h,
            speed=UAV_SPEED,
            max_turn_rate=MAX_TURN_RATE,
            dt=DT,
            turn_radius=TURN_RADIUS,
        )
        segments.append(seg)
        pos = goal
    return stitch_paths(segments), segments


def _make_birds(rng: np.random.Generator, radius: float, n_birds: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    ang = rng.uniform(-math.pi, math.pi, size=n_birds)
    rr = radius * rng.uniform(BIRD_MIN_RADIUS_FACTOR, BIRD_MAX_RADIUS_FACTOR, size=n_birds)
    birds = np.column_stack([rr * np.cos(ang), rr * np.sin(ang)]).astype(np.float32)
    vel_ang = rng.uniform(-math.pi, math.pi, size=n_birds)
    speed = rng.uniform(BIRD_MIN_SPEED, BIRD_MAX_SPEED, size=n_birds)
    vels = np.column_stack([speed * np.cos(vel_ang), speed * np.sin(vel_ang)]).astype(np.float32)
    return birds, vels


def _force_bird_on_path(birds: np.ndarray, points: np.ndarray, order: List[int], radius: float) -> None:
    if len(birds) == 0 or not order:
        return
    start = np.asarray([0.0, 0.0], dtype=np.float32)
    first = points[order[0]]
    mid = (start + first) * 0.5
    r = float(np.linalg.norm(mid))
    max_r = max(radius - BIRD_RADIUS, 1.0)
    if r > max_r:
        mid = mid / (r + EPS) * max_r
    birds[0] = mid


def _update_birds(birds: np.ndarray, vels: np.ndarray, radius: float, dt: float) -> None:
    birds += vels * float(dt)
    r = np.linalg.norm(birds, axis=1)
    hit = r > radius
    if np.any(hit):
        n = birds[hit] / (r[hit][:, None] + EPS)
        vels[hit] -= 2.0 * np.sum(vels[hit] * n, axis=1, keepdims=True) * n
        birds[hit] = n * (radius - BOUNDARY_REFLECTION_OFFSET)


def _world_to_grid(p: np.ndarray, obstacle_map: np.ndarray, resolution_m: float) -> Optional[Tuple[int, int]]:
    if obstacle_map is None:
        return None
    h, w = obstacle_map.shape
    cx, cy = w // 2, h // 2
    j = int(cx + float(p[0]) / float(resolution_m))
    i = int(cy - float(p[1]) / float(resolution_m))
    if 0 <= i < h and 0 <= j < w:
        return i, j
    return None


def _sample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    if len(path) < 2:
        return path
    out = [path[0].copy()]
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        seg = b - a
        d = float(np.linalg.norm(seg))
        if d < 1e-6:
            continue
        n = max(1, int(math.ceil(d / spacing)))
        for k in range(1, n + 1):
            out.append((a + seg * (k / n)).astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def _count_obstacle_hits(path: np.ndarray, obstacle_map: Optional[np.ndarray], resolution_m: float) -> int:
    if obstacle_map is None or len(path) == 0:
        return 0
    spacing = max(1.0, min(UAV_SPEED * DT, resolution_m * 0.5))
    pts = _sample_path(path, spacing=spacing)
    hits = 0
    for p in pts:
        cell = _world_to_grid(p, obstacle_map, resolution_m)
        if cell is not None and obstacle_map[cell]:
            hits += 1
    return hits


def _evaluate_bird_metrics(path: np.ndarray, birds_init: np.ndarray, birds_vel_init: np.ndarray, birds_mode: str, radius: float) -> Tuple[int, float]:
    pts = _sample_path(path, spacing=UAV_SPEED * DT)
    birds = birds_init.copy()
    vels = birds_vel_init.copy()
    collisions = 0
    min_d = float('inf')
    for p in pts:
        d = np.linalg.norm(birds - p[None, :], axis=1)
        min_d = min(min_d, float(np.min(d)))
        collisions += int(np.sum(d <= BIRD_RADIUS))
        if birds_mode == 'moving':
            _update_birds(birds, vels, radius, DT)
    if not np.isfinite(min_d):
        min_d = float('nan')
    return collisions, min_d


def _baseline1(points: np.ndarray, obstacle_map: Optional[np.ndarray], resolution_m: float) -> PlannerOutput:
    order = nearest_neighbor_order(points)
    path, segments = _build_path_simple(points, order)
    infeasible = _count_infeasible_segments(segments, obstacle_map, resolution_m)
    plan_len = _plan_length_euclidean(points, order)
    return PlannerOutput(path=path, order=order, plan_length=plan_len, infeasible_segments=infeasible, visited_count=len(order))


def _baseline2(points: np.ndarray, obstacle_map: Optional[np.ndarray], resolution_m: float) -> PlannerOutput:
    order = nearest_neighbor_order(points)
    path, segments = _build_path_dubins(points, order)
    infeasible = _count_infeasible_segments(segments, obstacle_map, resolution_m)
    plan_len = route_cost_dubins(points, order, np.asarray([0.0, 0.0], dtype=np.float32), 0.0, TURN_RADIUS)
    return PlannerOutput(path=path, order=order, plan_length=plan_len, infeasible_segments=infeasible, visited_count=len(order))


def _baseline3(
    points: np.ndarray,
    rng: np.random.Generator,
    birds: np.ndarray,
    radius: float,
    obstacle_map: Optional[np.ndarray] = None,
    resolution_m: float = 50.0,
) -> PlannerOutput:
    order = turning_aware_order(points, np.asarray([0.0, 0.0], dtype=np.float32), 0.0, TURN_RADIUS)
    order = two_opt_improve(order, points, np.asarray([0.0, 0.0], dtype=np.float32), 0.0, TURN_RADIUS)
    plan_len = route_cost_dubins(points, order, np.asarray([0.0, 0.0], dtype=np.float32), 0.0, TURN_RADIUS)

    pos = np.asarray([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    infeasible = 0
    visited = 0
    for k, idx in enumerate(order):
        goal = points[idx]
        goal_h = heading
        if k + 1 < len(order):
            goal_h = _heading_to(goal, points[order[k + 1]])
        seg, _ = pdubins_rrt_star_connect(
            pos,
            heading,
            goal,
            goal_h,
            radius=radius,
            rng=rng,
            birds_xy=birds,
            bird_radius=BIRD_RADIUS,
            speed=UAV_SPEED,
            max_turn_rate=MAX_TURN_RATE,
            dt=DT,
            obstacle_map=obstacle_map,
            resolution_m=resolution_m,
        )
        if len(seg) == 0:
            seg, heading = dubins_like_connect(pos, heading, goal, goal_h, UAV_SPEED, MAX_TURN_RATE, DT, TURN_RADIUS)
            if _segment_has_obstacle(seg, obstacle_map, resolution_m):
                seg = np.zeros((0, 2), dtype=np.float32)
        if len(seg) == 0:
            infeasible += 1
            continue
        heading = _heading_to(seg[-2], seg[-1]) if len(seg) >= 2 else heading
        segments.append(seg)
        pos = goal
        visited += 1
    path = stitch_paths(segments)
    infeasible += _count_infeasible_segments(segments, obstacle_map, resolution_m)
    return PlannerOutput(path=path, order=order, plan_length=plan_len, infeasible_segments=infeasible, visited_count=visited)


def _run_method(
    name: str,
    planner_fn,
    points: np.ndarray,
    birds: np.ndarray,
    birds_vel: np.ndarray,
    birds_mode: str,
    radius: float,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
) -> EvalResult:
    t0 = time.perf_counter()
    plan = planner_fn(points)
    planning_time = time.perf_counter() - t0
    path = plan.path
    exec_len = path_length(path)
    total_t = exec_len / UAV_SPEED if UAV_SPEED > 1e-6 else float('nan')
    collisions, min_d = _evaluate_bird_metrics(path, birds, birds_vel, birds_mode, radius)
    obstacle_hits = _count_obstacle_hits(path, obstacle_map, resolution_m)
    return EvalResult(
        name,
        path,
        planning_time,
        exec_len,
        plan.plan_length,
        total_t,
        collisions,
        min_d,
        obstacle_hits,
        plan.infeasible_segments,
        plan.visited_count,
        len(points),
    )


def _plot_results(
    points: np.ndarray,
    birds: np.ndarray,
    results: List[EvalResult],
    radius: float,
    save_path: str,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, res in zip(axes, results):
        if obstacle_map is not None:
            h, w = obstacle_map.shape
            ext = [-w // 2 * resolution_m, w // 2 * resolution_m, -h // 2 * resolution_m, h // 2 * resolution_m]
            ax.imshow(obstacle_map.astype(np.float32), cmap='gray', alpha=0.35, extent=ext, origin='upper')
        if len(res.path) > 1:
            ax.plot(res.path[:, 0], res.path[:, 1], '-', lw=2.0, color='tab:blue', label='Trajectory')
        ax.scatter(points[:, 0], points[:, 1], s=55, c='orange', marker='*', label='Fire points')
        ax.scatter(birds[:, 0], birds[:, 1], s=80, c='red', marker='^', label='Birds')
        ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, linestyle='--', color='gray', linewidth=1.0))
        ax.set_aspect('equal')
        ax.set_title(
            f"{res.method}\n"
            f"L_exec={res.exec_length:.1f}m  T={res.total_time:.1f}s  Plan={res.planning_time:.3f}s\n"
            f"Coll={res.collisions}  Infeas={res.infeasible_segments}  MinD={res.min_distance:.1f}m\n"
            f"Visited={res.visited_count}/{res.total_points}  ObsHits={res.obstacle_hits}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Offline Path Planning Evaluation (Circle Fire Coverage)')
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)


def _plot_context(
    points: np.ndarray,
    birds: np.ndarray,
    radius: float,
    save_path: str,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    if obstacle_map is not None:
        h, w = obstacle_map.shape
        ext = [-w // 2 * resolution_m, w // 2 * resolution_m, -h // 2 * resolution_m, h // 2 * resolution_m]
        ax.imshow(obstacle_map.astype(np.float32), cmap='gray', alpha=0.35, extent=ext, origin='upper')
    ax.scatter(points[:, 0], points[:, 1], s=55, c='orange', marker='*', label='Fire points')
    ax.scatter(birds[:, 0], birds[:, 1], s=80, c='red', marker='^', label='Birds')
    ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, linestyle='--', color='gray', linewidth=1.0))
    ax.set_aspect('equal')
    ax.set_title('Circle context: fire points + obstacles + birds')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline path-planning evaluation for Circle1/Circle8 (no RL training).')
    parser.add_argument('--task', type=str, default='circle8', choices=['circle1', 'circle8'])
    parser.add_argument('--cluster_id', type=int, default=0, help='Circle1 cluster id (0-based). Ignored for circle8.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--birds_mode', type=str, default='frozen', choices=['frozen', 'moving', 'force_on_path'])
    parser.add_argument('--save_path', type=str, default='offline_planner_eval.png')
    parser.add_argument('--context_path', type=str, default='')
    parser.add_argument('--center_csv', type=str, default='')
    parser.add_argument('--points_file', type=str, default='')
    parser.add_argument('--circle_id', type=int, default=None)
    parser.add_argument('--elevation_tif', type=str, default='')
    parser.add_argument('--elev_threshold', type=float, default=2000.0)
    parser.add_argument('--resolution_m', type=float, default=50.0)
    parser.add_argument('--allow_sample', action='store_true', help='Allow fallback to synthetic/sample data.')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lat_c, lon_c, radius, fire_points, meta, used_sample = _load_circle_dataset(
        args.task,
        args.seed,
        args.center_csv,
        args.points_file,
        args.circle_id,
        args.allow_sample,
    )
    center_csv, points_file = _resolve_prepare_paths(args.task, args.center_csv, args.points_file)
    elevation_tif = _resolve_elevation_path(args.task, args.elevation_tif)

    print(f'[Data] task={args.task}  seed={args.seed}  sample={used_sample}')
    print(f'[Data] center_csv={os.path.abspath(center_csv)}')
    print(f'[Data] points_file={os.path.abspath(points_file)}')
    print(f'[Data] center=({lat_c:.6f}°N, {lon_c:.6f}°E)  radius={radius:.1f} m')
    if meta:
        coord_mode = meta.get('coord_mode', 'unknown')
        print(f'[Data] coord_mode={coord_mode}')
        if meta.get('scale_m_per_deg_lat') is not None:
            print(
                f"[Data] scale_m_per_deg: lat={meta['scale_m_per_deg_lat']:.1f}, "
                f"lon={meta['scale_m_per_deg_lon']:.1f}"
            )
        if meta.get('utm_epsg') is not None:
            print(f"[Data] utm_epsg={meta['utm_epsg']}")
        if meta.get('crs'):
            print(f"[Data] source_crs={meta['crs']}")

    obstacle_map = None
    resolution_m = float(args.resolution_m)
    if args.task == 'circle8':
        if elevation_tif and os.path.exists(elevation_tif):
            print(f'[Data] elevation_tif={os.path.abspath(elevation_tif)}')
            _log_elevation_source(elevation_tif)
            try:
                obstacle_map, resolution_m = load_elevation_obstacle_map(
                    elevation_tif,
                    lat_c,
                    lon_c,
                    region_radius_m=radius,
                    elevation_threshold=args.elev_threshold,
                    target_resolution_m=resolution_m,
                )
                n_obs = int(np.sum(obstacle_map))
                print(
                    f'[Data] elev_threshold={args.elev_threshold:.1f}m  '
                    f'obstacle_map={obstacle_map.shape}  resolution={resolution_m:.1f}m  '
                    f'obstacle_pixels={n_obs} ({n_obs/obstacle_map.size*100:.2f}%)'
                )
            except Exception as e:
                print(f'[WARNING] Could not load elevation map: {e}')
        else:
            print('[WARNING] No elevation_tif found; obstacle map disabled.')

    if args.task == 'circle1':
        clusters = _split_circle1_clusters(fire_points, args.seed)
        if not (0 <= args.cluster_id < len(clusters)):
            raise ValueError(
                f'cluster_id out of range: {args.cluster_id}, must be between 0 and {len(clusters)-1} (inclusive)'
            )
        fire_points = clusters[args.cluster_id]

    birds, bird_vels = _make_birds(rng, radius)
    if args.birds_mode == 'force_on_path':
        force_order = nearest_neighbor_order(fire_points)
        _force_bird_on_path(birds, fire_points, force_order, radius)
        print('[Birds] force_on_path enabled: placing one bird on baseline2 first leg.')

    methods = [
        (
            'Baseline1: Euclidean-order + simple turning',
            lambda pts: _baseline1(pts, obstacle_map=obstacle_map, resolution_m=resolution_m),
        ),
        (
            'Baseline2: Euclidean-order + Dubins',
            lambda pts: _baseline2(pts, obstacle_map=obstacle_map, resolution_m=resolution_m),
        ),
        (
            'Baseline3: 2-opt turn-cost + P-Dubins-RRT*',
            lambda pts: _baseline3(
                pts,
                rng=np.random.default_rng(args.seed + BASELINE3_SEED_OFFSET),
                birds=birds,
                radius=radius,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
            ),
        ),
    ]

    results = []
    for name, fn in methods:
        res = _run_method(
            name,
            fn,
            fire_points,
            birds,
            bird_vels,
            args.birds_mode,
            radius,
            obstacle_map,
            resolution_m,
        )
        results.append(res)
        print(
            f'[{name}] L_exec={res.exec_length:.2f}m L_plan={res.plan_length:.2f}m '
            f'time={res.total_time:.2f}s planning={res.planning_time:.3f}s '
            f'collisions={res.collisions} infeasible={res.infeasible_segments} '
            f'min_distance={res.min_distance:.2f}m obstacle_hits={res.obstacle_hits} '
            f'visited={res.visited_count}/{res.total_points}'
        )

    context_path = args.context_path
    if not context_path:
        base, ext = os.path.splitext(args.save_path)
        context_path = f'{base}_context{ext or ".png"}'

    _plot_results(fire_points, birds, results, radius, args.save_path, obstacle_map, resolution_m)
    _plot_context(fire_points, birds, radius, context_path, obstacle_map, resolution_m)
    print(f'Saved: {args.save_path}')
    print(f'Saved context: {context_path}')


if __name__ == '__main__':
    main()
