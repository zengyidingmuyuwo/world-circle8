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
from planners.goal_region_utils import optimize_entry_points
from planners.rrt_star_pdubins import pdubins_rrt_star_connect
from planners.tsp_heuristics import (
    exec_rollout_length,
    nearest_neighbor_order,
    route_cost_dubins,
    turning_aware_order,
    two_opt_improve,
    two_opt_improve_exec,
)


UAV_SPEED = 20.0
MAX_TURN_RATE = 0.25
DT = 1.0
TURN_RADIUS = UAV_SPEED / MAX_TURN_RATE
BIRD_RADIUS = 85.0
VISIT_RADIUS = 120.0
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
    smoothness: float
    total_time: float
    collisions: int
    min_distance: float
    obstacle_hits: int
    infeasible_segments: int
    visited_count: int
    total_points: int
    coverage_rate: float
    success: bool


@dataclass
class PlannerOutput:
    path: np.ndarray
    order: List[int]
    plan_length: float
    infeasible_segments: int
    visited_count: int


@dataclass
class LocalPairResult:
    start_idx: int
    goal_idx: int
    path: np.ndarray
    planning_time: float
    exec_length: float
    smoothness: float
    obstacle_hits: int
    success: bool


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


def _plan_length_euclidean(points: np.ndarray, order: List[int], visit_radius: float = 0.0) -> float:
    if not order:
        return 0.0
    cur = np.asarray([0.0, 0.0], dtype=np.float32)
    cur_radius = 0.0
    total = 0.0
    for idx in order:
        d = float(np.linalg.norm(points[idx] - cur))
        if visit_radius > 0.0:
            d = max(0.0, d - (cur_radius + visit_radius))
        total += d
        cur = points[idx]
        cur_radius = visit_radius
    return float(total)


def _trim_path_on_visit(path: np.ndarray, goal: np.ndarray, visit_radius: float) -> Tuple[np.ndarray, bool]:
    if visit_radius <= 0.0 or len(path) == 0:
        return path, False
    d = np.linalg.norm(path - goal[None, :], axis=1)
    hit = np.where(d <= visit_radius)[0]
    if len(hit) == 0:
        return path, False
    cut = int(hit[0])
    return np.asarray(path[:cut + 1], dtype=np.float32), True


def _segment_reaches(path: np.ndarray, goal: np.ndarray, visit_radius: float, goal_tolerance: float = 20.0) -> bool:
    if len(path) == 0:
        return False
    if visit_radius > 0.0:
        return bool(np.any(np.linalg.norm(path - goal[None, :], axis=1) <= visit_radius))
    return bool(np.linalg.norm(path[-1] - goal) <= goal_tolerance)


def _line_connect(start_xy: np.ndarray, goal_xy: np.ndarray, step: float) -> np.ndarray:
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    seg = goal - start
    dist = float(np.linalg.norm(seg))
    if dist < 1e-6:
        return np.asarray([start.copy()], dtype=np.float32)
    n = max(1, int(math.ceil(dist / max(step, 1e-6))))
    pts = [start + seg * (k / n) for k in range(n + 1)]
    return np.asarray(pts, dtype=np.float32)


def _resolve_entry_points(
    points: np.ndarray,
    order: List[int],
    visit_radius: float,
    entry_point_opt: str,
    entry_point_k: int,
    turn_then_straight: bool,
) -> Optional[np.ndarray]:
    if entry_point_opt != 'sample_circle' or visit_radius <= 0.0 or entry_point_k <= 0:
        return None
    return optimize_entry_points(
        points,
        order,
        visit_radius=visit_radius,
        start_xy=np.asarray([0.0, 0.0], dtype=np.float32),
        start_heading=0.0,
        speed=UAV_SPEED,
        max_turn_rate=MAX_TURN_RATE,
        dt=DT,
        turn_radius=TURN_RADIUS,
        entry_point_k=entry_point_k,
        turn_then_straight=turn_then_straight,
    )


def _connector_label(connector: str) -> str:
    mapping = {
        'straight': 'Straight',
        'dubins_like': 'Dubins-like',
        'rrtstar': 'RRT*',
        'dubins_rrtstar': 'Dubins-RRT*',
        'pdubins_rrtstar': 'P-Dubins-RRT*',
        'goal_region_pdubins_rrtstar': 'Goal-region P-Dubins-RRT*',
    }
    return mapping.get(connector, connector)


def _connect_with_connector(
    connector: str,
    pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    goal_heading: float,
    radius: float,
    rng: np.random.Generator,
    birds: np.ndarray,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
    visit_center: np.ndarray,
    visit_radius: float,
    early_terminate_on_visit: bool,
    time_budget: Optional[float],
    turn_then_straight: bool,
) -> np.ndarray:
    if connector == 'straight':
        seg = _line_connect(pos, goal, step=UAV_SPEED * DT)
    elif connector == 'dubins_like':
        seg, _ = dubins_like_connect(
            pos,
            heading,
            goal,
            goal_heading,
            UAV_SPEED,
            MAX_TURN_RATE,
            DT,
            TURN_RADIUS,
            visit_radius=visit_radius,
            terminate_on_visit=early_terminate_on_visit,
            turn_then_straight=turn_then_straight,
        )
    else:
        use_turn_then = turn_then_straight
        max_turn_rate = MAX_TURN_RATE
        goal_region = False
        if connector == 'rrtstar':
            max_turn_rate = max(MAX_TURN_RATE * 1000.0, 1.0)
            use_turn_then = True
        elif connector == 'dubins_rrtstar':
            use_turn_then = True
        elif connector == 'goal_region_pdubins_rrtstar':
            goal_region = True
        seg, _ = pdubins_rrt_star_connect(
            pos,
            heading,
            goal,
            goal_heading,
            radius=radius,
            rng=rng,
            birds_xy=birds,
            bird_radius=BIRD_RADIUS,
            speed=UAV_SPEED,
            max_turn_rate=max_turn_rate,
            dt=DT,
            obstacle_map=obstacle_map,
            resolution_m=resolution_m,
            visit_radius=visit_radius,
            terminate_on_visit=early_terminate_on_visit,
            turn_then_straight=use_turn_then,
            time_budget=time_budget,
            goal_region=goal_region,
        )
    if visit_radius > 0.0:
        seg, _ = _trim_path_on_visit(seg, visit_center, visit_radius)
    return seg


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


def _build_path_simple(
    points: np.ndarray,
    order: List[int],
    visit_radius: float = 0.0,
    early_terminate_on_visit: bool = False,
    entry_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    pos = np.asarray([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    visited = 0
    targets = points if entry_points is None else np.asarray(entry_points, dtype=np.float32)
    for idx in order:
        seg, heading = simple_turning_connect(
            pos,
            heading,
            targets[idx],
            UAV_SPEED,
            MAX_TURN_RATE,
            DT,
            visit_radius=visit_radius,
            terminate_on_visit=early_terminate_on_visit,
        )
        if visit_radius > 0.0:
            seg, _ = _trim_path_on_visit(seg, points[idx], visit_radius)
        if _segment_reaches(seg, points[idx], visit_radius):
            visited += 1
        if len(seg) >= 2:
            heading = _heading_to(seg[-2], seg[-1])
        if len(seg) > 0:
            pos = seg[-1]
        segments.append(seg)
    return stitch_paths(segments), segments, visited


def _build_path_dubins(
    points: np.ndarray,
    order: List[int],
    visit_radius: float = 0.0,
    early_terminate_on_visit: bool = False,
    turn_then_straight: bool = False,
    entry_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    pos = np.asarray([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    visited = 0
    targets = points if entry_points is None else np.asarray(entry_points, dtype=np.float32)
    for k, idx in enumerate(order):
        goal = targets[idx]
        goal_h = heading
        if k + 1 < len(order):
            goal_h = _heading_to(goal, targets[order[k + 1]])
        seg, heading = dubins_like_connect(
            pos,
            heading,
            goal,
            goal_h,
            speed=UAV_SPEED,
            max_turn_rate=MAX_TURN_RATE,
            dt=DT,
            turn_radius=TURN_RADIUS,
            visit_radius=visit_radius,
            terminate_on_visit=early_terminate_on_visit,
            turn_then_straight=turn_then_straight,
        )
        if visit_radius > 0.0:
            seg, _ = _trim_path_on_visit(seg, goal, visit_radius)
        if _segment_reaches(seg, goal, visit_radius):
            visited += 1
        if len(seg) >= 2:
            heading = _heading_to(seg[-2], seg[-1])
        if len(seg) > 0:
            pos = seg[-1]
        segments.append(seg)
    return stitch_paths(segments), segments, visited


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


def _path_smoothness(path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    pts = _sample_path(path, spacing=UAV_SPEED * DT)
    if len(pts) < 3:
        return 0.0
    headings = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    dtheta = np.diff(headings)
    dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.sum(np.abs(dtheta)))


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


def _baseline1(
    points: np.ndarray,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
    visit_radius: float = 0.0,
    early_terminate_on_visit: bool = False,
    entry_point_opt: str = 'none',
    entry_point_k: int = 16,
    turn_then_straight: bool = False,
) -> PlannerOutput:
    order = nearest_neighbor_order(points)
    entry_points = _resolve_entry_points(
        points,
        order,
        visit_radius,
        entry_point_opt,
        entry_point_k,
        turn_then_straight,
    )
    path, segments, visited = _build_path_simple(
        points,
        order,
        visit_radius=visit_radius,
        early_terminate_on_visit=early_terminate_on_visit,
        entry_points=entry_points,
    )
    infeasible = _count_infeasible_segments(segments, obstacle_map, resolution_m)
    plan_len = _plan_length_euclidean(points, order, visit_radius=visit_radius)
    return PlannerOutput(path=path, order=order, plan_length=plan_len, infeasible_segments=infeasible, visited_count=visited)


def _baseline2(
    points: np.ndarray,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
    visit_radius: float = 0.0,
    early_terminate_on_visit: bool = False,
    turn_then_straight: bool = False,
    entry_point_opt: str = 'none',
    entry_point_k: int = 16,
) -> PlannerOutput:
    order = nearest_neighbor_order(points)
    entry_points = _resolve_entry_points(
        points,
        order,
        visit_radius,
        entry_point_opt,
        entry_point_k,
        turn_then_straight,
    )
    path, segments, visited = _build_path_dubins(
        points,
        order,
        visit_radius=visit_radius,
        early_terminate_on_visit=early_terminate_on_visit,
        turn_then_straight=turn_then_straight,
        entry_points=entry_points,
    )
    infeasible = _count_infeasible_segments(segments, obstacle_map, resolution_m)
    plan_len = route_cost_dubins(
        points,
        order,
        np.asarray([0.0, 0.0], dtype=np.float32),
        0.0,
        TURN_RADIUS,
        visit_radius=visit_radius,
        entry_points=entry_points,
    )
    return PlannerOutput(path=path, order=order, plan_length=plan_len, infeasible_segments=infeasible, visited_count=visited)


def _baseline3(
    points: np.ndarray,
    rng: np.random.Generator,
    birds: np.ndarray,
    radius: float,
    obstacle_map: Optional[np.ndarray] = None,
    resolution_m: float = 50.0,
    visit_radius: float = 0.0,
    use_visit_radius_planning: bool = False,
    early_terminate_on_visit: bool = False,
    two_opt_use_exec_rollout: bool = False,
    turn_then_straight: bool = False,
    entry_point_opt: str = 'none',
    entry_point_k: int = 16,
    connector: str = 'pdubins_rrtstar',
    time_budget: Optional[float] = None,
) -> PlannerOutput:
    start = np.asarray([0.0, 0.0], dtype=np.float32)
    order = turning_aware_order(
        points,
        start,
        0.0,
        TURN_RADIUS,
        visit_radius=visit_radius if use_visit_radius_planning else 0.0,
    )
    if two_opt_use_exec_rollout:
        def _exec_cost(cand: List[int]) -> float:
            return exec_rollout_length(
                points,
                cand,
                start,
                0.0,
                UAV_SPEED,
                MAX_TURN_RATE,
                DT,
                TURN_RADIUS,
                visit_radius=visit_radius,
                terminate_on_visit=early_terminate_on_visit or use_visit_radius_planning,
                turn_then_straight=turn_then_straight,
            )
        order = two_opt_improve_exec(order, _exec_cost)
    else:
        order = two_opt_improve(
            order,
            points,
            start,
            0.0,
            TURN_RADIUS,
            visit_radius=visit_radius if use_visit_radius_planning else 0.0,
        )
    entry_points = _resolve_entry_points(
        points,
        order,
        visit_radius,
        entry_point_opt,
        entry_point_k,
        turn_then_straight,
    )
    plan_len = route_cost_dubins(
        points,
        order,
        start,
        0.0,
        TURN_RADIUS,
        visit_radius=visit_radius if use_visit_radius_planning else 0.0,
        entry_points=entry_points,
    )

    pos = start.copy()
    heading = 0.0
    segments = [np.asarray([pos], dtype=np.float32)]
    infeasible = 0
    visited = 0
    for k, idx in enumerate(order):
        target_points = points if entry_points is None else entry_points
        goal = target_points[idx]
        goal_h = heading
        if k + 1 < len(order):
            goal_h = _heading_to(goal, target_points[order[k + 1]])
        seg = _connect_with_connector(
            connector,
            pos,
            heading,
            goal,
            goal_h,
            radius,
            rng,
            birds,
            obstacle_map,
            resolution_m,
            points[idx],
            visit_radius,
            early_terminate_on_visit,
            time_budget,
            turn_then_straight,
        )
        if len(seg) == 0:
            infeasible += 1
            continue
        if _segment_reaches(seg, points[idx], visit_radius):
            visited += 1
        heading = _heading_to(seg[-2], seg[-1]) if len(seg) >= 2 else heading
        segments.append(seg)
        pos = seg[-1]
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
    smoothness = _path_smoothness(path)
    total_t = exec_len / UAV_SPEED if UAV_SPEED > 1e-6 else float('nan')
    collisions, min_d = _evaluate_bird_metrics(path, birds, birds_vel, birds_mode, radius)
    obstacle_hits = _count_obstacle_hits(path, obstacle_map, resolution_m)
    total_points = len(points)
    coverage_rate = float(plan.visited_count) / float(total_points) if total_points > 0 else 0.0
    success = (
        plan.visited_count == total_points
        and plan.infeasible_segments == 0
        and obstacle_hits == 0
    )
    return EvalResult(
        name,
        path,
        planning_time,
        exec_len,
        plan.plan_length,
        smoothness,
        total_t,
        collisions,
        min_d,
        obstacle_hits,
        plan.infeasible_segments,
        plan.visited_count,
        total_points,
        coverage_rate,
        success,
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
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]
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
            f"Smooth={res.smoothness:.2f}  Success={int(res.success)}  Infeas={res.infeasible_segments}\n"
            f"Visited={res.visited_count}/{res.total_points} ({res.coverage_rate*100:.1f}%)  "
            f"ObsHits={res.obstacle_hits}  Coll={res.collisions}  MinD={res.min_distance:.1f}m"
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


def _evaluate_local_pairs(
    points: np.ndarray,
    order: List[int],
    connector: str,
    rng: np.random.Generator,
    radius: float,
    birds: np.ndarray,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
    visit_radius: float,
    early_terminate_on_visit: bool,
    entry_point_opt: str,
    entry_point_k: int,
    time_budget: Optional[float],
    turn_then_straight: bool,
    max_pairs: int,
) -> List[LocalPairResult]:
    if len(order) < 2:
        return []
    pairs = list(zip(order[:-1], order[1:]))
    if max_pairs > 0 and max_pairs < len(pairs):
        choice = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in choice]

    results = []
    for start_idx, goal_idx in pairs:
        start = points[start_idx]
        goal = points[goal_idx]
        start_heading = _heading_to(start, goal)
        target = goal
        if entry_point_opt == 'sample_circle' and visit_radius > 0.0 and entry_point_k > 0:
            entry_points = optimize_entry_points(
                points,
                [goal_idx],
                visit_radius=visit_radius,
                start_xy=start,
                start_heading=start_heading,
                speed=UAV_SPEED,
                max_turn_rate=MAX_TURN_RATE,
                dt=DT,
                turn_radius=TURN_RADIUS,
                entry_point_k=entry_point_k,
                turn_then_straight=turn_then_straight,
            )
            target = entry_points[goal_idx]
        goal_heading = _heading_to(target, goal)
        t0 = time.perf_counter()
        seg = _connect_with_connector(
            connector,
            start,
            start_heading,
            target,
            goal_heading,
            radius,
            rng,
            birds,
            obstacle_map,
            resolution_m,
            goal,
            visit_radius,
            early_terminate_on_visit,
            time_budget,
            turn_then_straight,
        )
        planning_time = time.perf_counter() - t0
        exec_len = path_length(seg)
        smoothness = _path_smoothness(seg)
        obstacle_hits = _count_obstacle_hits(seg, obstacle_map, resolution_m)
        success = _segment_reaches(seg, goal, visit_radius) and obstacle_hits == 0
        results.append(
            LocalPairResult(
                start_idx=start_idx,
                goal_idx=goal_idx,
                path=seg,
                planning_time=planning_time,
                exec_length=exec_len,
                smoothness=smoothness,
                obstacle_hits=obstacle_hits,
                success=success,
            )
        )
    return results


def _plot_local_pairs(
    points: np.ndarray,
    results: List[LocalPairResult],
    visit_radius: float,
    save_path: str,
    obstacle_map: Optional[np.ndarray],
    resolution_m: float,
) -> None:
    if not results:
        return
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        if obstacle_map is not None:
            h, w = obstacle_map.shape
            ext = [-w // 2 * resolution_m, w // 2 * resolution_m, -h // 2 * resolution_m, h // 2 * resolution_m]
            ax.imshow(obstacle_map.astype(np.float32), cmap='gray', alpha=0.35, extent=ext, origin='upper')
        if len(res.path) > 1:
            ax.plot(res.path[:, 0], res.path[:, 1], '-', lw=2.0, color='tab:blue')
        start = points[res.start_idx]
        goal = points[res.goal_idx]
        ax.scatter([start[0]], [start[1]], s=70, c='tab:green', marker='o', label='Start')
        ax.scatter([goal[0]], [goal[1]], s=70, c='orange', marker='*', label='Goal')
        if visit_radius > 0.0:
            ax.add_patch(plt.Circle((goal[0], goal[1]), visit_radius, fill=False, linestyle='--', color='tab:orange'))
        ax.set_aspect('equal')
        ax.set_title(
            f"Pair {res.start_idx}->{res.goal_idx}\n"
            f"L_exec={res.exec_length:.1f}m Smooth={res.smoothness:.2f} "
            f"Success={int(res.success)} ObsHits={res.obstacle_hits}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc='upper right', fontsize=8)
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
    parser.add_argument('--visit_radius', type=float, default=VISIT_RADIUS, help='Visit radius (m) for goal regions.')
    parser.add_argument('--use_visit_radius_planning', type=int, default=1, help='Enable visit-radius TSPN costs (0/1).')
    parser.add_argument('--early_terminate_on_visit', type=int, default=1, help='Terminate segments once inside visit radius (0/1).')
    parser.add_argument('--two_opt_use_exec_rollout', type=int, default=1, help='Use L_exec rollout for 2-opt (0/1).')
    parser.add_argument('--entry_point_opt', type=str, default='none', choices=['none', 'sample_circle'])
    parser.add_argument('--entry_point_K', type=int, default=16)
    parser.add_argument('--time_budget', type=float, default=0.2, help='Per-connector time budget (seconds).')
    parser.add_argument(
        '--connector',
        type=str,
        default='pdubins_rrtstar',
        choices=[
            'straight',
            'dubins_like',
            'rrtstar',
            'dubins_rrtstar',
            'pdubins_rrtstar',
            'goal_region_pdubins_rrtstar',
        ],
    )
    parser.add_argument('--local_pairs', type=int, default=0, help='Number of local point pairs to evaluate.')
    parser.add_argument('--local_save_path', type=str, default='offline_planner_local.png')
    args = parser.parse_args()

    use_visit_radius_planning = bool(args.use_visit_radius_planning)
    early_terminate_on_visit = bool(args.early_terminate_on_visit)
    two_opt_use_exec_rollout = bool(args.two_opt_use_exec_rollout)
    visit_radius = float(args.visit_radius) if (use_visit_radius_planning or early_terminate_on_visit) else 0.0
    turn_then_straight = True
    entry_point_opt = args.entry_point_opt
    entry_point_k = int(args.entry_point_K)
    time_budget = float(args.time_budget) if args.time_budget > 0 else None

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

    print(
        f'[Eval] visit_radius={visit_radius:.1f}m early_terminate={int(early_terminate_on_visit)} '
        f'entry_opt={entry_point_opt} entry_K={entry_point_k} connector={args.connector} '
        f'time_budget={time_budget if time_budget is not None else "none"}'
    )

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
                if n_obs == 0:
                    print('[WARNING] obstacle_map has 0 obstacle pixels; check threshold/CRS or DEM coverage.')
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

    connector_label = _connector_label(args.connector)
    methods = [
        (
            'Baseline1: Euclidean-order + simple turning',
            lambda pts: _baseline1(
                pts,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
                visit_radius=visit_radius,
                early_terminate_on_visit=early_terminate_on_visit,
                entry_point_opt=entry_point_opt,
                entry_point_k=entry_point_k,
                turn_then_straight=turn_then_straight,
            ),
        ),
        (
            'Baseline2: Euclidean-order + Dubins',
            lambda pts: _baseline2(
                pts,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
                visit_radius=visit_radius,
                early_terminate_on_visit=early_terminate_on_visit,
                turn_then_straight=turn_then_straight,
                entry_point_opt=entry_point_opt,
                entry_point_k=entry_point_k,
            ),
        ),
        (
            f'Baseline3: 2-opt turn-cost + {connector_label}',
            lambda pts: _baseline3(
                pts,
                rng=np.random.default_rng(args.seed + BASELINE3_SEED_OFFSET),
                birds=birds,
                radius=radius,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
                visit_radius=visit_radius,
                use_visit_radius_planning=use_visit_radius_planning,
                early_terminate_on_visit=early_terminate_on_visit,
                two_opt_use_exec_rollout=two_opt_use_exec_rollout,
                turn_then_straight=turn_then_straight,
                entry_point_opt=entry_point_opt,
                entry_point_k=entry_point_k,
                connector=args.connector,
                time_budget=time_budget,
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
            f'smooth={res.smoothness:.2f} success={int(res.success)} '
            f'collisions={res.collisions} infeasible={res.infeasible_segments} '
            f'min_distance={res.min_distance:.2f}m obstacle_hits={res.obstacle_hits} '
            f'visited={res.visited_count}/{res.total_points} coverage={res.coverage_rate*100:.1f}%'
        )

    context_path = args.context_path
    if not context_path:
        base, ext = os.path.splitext(args.save_path)
        context_path = f'{base}_context{ext or ".png"}'

    _plot_results(fire_points, birds, results, radius, args.save_path, obstacle_map, resolution_m)
    _plot_context(fire_points, birds, radius, context_path, obstacle_map, resolution_m)
    print(f'Saved: {args.save_path}')
    print(f'Saved context: {context_path}')

    if args.local_pairs > 0:
        local_order = nearest_neighbor_order(fire_points)
        local_results = _evaluate_local_pairs(
            fire_points,
            local_order,
            args.connector,
            rng=np.random.default_rng(args.seed + 77),
            radius=radius,
            birds=birds,
            obstacle_map=obstacle_map,
            resolution_m=resolution_m,
            visit_radius=visit_radius,
            early_terminate_on_visit=early_terminate_on_visit,
            entry_point_opt=entry_point_opt,
            entry_point_k=entry_point_k,
            time_budget=time_budget,
            turn_then_straight=turn_then_straight,
            max_pairs=int(args.local_pairs),
        )
        if local_results:
            _plot_local_pairs(
                fire_points,
                local_results,
                visit_radius=visit_radius,
                save_path=args.local_save_path,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
            )
            avg_len = float(np.mean([r.exec_length for r in local_results]))
            avg_smooth = float(np.mean([r.smoothness for r in local_results]))
            avg_plan = float(np.mean([r.planning_time for r in local_results]))
            succ_rate = float(np.mean([1.0 if r.success else 0.0 for r in local_results])) * 100.0
            print(
                f'[LocalPairs] count={len(local_results)} L_exec={avg_len:.2f}m '
                f'smooth={avg_smooth:.2f} plan={avg_plan:.3f}s success={succ_rate:.1f}% '
                f'saved={args.local_save_path}'
            )


if __name__ == '__main__':
    main()
