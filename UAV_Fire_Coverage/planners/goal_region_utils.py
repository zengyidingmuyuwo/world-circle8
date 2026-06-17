import math
from typing import List

import numpy as np

from .dubins_utils import dubins_approx_length, dubins_like_connect, path_length


def _heading_to(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return float(math.atan2(float(d[1]), float(d[0])))


def sample_circle_points(center_xy: np.ndarray, radius: float, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    center = np.asarray(center_xy, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, int(k), endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    return center[None, :] + pts * float(radius)


def optimize_entry_points(
    points_xy: np.ndarray,
    order: List[int],
    visit_radius: float,
    start_xy: np.ndarray,
    start_heading: float,
    speed: float,
    max_turn_rate: float,
    dt: float,
    turn_radius: float,
    entry_point_k: int = 16,
    turn_then_straight: bool = False,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    entry_points = points.copy()
    if visit_radius <= 0.0 or entry_point_k <= 0 or len(order) == 0:
        return entry_points

    pos = np.asarray(start_xy, dtype=np.float32).copy()
    heading = float(start_heading)
    for idx_pos, idx in enumerate(order):
        center = points[idx]
        next_center = points[order[idx_pos + 1]] if idx_pos + 1 < len(order) else None
        candidates = sample_circle_points(center, visit_radius, entry_point_k)
        if len(candidates) == 0:
            continue

        best_cost = float('inf')
        best_point = center
        best_seg = None
        best_heading = heading
        for cand in candidates:
            goal_h = heading if next_center is None else _heading_to(cand, next_center)
            seg, seg_heading = dubins_like_connect(
                pos,
                heading,
                cand,
                goal_h,
                speed=speed,
                max_turn_rate=max_turn_rate,
                dt=dt,
                turn_radius=turn_radius,
                visit_radius=0.0,
                terminate_on_visit=False,
                turn_then_straight=turn_then_straight,
            )
            cost = path_length(seg)
            if next_center is not None:
                line_h = _heading_to(cand, next_center)
                cost += dubins_approx_length(cand, seg_heading, next_center, line_h, turn_radius)
            if cost < best_cost:
                best_cost = cost
                best_point = cand
                best_seg = seg
                best_heading = seg_heading

        entry_points[idx] = best_point
        if best_seg is None:
            best_seg, best_heading = dubins_like_connect(
                pos,
                heading,
                best_point,
                heading,
                speed=speed,
                max_turn_rate=max_turn_rate,
                dt=dt,
                turn_radius=turn_radius,
                visit_radius=0.0,
                terminate_on_visit=False,
                turn_then_straight=turn_then_straight,
            )
        if len(best_seg) > 0:
            pos = best_seg[-1]
        else:
            pos = best_point
        if len(best_seg) >= 2:
            heading = _heading_to(best_seg[-2], best_seg[-1])
        else:
            heading = best_heading

    return entry_points
