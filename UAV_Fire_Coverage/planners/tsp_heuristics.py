import math
from typing import Callable, List

import numpy as np

from .dubins_utils import angle_diff, dubins_approx_length

IMPROVEMENT_EPS = 1e-6


def nearest_neighbor_order(points_xy: np.ndarray, start_xy: np.ndarray = None) -> List[int]:
    points = np.asarray(points_xy, dtype=np.float32)
    n = len(points)
    if n == 0:
        return []
    remain = set(range(n))
    cur = np.asarray([0.0, 0.0], dtype=np.float32) if start_xy is None else np.asarray(start_xy, dtype=np.float32)
    order = []
    while remain:
        nxt = min(remain, key=lambda j: float(np.linalg.norm(points[j] - cur)))
        order.append(nxt)
        cur = points[nxt]
        remain.remove(nxt)
    return order


def turning_aware_order(
    points_xy: np.ndarray,
    start_xy: np.ndarray,
    start_heading: float,
    turn_radius: float,
) -> List[int]:
    points = np.asarray(points_xy, dtype=np.float32)
    n = len(points)
    if n == 0:
        return []
    remain = set(range(n))
    cur = np.asarray(start_xy, dtype=np.float32)
    heading = float(start_heading)
    order = []
    while remain:
        best = None
        best_cost = float('inf')
        best_heading = heading
        for j in remain:
            vec = points[j] - cur
            th = math.atan2(float(vec[1]), float(vec[0]))
            d = float(np.linalg.norm(vec))
            c = d + float(turn_radius) * abs(angle_diff(th, heading))
            if c < best_cost:
                best_cost = c
                best = j
                best_heading = th
        order.append(best)
        cur = points[best]
        heading = best_heading
        remain.remove(best)
    return order


def route_cost_dubins(points_xy: np.ndarray, order: List[int], start_xy: np.ndarray, start_heading: float, turn_radius: float) -> float:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(order) == 0:
        return 0.0
    cur = np.asarray(start_xy, dtype=np.float32)
    heading = float(start_heading)
    total = 0.0
    for k, idx in enumerate(order):
        goal = pts[idx]
        if k + 1 < len(order):
            nvec = pts[order[k + 1]] - goal
            goal_h = math.atan2(float(nvec[1]), float(nvec[0]))
        else:
            end_vec = goal - cur
            goal_h = math.atan2(float(end_vec[1]), float(end_vec[0]))
        total += dubins_approx_length(cur, heading, goal, goal_h, turn_radius)
        heading = goal_h
        cur = goal
    return float(total)


def two_opt_improve(
    order: List[int],
    points_xy: np.ndarray,
    start_xy: np.ndarray,
    start_heading: float,
    turn_radius: float,
    max_passes: int = 20,
) -> List[int]:
    best = list(order)
    if len(best) < 3:
        return best
    best_cost = route_cost_dubins(points_xy, best, start_xy, start_heading, turn_radius)
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                c = route_cost_dubins(points_xy, cand, start_xy, start_heading, turn_radius)
                if c + IMPROVEMENT_EPS < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
    return best
