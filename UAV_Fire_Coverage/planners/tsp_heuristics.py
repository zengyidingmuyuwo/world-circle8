import math
from typing import Callable, List

import numpy as np

from .dubins_utils import angle_diff, dubins_like_connect, path_length

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
    visit_radius: float = 0.0,
) -> List[int]:
    points = np.asarray(points_xy, dtype=np.float32)
    n = len(points)
    if n == 0:
        return []
    remain = set(range(n))
    cur = np.asarray(start_xy, dtype=np.float32)
    cur_radius = 0.0
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
            if visit_radius > 0.0:
                d = max(0.0, d - (cur_radius + visit_radius))
            c = d + float(turn_radius) * abs(angle_diff(th, heading))
            if c < best_cost:
                best_cost = c
                best = j
                best_heading = th
        order.append(best)
        cur = points[best]
        cur_radius = visit_radius
        heading = best_heading
        remain.remove(best)
    return order


def route_cost_dubins(
    points_xy: np.ndarray,
    order: List[int],
    start_xy: np.ndarray,
    start_heading: float,
    turn_radius: float,
    visit_radius: float = 0.0,
) -> float:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(order) == 0:
        return 0.0
    cur = np.asarray(start_xy, dtype=np.float32)
    cur_radius = 0.0
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
        d = float(np.linalg.norm(goal - cur))
        if visit_radius > 0.0:
            d = max(0.0, d - (cur_radius + visit_radius))
        line_h = math.atan2(float(goal[1] - cur[1]), float(goal[0] - cur[0]))
        a1 = abs(angle_diff(line_h, heading))
        a2 = abs(angle_diff(goal_h, line_h))
        total += d + float(turn_radius) * (a1 + a2)
        heading = goal_h
        cur = goal
        cur_radius = visit_radius
    return float(total)


def exec_rollout_length(
    points_xy: np.ndarray,
    order: List[int],
    start_xy: np.ndarray,
    start_heading: float,
    speed: float,
    max_turn_rate: float,
    dt: float,
    turn_radius: float,
    visit_radius: float = 0.0,
    terminate_on_visit: bool = False,
    turn_then_straight: bool = False,
) -> float:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(order) == 0:
        return 0.0
    cur = np.asarray(start_xy, dtype=np.float32)
    heading = float(start_heading)
    total = 0.0
    for k, idx in enumerate(order):
        goal = pts[idx]
        goal_h = heading
        if k + 1 < len(order):
            nvec = pts[order[k + 1]] - goal
            goal_h = math.atan2(float(nvec[1]), float(nvec[0]))
        seg, heading = dubins_like_connect(
            cur,
            heading,
            goal,
            goal_h,
            speed=speed,
            max_turn_rate=max_turn_rate,
            dt=dt,
            turn_radius=turn_radius,
            visit_radius=visit_radius,
            terminate_on_visit=terminate_on_visit,
            turn_then_straight=turn_then_straight,
        )
        total += path_length(seg)
        if len(seg) > 0:
            cur = seg[-1]
        else:
            cur = goal
    return float(total)


def two_opt_improve(
    order: List[int],
    points_xy: np.ndarray,
    start_xy: np.ndarray,
    start_heading: float,
    turn_radius: float,
    max_passes: int = 20,
    visit_radius: float = 0.0,
) -> List[int]:
    best = list(order)
    if len(best) < 3:
        return best
    best_cost = route_cost_dubins(points_xy, best, start_xy, start_heading, turn_radius, visit_radius=visit_radius)
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                c = route_cost_dubins(points_xy, cand, start_xy, start_heading, turn_radius, visit_radius=visit_radius)
                if c + IMPROVEMENT_EPS < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
    return best


def two_opt_improve_exec(
    order: List[int],
    exec_cost_fn: Callable[[List[int]], float],
    max_passes: int = 12,
) -> List[int]:
    best = list(order)
    if len(best) < 3:
        return best
    best_cost = exec_cost_fn(best)
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                c = exec_cost_fn(cand)
                if c + IMPROVEMENT_EPS < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
    return best
