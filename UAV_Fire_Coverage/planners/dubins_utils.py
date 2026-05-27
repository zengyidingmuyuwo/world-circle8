import math
from typing import Iterable, Tuple

import numpy as np

# Begin blending from line-heading to goal-heading when within this multiple of turn radius.
HEADING_BLEND_FACTOR = 2.5


def wrap_angle(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(target: float, source: float) -> float:
    return wrap_angle(float(target) - float(source))


def path_length(path_xy: np.ndarray) -> float:
    if path_xy is None or len(path_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def _step_forward(pos: np.ndarray, heading: float, speed: float, dt: float) -> np.ndarray:
    return pos + np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32) * float(speed * dt)


def simple_turning_connect(
    start_xy: np.ndarray,
    start_heading: float,
    goal_xy: np.ndarray,
    speed: float,
    max_turn_rate: float,
    dt: float,
    goal_tolerance: float = 20.0,
    max_steps: int = 2000,
) -> Tuple[np.ndarray, float]:
    """Simple kinematic connector: steer toward goal with bounded turn rate."""
    pos = np.asarray(start_xy, dtype=np.float32).copy()
    goal = np.asarray(goal_xy, dtype=np.float32)
    heading = float(start_heading)
    turn_step = float(max_turn_rate * dt)
    pts = [pos.copy()]

    init_dist = float(np.linalg.norm(goal - pos))
    dyn_limit = max(80, int(init_dist / max(speed * dt, 1e-6) * 4.0) + 120)
    step_limit = min(int(max_steps), dyn_limit)
    for _ in range(step_limit):
        d = goal - pos
        dist = float(np.linalg.norm(d))
        if dist <= goal_tolerance:
            break
        desired = math.atan2(float(d[1]), float(d[0]))
        delta = angle_diff(desired, heading)
        heading = wrap_angle(heading + float(np.clip(delta, -turn_step, turn_step)))
        pos = _step_forward(pos, heading, speed, dt)
        pts.append(pos.copy())

    if np.linalg.norm(goal - pos) > 1e-3:
        pts.append(goal.copy())
    return np.asarray(pts, dtype=np.float32), heading


def dubins_approx_length(
    start_xy: np.ndarray,
    start_heading: float,
    goal_xy: np.ndarray,
    goal_heading: float,
    turn_radius: float,
) -> float:
    """Fast relaxed Dubins length approximation."""
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    d = float(np.linalg.norm(goal - start))
    line_h = math.atan2(float(goal[1] - start[1]), float(goal[0] - start[0]))
    a1 = abs(angle_diff(line_h, start_heading))
    a2 = abs(angle_diff(goal_heading, line_h))
    return d + float(turn_radius) * (a1 + a2)


def dubins_like_connect(
    start_xy: np.ndarray,
    start_heading: float,
    goal_xy: np.ndarray,
    goal_heading: float,
    speed: float,
    max_turn_rate: float,
    dt: float,
    turn_radius: float,
    goal_tolerance: float = 20.0,
    max_steps: int = 2500,
) -> Tuple[np.ndarray, float]:
    """Relaxed Dubins-style connector using heading blending near the goal."""
    pos = np.asarray(start_xy, dtype=np.float32).copy()
    goal = np.asarray(goal_xy, dtype=np.float32)
    heading = float(start_heading)
    turn_step = float(max_turn_rate * dt)
    pts = [pos.copy()]

    init_dist = float(np.linalg.norm(goal - pos))
    dyn_limit = max(100, int(init_dist / max(speed * dt, 1e-6) * 5.0) + 150)
    step_limit = min(int(max_steps), dyn_limit)
    for _ in range(step_limit):
        vec = goal - pos
        dist = float(np.linalg.norm(vec))
        if dist <= goal_tolerance:
            break

        line_h = math.atan2(float(vec[1]), float(vec[0]))
        if dist > HEADING_BLEND_FACTOR * turn_radius:
            desired = line_h
        else:
            alpha = float(
                np.clip(
                    (HEADING_BLEND_FACTOR * turn_radius - dist) / (HEADING_BLEND_FACTOR * turn_radius),
                    0.0,
                    1.0,
                )
            )
            sx, sy = math.cos(line_h), math.sin(line_h)
            gx, gy = math.cos(goal_heading), math.sin(goal_heading)
            mix = np.array([(1 - alpha) * sx + alpha * gx, (1 - alpha) * sy + alpha * gy], dtype=np.float32)
            desired = math.atan2(float(mix[1]), float(mix[0]))

        delta = angle_diff(desired, heading)
        heading = wrap_angle(heading + float(np.clip(delta, -turn_step, turn_step)))
        pos = _step_forward(pos, heading, speed, dt)
        pts.append(pos.copy())

    if np.linalg.norm(goal - pos) > 1e-3:
        pts.append(goal.copy())
    return np.asarray(pts, dtype=np.float32), heading


def stitch_paths(paths: Iterable[np.ndarray]) -> np.ndarray:
    out = []
    for idx, p in enumerate(paths):
        seg = np.asarray(p, dtype=np.float32)
        if len(seg) == 0:
            continue
        if idx > 0:
            seg = seg[1:]
        out.append(seg)
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)
