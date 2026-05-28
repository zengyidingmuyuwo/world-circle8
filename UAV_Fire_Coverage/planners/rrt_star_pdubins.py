import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dubins_utils import angle_diff, dubins_like_connect

DEFAULT_MAX_ITERS = 220
GOAL_BIAS_PROB = 0.22
REWIRE_EPS = 1e-6


@dataclass
class _Node:
    pos: np.ndarray
    heading: float
    parent: int
    cost: float


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


def _segment_hits_obstacle(a: np.ndarray, b: np.ndarray, obstacle_map: np.ndarray, resolution_m: float) -> bool:
    if obstacle_map is None:
        return False
    seg = b - a
    dist = float(np.linalg.norm(seg))
    if dist < 1e-6:
        cell = _world_to_grid(a, obstacle_map, resolution_m)
        return bool(cell is not None and obstacle_map[cell])
    step = max(1.0, float(resolution_m) * 0.5)
    n = max(1, int(math.ceil(dist / step)))
    for k in range(n + 1):
        p = a + seg * (k / n)
        cell = _world_to_grid(p, obstacle_map, resolution_m)
        if cell is not None and obstacle_map[cell]:
            return True
    return False


def _segment_clear(
    a: np.ndarray,
    b: np.ndarray,
    birds: np.ndarray,
    bird_radius: float,
    safety_margin: float = 20.0,
    obstacle_map: Optional[np.ndarray] = None,
    resolution_m: float = 50.0,
) -> bool:
    if obstacle_map is not None and _segment_hits_obstacle(a, b, obstacle_map, resolution_m):
        return False
    if birds is None or len(birds) == 0:
        return True
    seg = b - a
    seg_len2 = float(np.dot(seg, seg)) + 1e-9
    for c in birds:
        v = c - a
        t = float(np.clip(np.dot(v, seg) / seg_len2, 0.0, 1.0))
        p = a + t * seg
        if float(np.linalg.norm(c - p)) <= float(bird_radius + safety_margin):
            return False
    return True


def _path_clear(
    path_xy: np.ndarray,
    birds: np.ndarray,
    bird_radius: float,
    safety_margin: float = 20.0,
    obstacle_map: Optional[np.ndarray] = None,
    resolution_m: float = 50.0,
) -> bool:
    if len(path_xy) < 2:
        return True
    for i in range(len(path_xy) - 1):
        if not _segment_clear(
            path_xy[i],
            path_xy[i + 1],
            birds,
            bird_radius,
            safety_margin=safety_margin,
            obstacle_map=obstacle_map,
            resolution_m=resolution_m,
        ):
            return False
    return True


def pdubins_rrt_star_connect(
    start_xy: Sequence[float],
    start_heading: float,
    goal_xy: Sequence[float],
    goal_heading: float,
    radius: float,
    rng: np.random.Generator,
    birds_xy: np.ndarray,
    bird_radius: float,
    speed: float,
    max_turn_rate: float,
    dt: float,
    obstacle_map: Optional[np.ndarray] = None,
    resolution_m: float = 50.0,
    max_iters: int = DEFAULT_MAX_ITERS,
    step_size: float = 80.0,
    connect_threshold: float = 120.0,
    neighbor_radius: float = 180.0,
) -> Tuple[np.ndarray, float]:
    """P-Dubins-RRT*: XY sampling, strict collision-free connect/rewire."""
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    direct, _ = dubins_like_connect(
        start,
        start_heading,
        goal,
        goal_heading,
        speed=speed,
        max_turn_rate=max_turn_rate,
        dt=dt,
        turn_radius=max(speed / max(max_turn_rate, 1e-6), 1.0),
    )
    if _path_clear(direct, birds_xy, bird_radius, obstacle_map=obstacle_map, resolution_m=resolution_m):
        plen = float(np.linalg.norm(np.diff(direct, axis=0), axis=1).sum()) if len(direct) > 1 else 0.0
        return direct, plen

    nodes: List[_Node] = [_Node(pos=start.copy(), heading=float(start_heading), parent=-1, cost=0.0)]
    best_goal_idx = -1
    best_goal_cost = float('inf')

    def nearest_index(sample_xy: np.ndarray) -> int:
        d = [float(np.linalg.norm(n.pos - sample_xy)) for n in nodes]
        return int(np.argmin(d))

    def near_indices(new_xy: np.ndarray) -> List[int]:
        out = []
        for i, n in enumerate(nodes):
            if float(np.linalg.norm(n.pos - new_xy)) <= neighbor_radius:
                out.append(i)
        return out

    for _ in range(int(max_iters)):
        if rng.random() < GOAL_BIAS_PROB:
            sample = goal
        else:
            ang = rng.uniform(-math.pi, math.pi)
            # In polar sampling, sqrt(random) gives uniform area coverage in a disk.
            rr = radius * math.sqrt(rng.random())
            sample = np.asarray([rr * math.cos(ang), rr * math.sin(ang)], dtype=np.float32)

        i_near = nearest_index(sample)
        n_near = nodes[i_near]
        vec = sample - n_near.pos
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            continue

        desired_h = math.atan2(float(vec[1]), float(vec[0]))
        turn = float(np.clip(angle_diff(desired_h, n_near.heading), -max_turn_rate * dt, max_turn_rate * dt))
        new_h = n_near.heading + turn
        step = min(step_size, dist)
        new_xy = n_near.pos + np.asarray([math.cos(new_h), math.sin(new_h)], dtype=np.float32) * step

        if float(np.linalg.norm(new_xy)) > radius:
            continue
        if not _segment_clear(n_near.pos, new_xy, birds_xy, bird_radius, obstacle_map=obstacle_map, resolution_m=resolution_m):
            continue

        parent = i_near
        edge_cost = float(np.linalg.norm(new_xy - n_near.pos))
        best_cost = n_near.cost + edge_cost

        for i in near_indices(new_xy):
            cand = nodes[i]
            if not _segment_clear(cand.pos, new_xy, birds_xy, bird_radius, obstacle_map=obstacle_map, resolution_m=resolution_m):
                continue
            c = cand.cost + float(np.linalg.norm(new_xy - cand.pos))
            if c < best_cost:
                parent = i
                best_cost = c

        nodes.append(_Node(pos=new_xy, heading=float(new_h), parent=parent, cost=best_cost))
        i_new = len(nodes) - 1

        # strict rewire
        for i in near_indices(new_xy):
            if i == i_new:
                continue
            cand = nodes[i]
            c_new = nodes[i_new].cost + float(np.linalg.norm(cand.pos - new_xy))
            if c_new + REWIRE_EPS < cand.cost and _segment_clear(
                new_xy,
                cand.pos,
                birds_xy,
                bird_radius,
                obstacle_map=obstacle_map,
                resolution_m=resolution_m,
            ):
                nodes[i] = _Node(pos=cand.pos, heading=cand.heading, parent=i_new, cost=c_new)

        # strict connection to goal via full Dubins-like path
        if float(np.linalg.norm(new_xy - goal)) <= connect_threshold:
            path_to_goal, _ = dubins_like_connect(
                new_xy,
                nodes[i_new].heading,
                goal,
                goal_heading,
                speed=speed,
                max_turn_rate=max_turn_rate,
                dt=dt,
                turn_radius=max(speed / max(max_turn_rate, 1e-6), 1.0),
            )
            if _path_clear(path_to_goal, birds_xy, bird_radius, obstacle_map=obstacle_map, resolution_m=resolution_m):
                c_goal = nodes[i_new].cost + float(np.linalg.norm(np.diff(path_to_goal, axis=0), axis=1).sum())
                if c_goal < best_goal_cost:
                    best_goal_cost = c_goal
                    best_goal_idx = i_new

    if best_goal_idx == -1:
        # fallback: use nearest-to-goal node then Dubins-like connect
        d = [float(np.linalg.norm(n.pos - goal)) for n in nodes]
        best_goal_idx = int(np.argmin(d))

    chain = []
    cur = best_goal_idx
    while cur != -1:
        chain.append(nodes[cur].pos.copy())
        cur = nodes[cur].parent
    chain.reverse()

    tail, _ = dubins_like_connect(
        chain[-1], nodes[best_goal_idx].heading, goal, goal_heading,
        speed=speed, max_turn_rate=max_turn_rate, dt=dt,
        turn_radius=max(speed / max(max_turn_rate, 1e-6), 1.0),
    )
    if len(tail) > 1:
        chain.extend(list(tail[1:]))

    path = np.asarray(chain, dtype=np.float32)
    plen = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    return path, plen
