import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int]
WorldPoint = Tuple[float, float]


@dataclass
class PlanResult:
    cost_matrix: np.ndarray
    tsp_order: List[int]
    ordered_fire_indices: List[int]
    waypoints: np.ndarray
    segment_paths: Dict[Tuple[int, int], np.ndarray]


class DronePlanner:
    """Global planner: pairwise A* cost matrix + greedy TSP + dense waypoints."""

    def __init__(
        self,
        obstacle_map: Optional[np.ndarray] = None,
        resolution_m: float = 50.0,
        waypoint_spacing_m: float = 50.0,
        max_line_steps: int = 10000,
    ):
        self.obstacle_map = None if obstacle_map is None else np.asarray(obstacle_map, dtype=bool)
        self.resolution_m = float(resolution_m)
        self.waypoint_spacing_m = float(max(1.0, waypoint_spacing_m))
        self.max_line_steps = int(max_line_steps)
        if self.obstacle_map is not None:
            self._h, self._w = self.obstacle_map.shape
            self._cx, self._cy = self._w // 2, self._h // 2
        else:
            self._h = self._w = self._cx = self._cy = 0

    def plan(self, start: Sequence[float], fire_points: np.ndarray) -> PlanResult:
        nodes = np.vstack([np.asarray(start, dtype=np.float32), np.asarray(fire_points, dtype=np.float32)])
        n = len(nodes)
        cost = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(cost, 0.0)
        paths: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(n):
            for j in range(i + 1, n):
                p, c = self._path_and_cost(nodes[i], nodes[j])
                cost[i, j] = cost[j, i] = float(c)
                paths[(i, j)] = p
                paths[(j, i)] = p[::-1].copy()

        order = self._greedy_tsp(cost)
        ordered_fire = [idx - 1 for idx in order[1:]]
        waypoints = self._stitch_paths(order, paths)
        return PlanResult(
            cost_matrix=cost,
            tsp_order=order,
            ordered_fire_indices=ordered_fire,
            waypoints=waypoints,
            segment_paths=paths,
        )

    def _path_and_cost(self, start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.obstacle_map is None:
            path = self._line_path(start, goal)
            return path, float(np.linalg.norm(goal - start))
        path = self._astar_world(start, goal)
        if len(path) == 0:
            fallback = self._line_path(start, goal)
            return fallback, float(np.linalg.norm(goal - start) * 5.0)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return path, float(seg.sum()) if len(seg) else 0.0

    def _line_path(self, start: np.ndarray, goal: np.ndarray, spacing_m: Optional[float] = None) -> np.ndarray:
        dist = float(np.linalg.norm(goal - start))
        if dist < 1e-6:
            return np.asarray([start], dtype=np.float32)
        spacing = self.waypoint_spacing_m if spacing_m is None else float(max(1.0, spacing_m))
        steps = max(2, int(math.ceil(dist / spacing)) + 1)
        if steps > self.max_line_steps:
            raise ValueError(
                f"Steps太大了: {steps}，请检查start {start} 和 goal {goal} 的坐标系是否一致！"
            )
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        return (start[None, :] + (goal - start)[None, :] * t).astype(np.float32)

    def _greedy_tsp(self, cost: np.ndarray) -> List[int]:
        n = cost.shape[0]
        remain = set(range(1, n))
        order = [0]
        cur = 0
        while remain:
            nxt = min(remain, key=lambda j: cost[cur, j])
            order.append(nxt)
            remain.remove(nxt)
            cur = nxt
        return order

    def _stitch_paths(self, order: List[int], paths: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
        pts: List[np.ndarray] = []
        for i in range(len(order) - 1):
            seg = paths[(order[i], order[i + 1])]
            if i > 0 and len(seg):
                seg = seg[1:]
            pts.append(seg)
        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(pts, axis=0).astype(np.float32)

    def _astar_world(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        s = self._world_to_grid(start)
        g = self._world_to_grid(goal)
        if s is None or g is None:
            return np.zeros((0, 2), dtype=np.float32)
        if self._is_obstacle_cell(s) or self._is_obstacle_cell(g):
            return np.zeros((0, 2), dtype=np.float32)
        path_cells = self._astar_cells(s, g)
        if not path_cells:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray([self._grid_to_world(c) for c in path_cells], dtype=np.float32)

    def _astar_cells(self, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def h(a: GridPoint, b: GridPoint) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(0.0, start)]
        g_score = {start: 0.0}
        parent: Dict[GridPoint, GridPoint] = {}
        closed = set()

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                return self._reconstruct(parent, goal)
            closed.add(cur)
            for dr, dc in neighbors:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self._in_bounds(nxt) or self._is_obstacle_cell(nxt):
                    continue
                step = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
                tentative = g_score[cur] + step
                if tentative < g_score.get(nxt, float('inf')):
                    parent[nxt] = cur
                    g_score[nxt] = tentative
                    heapq.heappush(open_heap, (tentative + h(nxt, goal), nxt))
        return []

    def _reconstruct(self, parent: Dict[GridPoint, GridPoint], goal: GridPoint) -> List[GridPoint]:
        out = [goal]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        out.reverse()
        return out

    def _world_to_grid(self, p: Sequence[float]) -> Optional[GridPoint]:
        if self.obstacle_map is None:
            return None
        x, y = float(p[0]), float(p[1])
        j = int(self._cx + x / self.resolution_m)
        i = int(self._cy - y / self.resolution_m)
        if not (0 <= i < self._h and 0 <= j < self._w):
            return None
        return i, j

    def _grid_to_world(self, cell: GridPoint) -> WorldPoint:
        i, j = cell
        x = (j - self._cx + 0.5) * self.resolution_m
        y = (self._cy - i + 0.5) * self.resolution_m
        return float(x), float(y)

    def _in_bounds(self, cell: GridPoint) -> bool:
        i, j = cell
        return 0 <= i < self._h and 0 <= j < self._w

    def _is_obstacle_cell(self, cell: GridPoint) -> bool:
        i, j = cell
        return bool(self.obstacle_map[i, j])