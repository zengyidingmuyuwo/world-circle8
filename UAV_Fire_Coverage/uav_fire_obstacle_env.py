"""
UAV fire-coverage environment with elevation-based obstacle avoidance.

Extends :class:`UAVFireEnv` by adding an obstacle layer derived from a
digital elevation model (DEM).  Pixels with elevation ≥ 2000 m are treated
as obstacles.

Additional state features (appended to the base state vector)
--------------------------------------------------------------
  8 × obstacle-distance sensors — one per cardinal / inter-cardinal direction,
  each normalised to [0, 1] by MAX_SENSOR_RANGE.
  Sensor angle offsets (relative to East):  0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

Additional rewards / penalties
-------------------------------
  PENALTY_COLLISION  (terminal)  — UAV enters an obstacle pixel
  PENALTY_PROXIMITY  — proportional to closeness to nearest obstacle
"""

import os
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_TUPLE_5 = True
except ImportError:
    import gym
    from gym import spaces
    _GYM_TUPLE_5 = False

from uav_fire_env import UAVFireEnv
from global_planner import DronePlanner


class UAVFireObstacleEnv(UAVFireEnv):
    """Single fixed-wing UAV fire-coverage + obstacle-avoidance environment."""

    NUM_SENSORS       = 8        # directional distance sensors
    MAX_SENSOR_RANGE  = 600.0    # metres
    RAY_STEP_M        = 20.0     # metres per ray-casting step

    PENALTY_COLLISION = -50.0
    PENALTY_PROXIMITY = -2.0     # multiplied by exp(-dist/scale)
    PROXIMITY_SCALE   = 80.0     # metres

    def __init__(self, fire_points, radius,
                 obstacle_map=None, resolution_m=50.0,
                 num_nearest=6, return_dict_obs=False, algorithm_name='RL',
                 env_name=None, lat_center=None, lon_center=None,
                 elevation_threshold=2000.0, dem_query_metadata=None,
                 radar_range_m=None, num_birds=3):
        """
        Parameters
        ----------
        fire_points : array-like, shape (N, 2)
        radius : float
        obstacle_map : np.ndarray (H, W) bool, optional
            Binary obstacle grid.  ``True`` = obstacle (elevation ≥ 2000 m).
            If *None*, the environment has no obstacles (same as UAVFireEnv).
        resolution_m : float
            Side length of each obstacle-map pixel in metres (default 50).
        num_nearest : int
        """
        super(UAVFireObstacleEnv, self).__init__(
            fire_points=fire_points,
            radius=radius,
            num_nearest=num_nearest,
            return_dict_obs=return_dict_obs,
            algorithm_name=algorithm_name,
            env_name=env_name,
            radar_range_m=radar_range_m,
            num_birds=num_birds,
        )

        self.obstacle_map  = obstacle_map   # (H, W) bool or None
        self.resolution_m  = float(resolution_m)
        self.lat_center = float(lat_center) if lat_center is not None else None
        self.lon_center = float(lon_center) if lon_center is not None else None
        self.elevation_threshold = float(elevation_threshold)
        self.dem_query_metadata = dem_query_metadata

        # Extend state dimension with NUM_SENSORS obstacle distances
        extra = self.NUM_SENSORS
        self.state_dim += extra
        self._planner = DronePlanner(obstacle_map=self.obstacle_map, resolution_m=self.resolution_m)
        os.makedirs(self.TRAJECTORY_RESULTS_DIR, exist_ok=True)
        if self.return_dict_obs:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32),
                'vector': spaces.Box(low=-1.0, high=1.0, shape=(self.vector_dim,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.state_dim + self.vector_dim,), dtype=np.float32
            )

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset; if the centroid-based start is inside an obstacle, fall back
        to the circle centre (0, 0) to avoid an immediate collision."""
        result = super().reset(seed=seed, options=options)
        if self._at_obstacle(self.pos):
            self.pos = np.zeros(2, dtype=np.float32)
            self._prev_min_dist = self._min_dist_to_nearest()
            self._plan_waypoints()
            if _GYM_TUPLE_5:
                return self._get_obs(), {}
            return self._get_obs()
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        if self._done:
            if _GYM_TUPLE_5:
                return self._get_obs(), 0.0, True, False, {}
            return self._get_obs(), 0.0, True, {}

        # ── Physics (same as base) ───────────────────────────────────────────
        delta = float(np.asarray(action).flat[0])
        delta = np.clip(delta, -1.0, 1.0) * self.MAX_TURN_RATE
        self.heading = (self.heading + delta) % (2.0 * np.pi)
        control_displacement = self.STEP_SIZE * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )
        wind_velocity = self._compute_wind_velocity()
        wind_displacement = wind_velocity * self.DT
        self.pos = self.pos + control_displacement + wind_displacement
        self._update_birds()
        self._trajectory.append(self.pos.copy())
        self._register_trajectory_for_snapshot()
        self.step_count += 1

        # ── Collision check ──────────────────────────────────────────────────
        reward = self.REWARD_STEP
        elev_m = self._query_elevation_m(self.pos)
        hit_mountain = (elev_m is not None and elev_m > self.elevation_threshold)
        hit_obstacle = self._at_obstacle(self.pos)
        bird_hit = self._bird_collision()
        if hit_mountain or hit_obstacle or bird_hit:
            reward += self.PENALTY_COLLISION
            if bird_hit:
                reward += self.PENALTY_BIRD_COLLISION
            self._done = True
            self._current_ep_score += float(reward)
            self._finalize_episode_record(float(np.sum(self.visited)) / self.n_fire)
            info = {
                'visited_count': int(np.sum(self.visited)),
                'total_fire_points': self.n_fire,
                'step': self.step_count,
                'coverage_rate': float(np.sum(self.visited)) / self.n_fire,
                'collision': True,
                'mountain_collision': bool(hit_mountain or hit_obstacle),
                'bird_collision': bool(bird_hit),
                'elevation_m': float(elev_m) if elev_m is not None else None,
            }
            if _GYM_TUPLE_5:
                return self._get_obs(), float(reward), True, False, info
            return self._get_obs(), float(reward), True, info

        # ── Proximity penalty ────────────────────────────────────────────────
        min_dist = self._min_obstacle_dist()
        if min_dist < self.PROXIMITY_SCALE * 3:
            reward += self.PENALTY_PROXIMITY * np.exp(-min_dist / self.PROXIMITY_SCALE)

        # ── Visit, shaping & boundary (same as base) ─────────────────────────
        n_before = int(np.sum(self.visited))
        reward  += self._check_visits()
        self._register_visit_for_snapshot()
        reward  += self._shaping_reward(n_before)
        reward  += self._boundary_penalty()
        reward  += self._waypoint_reward()
        off_path = self._is_off_path()

        done = self._done or bool(np.all(self.visited)) or (self.step_count >= self.MAX_STEPS)
        if np.all(self.visited):
            reward += self.REWARD_COMPLETE
        self._current_ep_score += float(reward)
        if done:
            self._finalize_episode_record(float(np.sum(self.visited)) / self.n_fire)
        self._done = done

        info = {
            'visited_count': int(np.sum(self.visited)),
            'total_fire_points': self.n_fire,
            'step': self.step_count,
            'coverage_rate': float(np.sum(self.visited)) / self.n_fire,
            'collision': False,
            'mountain_collision': False,
            'bird_collision': False,
            'elevation_m': float(elev_m) if elev_m is not None else None,
            'off_path': bool(off_path),
        }
        if _GYM_TUPLE_5:
            return self._get_obs(), float(reward), done, False, info
        return self._get_obs(), float(reward), done, info

    def _finalize_episode_record(self, coverage_rate):
        self._episode_count += 1
        class_name = self.__class__.__name__
        pid = os.getpid()
        if self._episode_count == 1:
            initial_path = os.path.join(
                self.TRAJECTORY_RESULTS_DIR,
                f'initial_{self.algorithm_name}_{self.env_name}_PID{pid}.png',
            )
            self._save_trajectory_snapshot(
                save_path=initial_path,
                coverage_rate=coverage_rate,
                title_prefix='Initial Episode Trajectory',
            )
        if self._current_ep_score > self._best_ep_score:
            self._best_ep_score = self._current_ep_score
            best_path = os.path.join(
                self.TRAJECTORY_RESULTS_DIR,
                f'best_{self.algorithm_name}_{self.env_name}_PID{pid}.png',
            )
            self._save_trajectory_snapshot(
                save_path=best_path,
                coverage_rate=coverage_rate,
                title_prefix='Best Episode Trajectory',
            )
            print(f'New best trajectory saved with score: {self._current_ep_score:.2f}')

    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        base_obs = super()._get_obs()
        sensors  = self._obstacle_sensors()
        if self.return_dict_obs:
            img = np.concatenate([base_obs['image'], sensors]).astype(np.float32)
            return {'image': img, 'vector': base_obs['vector']}
        return np.concatenate([base_obs, sensors]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────

    def _local_to_latlon(self, pos):
        if self.lat_center is None or self.lon_center is None:
            return None
        east_m, north_m = float(pos[0]), float(pos[1])
        R = 111_000.0
        lat = self.lat_center + north_m / R
        lon = self.lon_center + east_m / (R * np.cos(np.radians(self.lat_center)))
        return float(lat), float(lon)

    def _query_elevation_m(self, pos):
        meta = self.dem_query_metadata
        if not meta:
            return None
        ll = self._local_to_latlon(pos)
        if ll is None:
            return None
        lat, lon = ll
        to_raster = meta.get('to_raster')
        if to_raster is None:
            return None
        try:
            rx, ry = to_raster.transform(lon, lat)
            rowcol = meta.get('rowcol')
            if rowcol is None:
                return None
            row, col = rowcol(meta['transform'], rx, ry)
            if row < 0 or col < 0 or row >= meta['height'] or col >= meta['width']:
                return None
            val = float(meta['elevation'][row, col])
            nodata = meta.get('nodata')
            if nodata is not None and np.isclose(val, nodata):
                return None
            return val
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────

    def _obstacle_sensors(self):
        """Compute NUM_SENSORS normalised obstacle-distance readings."""
        sensors = np.ones(self.NUM_SENSORS, dtype=np.float32)
        if self.obstacle_map is None:
            return sensors
        for i in range(self.NUM_SENSORS):
            angle = i * (2.0 * np.pi / self.NUM_SENSORS)  # absolute angle from East
            d = self._ray_cast(self.pos, angle)
            sensors[i] = float(np.clip(d / self.MAX_SENSOR_RANGE, 0.0, 1.0))
        return sensors

    def _ray_cast(self, start, angle):
        """Return distance (metres) to the nearest obstacle along *angle*."""
        if self.obstacle_map is None:
            return self.MAX_SENSOR_RANGE
        dx = np.cos(angle) * self.RAY_STEP_M
        dy = np.sin(angle) * self.RAY_STEP_M
        pos = start.astype(float).copy()
        for step in range(int(self.MAX_SENSOR_RANGE / self.RAY_STEP_M)):
            pos[0] += dx
            pos[1] += dy
            if self._at_obstacle(pos):
                return step * self.RAY_STEP_M
        return self.MAX_SENSOR_RANGE

    def _at_obstacle(self, pos):
        """Return True if *pos* (local metric) falls inside an obstacle cell."""
        if self.obstacle_map is None:
            return False
        H, W = self.obstacle_map.shape
        cx, cy = W // 2, H // 2
        j = int(cx + pos[0] / self.resolution_m)
        i = int(cy - pos[1] / self.resolution_m)   # y-axis is north (up)
        if 0 <= i < H and 0 <= j < W:
            return bool(self.obstacle_map[i, j])
        return False

    def _min_obstacle_dist(self):
        """Minimum distance to any obstacle in the 8 sensor directions."""
        if self.obstacle_map is None:
            return self.MAX_SENSOR_RANGE
        sensors = self._obstacle_sensors()
        return float(np.min(sensors) * self.MAX_SENSOR_RANGE)

    def _save_trajectory_snapshot(self, save_path, coverage_rate, title_prefix):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            abs_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'wb') as f:
                f.write(b'')
            print(f'[Trajectory] Matplotlib unavailable, created placeholder: {abs_path}')
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        if self.obstacle_map is not None:
            H, W = self.obstacle_map.shape
            half_w = 0.5 * W * self.resolution_m
            half_h = 0.5 * H * self.resolution_m
            ext = [-half_w, half_w, -half_h, half_h]
            # Build a float copy and mask out pixels outside the mission circle
            # so only in-bounds terrain is visible (prevents full-raster noise fill).
            cx_px, cy_px = W // 2, H // 2
            ys_px, xs_px = np.ogrid[:H, :W]
            x_m = (xs_px - cx_px) * self.resolution_m
            y_m = (cy_px - ys_px) * self.resolution_m   # north-up
            outside_circle = (x_m ** 2 + y_m ** 2) > self.radius ** 2
            obs_float = self.obstacle_map.astype(np.float32)
            obs_float[outside_circle] = np.nan            # transparent outside
            ax.imshow(obs_float, cmap='Greys', vmin=0.0, vmax=1.0,
                      alpha=0.85, extent=ext, origin='upper', zorder=1)

        ax.add_patch(mpatches.Circle((0, 0), self.radius, fill=False, color='steelblue', lw=2))

        unv = self.fire_points[~self.visited]
        vis = self.fire_points[self.visited]
        if len(unv):
            ax.scatter(unv[:, 0], unv[:, 1], c='red', s=30, zorder=3, label='Unvisited')
        if len(vis):
            ax.scatter(vis[:, 0], vis[:, 1], c='limegreen', s=30, zorder=3, label='Visited')

        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory, dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.0, alpha=0.8, label='Trajectory')

        # Bird flock positions
        if self.num_birds > 0 and len(self._birds_pos):
            ax.scatter(self._birds_pos[:, 0], self._birds_pos[:, 1],
                       c='red', s=60, marker='^', zorder=5, label='Birds')

        lim = self.radius * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(
            f'{title_prefix} | score={self._current_ep_score:.2f} | '
            f'coverage={coverage_rate * 100:.1f}%'
        )
        abs_path = os.path.abspath(save_path)
        plt.savefig(abs_path, dpi=300)
        plt.close(fig)
        print(f'[Trajectory] Saved snapshot: {abs_path}')

    # ─────────────────────────────────────────────────────────────────────────

    def render(self, mode='human'):
        """Visualise environment including obstacle map."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return

        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(7, 7))
            plt.ion()

        ax = self._ax
        ax.clear()

        # Obstacle map background — clipped to mission circle
        if self.obstacle_map is not None:
            H, W = self.obstacle_map.shape
            half_w = 0.5 * W * self.resolution_m
            half_h = 0.5 * H * self.resolution_m
            ext = [-half_w, half_w, -half_h, half_h]
            cx_px, cy_px = W // 2, H // 2
            ys_px, xs_px = np.ogrid[:H, :W]
            x_m = (xs_px - cx_px) * self.resolution_m
            y_m = (cy_px - ys_px) * self.resolution_m
            outside_circle = (x_m ** 2 + y_m ** 2) > self.radius ** 2
            obs_float = self.obstacle_map.astype(np.float32)
            obs_float[outside_circle] = np.nan
            ax.imshow(obs_float, cmap='Greys', vmin=0.0, vmax=1.0,
                      alpha=0.85, extent=ext, origin='upper', zorder=1)

        # Boundary circle
        ax.add_patch(mpatches.Circle((0, 0), self.radius,
                                     fill=False, color='steelblue', lw=2))

        # Fire points
        unv = self.fire_points[~self.visited]
        vis = self.fire_points[self.visited]
        if len(unv): ax.scatter(unv[:, 0], unv[:, 1], c='red',      s=40, zorder=3, label='Unvisited')
        if len(vis): ax.scatter(vis[:, 0], vis[:, 1], c='limegreen', s=40, zorder=3, label='Visited')

        # Bird flock
        if self.num_birds > 0 and len(self._birds_pos):
            ax.scatter(self._birds_pos[:, 0], self._birds_pos[:, 1],
                       c='red', s=60, marker='^', zorder=5, label='Birds')

        # Trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=0.5, alpha=0.5)

        # UAV
        ax.scatter(*self.pos, c='blue', s=120, marker='^', zorder=5)
        aln = self.radius * 0.06
        ax.annotate('', xy=(self.pos[0] + aln * np.cos(self.heading),
                             self.pos[1] + aln * np.sin(self.heading)),
                    xytext=self.pos,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

        lim = self.radius * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'UAV Fire+Obstacle  step={self.step_count}  '
                     f'visited={np.sum(self.visited)}/{self.n_fire}')
        self._fig.canvas.draw()
        plt.pause(0.001)
