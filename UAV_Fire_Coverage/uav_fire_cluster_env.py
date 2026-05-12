"""
Clustered UAV fire-coverage environment (Circle 1).

This environment clusters all fire points into K=3 groups (K-Means) and
lets a single UAV operate on one cluster per episode (local observation).
It inherits the wind and bird dynamics from :class:`UAVFireEnv` while
adding cluster selection logic in reset/step plus bird-aware rendering.
"""

import os
import numpy as np

from uav_fire_env import UAVFireEnv, _GYM_TUPLE_5


class UAVFireClusterEnv(UAVFireEnv):
    """Single-UAV environment that auto-clusters fire points into K groups."""

    DEFAULT_NUM_CLUSTERS = 3

    def __init__(
        self,
        fire_points,
        radius,
        num_clusters=DEFAULT_NUM_CLUSTERS,
        cluster_strategy='round_robin',
        cluster_index=None,
        cluster_seed=0,
        num_nearest=6,
        return_dict_obs=False,
        algorithm_name='RL',
        env_name=None,
        radar_range_m=None,
        num_birds=3,
    ):
        self._all_fire_points = np.asarray(fire_points, dtype=np.float32)
        self.num_clusters = int(max(1, num_clusters))
        self.cluster_strategy = str(cluster_strategy).lower()
        self.cluster_index = None if cluster_index is None else int(cluster_index)
        self.cluster_seed = int(cluster_seed)
        self._clusters = []
        self._active_cluster_idx = 0
        self._cluster_cycle = 0

        env_name = env_name or 'Circle1Cluster'
        super().__init__(
            fire_points=self._all_fire_points,
            radius=radius,
            num_nearest=num_nearest,
            return_dict_obs=return_dict_obs,
            algorithm_name=algorithm_name,
            env_name=env_name,
            radar_range_m=radar_range_m,
            num_birds=num_birds,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        options = options or {}
        self._clusters = self._cluster_fire_points(self._all_fire_points)
        self._active_cluster_idx = self._select_cluster_index(options)
        self.fire_points = self._clusters[self._active_cluster_idx]
        self.n_fire = len(self.fire_points)
        result = super().reset(seed=seed, options=options)
        if _GYM_TUPLE_5:
            obs, info = result
            info = dict(info)
            info.update(self._cluster_info())
            return obs, info
        return result

    def step(self, action):
        result = super().step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            info = dict(info)
            info.update(self._cluster_info())
            return obs, reward, terminated, truncated, info
        obs, reward, done, info = result
        info = dict(info)
        info.update(self._cluster_info())
        return obs, reward, done, info

    # ─────────────────────────────────────────────────────────────────────────

    def _cluster_fire_points(self, fire_points):
        points = np.asarray(fire_points, dtype=np.float32)
        if len(points) == 0:
            return [points]
        k = int(min(self.num_clusters, len(points)))
        if k <= 1:
            return [points]
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=self.cluster_seed, n_init='auto')
            labels = km.fit_predict(points)
        except Exception:
            labels = np.arange(len(points)) % k
        clusters = [points[labels == idx] for idx in range(k)]
        clusters = [c for c in clusters if len(c) > 0]
        return clusters if clusters else [points]

    def _select_cluster_index(self, options):
        if self.cluster_index is not None:
            return int(self.cluster_index) % len(self._clusters)
        if 'cluster_id' in options:
            return int(options['cluster_id']) % len(self._clusters)
        if self.cluster_strategy == 'random':
            return int(np.random.randint(len(self._clusters)))
        idx = self._cluster_cycle % len(self._clusters)
        self._cluster_cycle += 1
        return idx

    def _cluster_info(self):
        return {
            'cluster_id': int(self._active_cluster_idx),
            'cluster_size': int(len(self.fire_points)),
        }

    # ─────────────────────────────────────────────────────────────────────────

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
        ax.add_patch(mpatches.Circle((0, 0), self.radius, fill=False, color='steelblue', lw=2))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
        group = self._TRAJECTORY_REGISTRY.get(self._registry_key(), {})
        if self.env_name.lower() != 'circle1':
            unv = self.fire_points[~self.visited]
            vis = self.fire_points[self.visited]
            if len(unv):
                ax.scatter(unv[:, 0], unv[:, 1], c='red', s=30, zorder=3, label='Unvisited')
            if len(vis):
                ax.scatter(vis[:, 0], vis[:, 1], c='limegreen', s=30, zorder=3, label='Visited')
        else:
            ids = sorted(group.keys())
            for idx, env_id in enumerate(ids):
                item = group[env_id]
                fp = item.get('fire_points', np.zeros((0, 2), dtype=np.float32))
                vm = item.get('visited_mask', np.zeros((0,), dtype=bool))
                if len(fp) == 0:
                    continue
                color = colors[idx % len(colors)]
                unv = fp[~vm]
                vis = fp[vm]
                if len(unv):
                    ax.scatter(unv[:, 0], unv[:, 1], c='lightcoral', s=20, alpha=0.35, zorder=2)
                if len(vis):
                    ax.scatter(vis[:, 0], vis[:, 1], c=color, s=28, marker='o', zorder=4,
                               label=f'UAV{idx + 1} Visited')

        if self.env_name.lower() != 'circle1' and len(self._trajectory) > 1:
            traj = np.array(self._trajectory, dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.0, alpha=0.8, label='Trajectory')
        if self.env_name.lower() == 'circle1':
            ids = sorted(group.keys())
            for idx, env_id in enumerate(ids):
                tr = np.asarray(group[env_id].get('trajectory', []), dtype=np.float32)
                if len(tr) <= 1:
                    continue
                color = colors[idx % len(colors)]
                ax.plot(tr[:, 0], tr[:, 1], '-', lw=1.5, alpha=0.85, color=color,
                        label=f'UAV{idx + 1} Trajectory')

        # Bird flock positions
        if self.env_name.lower() != 'circle1':
            if self.num_birds > 0 and len(self._birds_pos):
                ax.scatter(self._birds_pos[:, 0], self._birds_pos[:, 1],
                           c='darkorange', s=80, marker='*', zorder=4, label='Birds')
        else:
            ids = sorted(group.keys())
            for env_id in ids:
                birds = np.asarray(group[env_id].get('bird_trail_last', []), dtype=np.float32)
                if len(birds):
                    ax.scatter(birds[:, 0], birds[:, 1],
                               c='darkorange', s=70, marker='*', zorder=3, label='Birds')

        lim = self.radius * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(
            f'{title_prefix} | score={self._current_ep_score:.2f} | '
            f'coverage={coverage_rate * 100:.1f}% | cluster={self._active_cluster_idx}'
        )
        abs_path = os.path.abspath(save_path)
        plt.savefig(abs_path, dpi=300)
        plt.close(fig)
        print(f'[Trajectory] Saved snapshot: {abs_path}')

    # ─────────────────────────────────────────────────────────────────────────

    def render(self, mode='human'):
        """Matplotlib visualisation with bird flock markers."""
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

        # Boundary circle
        ax.add_patch(mpatches.Circle((0, 0), self.radius,
                                     fill=False, color='steelblue', lw=2))

        # Fire points
        unv = self.fire_points[~self.visited]
        vis = self.fire_points[self.visited]
        if len(unv):
            ax.scatter(unv[:, 0], unv[:, 1], c='red', s=40, zorder=3, label='Unvisited')
        if len(vis):
            ax.scatter(vis[:, 0], vis[:, 1], c='limegreen', s=40, zorder=3, label='Visited')

        # Bird flock
        if self.num_birds > 0 and len(self._birds_pos):
            ax.scatter(self._birds_pos[:, 0], self._birds_pos[:, 1],
                       c='darkorange', s=80, marker='*', zorder=4, label='Birds')

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
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(
            f'UAV Fire Cluster step={self.step_count}  '
            f'cluster={self._active_cluster_idx}  '
            f'visited={np.sum(self.visited)}/{self.n_fire}'
        )
        self._fig.canvas.draw()
        plt.pause(0.001)
