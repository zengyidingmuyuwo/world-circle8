import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_circle_data, load_elevation_obstacle_map,
    generate_sample_circle8_data, find_default_elevation_source,
)
from uav_fire_obstacle_env import UAVFireObstacleEnv


def rollout(env, max_steps):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    traj = [env.pos.copy()]
    for _ in range(max_steps):
        vec = obs['vector'] if isinstance(obs, dict) else obs[-2:]
        target_angle = np.arctan2(vec[1], vec[0] + 1e-8)
        delta = (target_angle - env.heading + np.pi) % (2 * np.pi) - np.pi
        action = np.array([np.clip(delta / max(env.MAX_TURN_RATE, 1e-6), -1.0, 1.0)], dtype=np.float32)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, _, done, _ = step_out
        traj.append(env.pos.copy())
        if done:
            break
    return np.asarray(traj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Render planned and executed UAV trajectories.')
    parser.add_argument('--center_csv', type=str, default='')
    parser.add_argument('--points_file', type=str, default='')
    parser.add_argument('--elevation_tif', type=str, default='')
    parser.add_argument('--circle_id', type=int, default=None)
    parser.add_argument('--elev_threshold', type=float, default=2000.0)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--save_path', type=str, default='trajectory_plot.png')
    args = parser.parse_args()

    if args.center_csv and args.points_file and os.path.exists(args.center_csv) and os.path.exists(args.points_file):
        lat_c, lon_c, radius, fire_points = load_circle_data(args.center_csv, args.points_file, circle_id=args.circle_id)
        resolution_m = 50.0
        tif_path = args.elevation_tif if (args.elevation_tif and os.path.exists(args.elevation_tif)) else ''
        if not tif_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tif_path = find_default_elevation_source(os.path.join(os.path.dirname(script_dir), 'prepare'))
        if tif_path:
            obstacle_map, resolution_m = load_elevation_obstacle_map(
                tif_path, lat_c, lon_c, region_radius_m=radius,
                elevation_threshold=args.elev_threshold)
        else:
            obstacle_map = None
    else:
        (_, _, radius), fire_points, obstacle_map, resolution_m = generate_sample_circle8_data()

    env = UAVFireObstacleEnv(
        fire_points=fire_points,
        radius=radius,
        obstacle_map=obstacle_map,
        resolution_m=resolution_m,
        return_dict_obs=True,
    )
    env.reset()
    planned = env._global_plan_path.copy() if hasattr(env, '_global_plan_path') else np.zeros((0, 2), dtype=np.float32)
    executed = rollout(env, args.max_steps)

    fig, ax = plt.subplots(figsize=(8, 8))
    if obstacle_map is not None:
        h, w = obstacle_map.shape
        half_w = 0.5 * w * resolution_m
        half_h = 0.5 * h * resolution_m
        ext = [-half_w, half_w, -half_h, half_h]
        obstacle_layer = np.where(obstacle_map, 1.0, np.nan)
        ax.imshow(
            obstacle_layer,
            cmap=ListedColormap(['black']),
            alpha=1.0,
            extent=ext,
            origin='upper',
            interpolation='nearest',
            zorder=0,
        )
    if len(planned) > 1:
        ax.plot(planned[:, 0], planned[:, 1], '--', lw=2.0, color='tab:purple', label='A* global path')
    if len(executed) > 1:
        ax.plot(executed[:, 0], executed[:, 1], '-', lw=2.0, color='tab:blue', label='RL trajectory')
    ax.scatter(fire_points[:, 0], fire_points[:, 1], s=50, c='red', marker='*', label='Fire points')
    ax.set_aspect('equal')
    ax.set_title('UAV Fire Navigation: Global A* Guide + RL Local Execution')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(args.save_path, dpi=180)
    print(f'Saved: {args.save_path}')


if __name__ == '__main__':
    main()
