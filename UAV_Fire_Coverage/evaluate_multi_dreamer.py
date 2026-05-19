import argparse
import os
import sys
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    raise RuntimeError('matplotlib is required for evaluate_multi_dreamer.py') from e

import elements
import ruamel.yaml as yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import load_circle_data, generate_sample_circle1_data
from dreamerv3 import main as dreamer_main

NUM_CLUSTERS = 3


def angle_slice_clusters(points: np.ndarray) -> List[np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(NUM_CLUSTERS)]
    angles = np.arctan2(points[:, 1], points[:, 0])
    c0 = points[(angles >= -np.pi) & (angles < -np.pi / 3.0)]
    c1 = points[(angles >= -np.pi / 3.0) & (angles < np.pi / 3.0)]
    c2 = points[(angles >= np.pi / 3.0) & (angles <= np.pi)]
    return [c0.astype(np.float32), c1.astype(np.float32), c2.astype(np.float32)]


def load_dreamer_config(task: str, logdir: str) -> elements.Config:
    cfg_path = os.path.join(ROOT, 'dreamerv3', 'configs.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        configs = yaml.YAML(typ='safe').load(f.read())
    config = elements.Config(configs['defaults']).update(configs['uavfire'])
    config = config.update(task=task, logdir=logdir)
    return config


def add_batch_dim(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        out[key] = np.expand_dims(arr, 0)
    return out


def unwrap_uav_env(env):
    cur = env
    for _ in range(20):
        if hasattr(cur, '_env'):
            cur = cur._env
        else:
            break
    return cur


def rollout_one_cluster(agent, config, cluster_id: int, max_steps: int):
    env = dreamer_main.make_env(
        config,
        0,
        num_clusters=NUM_CLUSTERS,
        cluster_index=cluster_id,
        cluster_strategy='fixed',
    )
    carry = agent.init_policy(1)
    zero_action = np.zeros(env.act_space['action'].shape, dtype=np.float32)
    obs = env.step({'reset': True, 'action': zero_action})
    score = 0.0
    for _ in range(max_steps):
        bobs = add_batch_dim(obs)
        carry, acts, _ = agent.policy(carry, bobs, mode='eval')
        action = np.asarray(acts['action'][0], dtype=np.float32)
        obs = env.step({'reset': False, 'action': action})
        score += float(obs['reward'])
        if bool(obs['is_last']) or bool(obs['is_terminal']):
            break

    raw = unwrap_uav_env(env)
    result = {
        'cluster_id': cluster_id,
        'score': float(score),
        'trajectory': np.asarray(raw._trajectory, dtype=np.float32),
        'birds_pos': np.asarray(raw._birds_pos, dtype=np.float32),
        'birds_trails': [np.asarray(tr, dtype=np.float32) for tr in raw._bird_trails],
        'fire_points': np.asarray(raw.fire_points, dtype=np.float32),
        'visited': np.asarray(raw.visited, dtype=bool),
        'radius': float(raw.radius),
    }
    try:
        env.close()
    except Exception:
        pass
    return result


def plot_combined(results, all_fire_points, radius, save_path):
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(mpatches.Circle((0, 0), radius, fill=False, color='steelblue', lw=2))

    ax.scatter(all_fire_points[:, 0], all_fire_points[:, 1], c='lightcoral', s=20, alpha=0.35, label='All fire points')

    total_score = 0.0
    total_visited = 0
    for idx, item in enumerate(results):
        color = colors[idx % len(colors)]
        traj = item['trajectory']
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], '-', lw=1.8, color=color, alpha=0.9, label=f'UAV{idx + 1} Trajectory')
        fp = item['fire_points']
        vm = item['visited']
        if len(fp):
            vis = fp[vm]
            if len(vis):
                ax.scatter(vis[:, 0], vis[:, 1], c=color, s=26, marker='o', zorder=4, label=f'UAV{idx + 1} Visited')
        for tr in item['birds_trails']:
            if len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], '--', lw=0.8, color='darkorange', alpha=0.35)
        if len(item['birds_pos']):
            ax.scatter(
                item['birds_pos'][:, 0],
                item['birds_pos'][:, 1],
                c='red',
                s=60,
                marker='^',
                zorder=5,
                label='Birds',
            )
        total_score += float(item['score'])
        total_visited += int(np.sum(vm))

    coverage = 100.0 * (total_visited / max(len(all_fire_points), 1))
    ax.set_title(f'Best Episode Trajectory | score={total_score:.2f} | coverage={coverage:.1f}%')
    lim = radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    ax.legend(dedup.values(), dedup.keys(), loc='upper right', fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=240)
    plt.close(fig)
    print(f'[evaluate_multi_dreamer] Saved: {os.path.abspath(save_path)}')
    print(f'[evaluate_multi_dreamer] score={total_score:.2f} coverage={coverage:.1f}%')


def resolve_checkpoint_paths(args):
    if args.checkpoint:
        return [args.checkpoint for _ in range(NUM_CLUSTERS)]
    if not (args.checkpoint0 and args.checkpoint1 and args.checkpoint2):
        raise ValueError('Provide either --checkpoint or all of --checkpoint0 --checkpoint1 --checkpoint2')
    return [args.checkpoint0, args.checkpoint1, args.checkpoint2]


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3 Dreamer experts on circle1 angle-sliced clusters.')
    parser.add_argument('--checkpoint', type=str, default='', help='One shared Dreamer checkpoint path for all 3 clusters.')
    parser.add_argument('--checkpoint0', type=str, default='', help='Dreamer checkpoint path for cluster 0.')
    parser.add_argument('--checkpoint1', type=str, default='', help='Dreamer checkpoint path for cluster 1.')
    parser.add_argument('--checkpoint2', type=str, default='', help='Dreamer checkpoint path for cluster 2.')
    parser.add_argument('--center_csv', type=str, default='')
    parser.add_argument('--points_file', type=str, default='')
    parser.add_argument('--circle_id', type=int, default=-1)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--save_path', type=str, default='trajectory_results/multi_dreamer_circle1.png')
    parser.add_argument('--logdir', type=str, default='~/logdir/uavfire_eval_multi')
    args = parser.parse_args()

    if args.center_csv and args.points_file and os.path.exists(args.center_csv) and os.path.exists(args.points_file):
        kwargs = {} if args.circle_id == -1 else {'circle_id': int(args.circle_id)}
        _, _, radius, all_fire_points = load_circle_data(args.center_csv, args.points_file, **kwargs)
    else:
        (_, _, radius), all_fire_points = generate_sample_circle1_data()
    clusters = angle_slice_clusters(all_fire_points)
    print('[evaluate_multi_dreamer] Angle-sliced clusters:', [len(c) for c in clusters])

    ckpts = resolve_checkpoint_paths(args)
    config = load_dreamer_config(task='uavfire_circle1_cluster', logdir=os.path.expanduser(args.logdir))
    agent = dreamer_main.make_agent(config)
    cp = elements.Checkpoint()
    cp.agent = agent

    results = []
    for cluster_id, ckpt in enumerate(ckpts):
        if not ckpt:
            raise ValueError(f'Empty checkpoint for cluster {cluster_id}')
        cp.load(ckpt, keys=['agent'])
        run = rollout_one_cluster(agent, config, cluster_id=cluster_id, max_steps=args.max_steps)
        results.append(run)
        print(f'[evaluate_multi_dreamer] cluster={cluster_id} score={run["score"]:.2f} visited={int(np.sum(run["visited"]))}/{len(run["visited"])}')

    plot_combined(results, all_fire_points, radius, args.save_path)


if __name__ == '__main__':
    main()
