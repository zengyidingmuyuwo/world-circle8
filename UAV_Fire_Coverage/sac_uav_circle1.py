"""
SAC training for Circle 1 — three-UAV fire-point coverage.

Uses the dual-Q (SAC v2) formulation consistent with
``Char09 SAC/SAC_BipedalWalker-v2.py`` in this repository.

Strategy
--------
Fire points in Circle 1 are divided into 3 sub-clusters (angle slices).
A single SAC policy is trained by cycling through all three sub-cluster
environments.  The shared replay buffer enables cross-cluster experience.

Usage
-----
# Train with sample data (no real data files needed):
    python sac_uav_circle1.py

# Train with your own data files:
    python sac_uav_circle1.py \
        --center_csv  "/path/to/prepare/circle_1_center.csv" \
        --points_file "/path/to/prepare/circle_1_points.shp"

# Resume (load saved model):
    python sac_uav_circle1.py --load
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_fire_cluster_env import UAVFireClusterEnv
from data_utils import (load_circle_data, generate_sample_circle1_data)
from comparison_logging import EpisodeCSVLogger


# ── gym / gymnasium compatibility helpers ─────────────────────────────────────

def env_reset(env):
    result = env.reset()
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return np.concatenate([result['image'], result['vector']]).astype(np.float32)
    return result


def env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, rew, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, rew, done, info = result
    if isinstance(obs, dict):
        obs = np.concatenate([obs['image'], obs['vector']]).astype(np.float32)
    return obs, rew, done, info

# ── argument parser ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PREPARE_DIR = os.path.join(BASE_DIR, 'prepare')

parser = argparse.ArgumentParser(description='SAC — Circle 1 multi-UAV fire coverage')
parser.add_argument('--center_csv',
    default=os.path.join(PREPARE_DIR, 'circle_1_center.csv'),
    type=str, help='Circle-1 centre CSV (columns: circle_id, center_x, center_y, radius_m, diameter_m)')
parser.add_argument('--circle_id',
    default=None, type=int,
    help='circle_id value to select from the centre CSV (default: first row)')
parser.add_argument('--points_file',
    default=os.path.join(PREPARE_DIR, 'circle_1_points.shp'),
    type=str, help='Circle-1 fire-point SHP or CSV file')
parser.add_argument('--num_uavs',     default=3,  type=int)
parser.add_argument('--gamma',        default=0.99, type=float)
parser.add_argument('--tau',          default=0.005, type=float)
parser.add_argument('--lr',           default=3e-4,  type=float)
parser.add_argument('--capacity',     default=100000, type=int)
parser.add_argument('--batch_size',   default=256,   type=int)
parser.add_argument('--warmup_steps', default=1000,  type=int)
parser.add_argument('--max_episodes', default=2000,  type=int)
parser.add_argument('--log_interval', default=20,    type=int)
parser.add_argument('--save_interval',default=200,   type=int)
parser.add_argument('--render',       action='store_true')
parser.add_argument('--load',         action='store_true')
parser.add_argument('--save_dir',     default='./sac_circle1_model', type=str)
parser.add_argument('--log_dir',      default=os.path.join(SCRIPT_DIR, 'logs'), type=str,
                    help='Directory for unified comparison CSV logs')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_LOG_STD = -20
MAX_LOG_STD = 2
LOG_MIN_VAL = torch.tensor(1e-7).to(device)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity  = capacity
        self.ptr       = 0
        self.size      = 0
        self.s  = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.a  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, 1),          dtype=np.float32)
        self.s_ = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.d  = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, s, a, r, s_, d):
        i = self.ptr % self.capacity
        self.s[i]  = s
        self.a[i]  = a
        self.r[i]  = r
        self.s_[i] = s_
        self.d[i]  = float(d)
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx  = np.random.choice(self.size, batch_size, replace=False)
        return (torch.FloatTensor(self.s[idx]).to(device),
                torch.FloatTensor(self.a[idx]).to(device),
                torch.FloatTensor(self.r[idx]).to(device),
                torch.FloatTensor(self.s_[idx]).to(device),
                torch.FloatTensor(self.d[idx]).to(device))

    def ready(self, batch_size):
        return self.size >= batch_size


# ── Networks ──────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        x       = self.net(x)
        mu      = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mu, std)
        z    = dist.rsample()
        action   = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + LOG_MIN_VAL)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """V-function network."""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    """Q-function network."""
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ── SAC agent ─────────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.policy       = Actor(state_dim, action_dim).to(device)
        self.value        = Critic(state_dim).to(device)
        self.value_target = Critic(state_dim).to(device)
        self.q1           = QNet(state_dim, action_dim).to(device)
        self.q2           = QNet(state_dim, action_dim).to(device)

        self.value_target.load_state_dict(self.value.state_dict())

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_value  = optim.Adam(self.value.parameters(),  lr=args.lr)
        self.opt_q1     = optim.Adam(self.q1.parameters(),     lr=args.lr)
        self.opt_q2     = optim.Adam(self.q2.parameters(),     lr=args.lr)

        self.replay     = ReplayBuffer(args.capacity, state_dim, action_dim)
        self.num_updates = 0

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        if deterministic:
            mu, _ = self.policy(s)
            return torch.tanh(mu).detach().cpu().numpy().flatten()
        action, _ = self.policy.sample(s)
        return action.detach().cpu().numpy().flatten()

    def update(self):
        if not self.replay.ready(args.batch_size):
            return

        s, a, r, s_, d = self.replay.sample(args.batch_size)

        with torch.no_grad():
            target_v    = self.value_target(s_)
            next_q_val  = r + (1 - d) * args.gamma * target_v

        # ── Critic Q1 & Q2 ────────────────────────────────────────────────────
        q1_loss = F.mse_loss(self.q1(s, a), next_q_val)
        self.opt_q1.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 0.5)
        self.opt_q1.step()

        q2_loss = F.mse_loss(self.q2(s, a), next_q_val)
        self.opt_q2.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 0.5)
        self.opt_q2.step()

        # ── Critic V ──────────────────────────────────────────────────────────
        with torch.no_grad():
            a_v, lp_v = self.policy.sample(s)
            q_min_v   = torch.min(self.q1(s, a_v), self.q2(s, a_v))
        v_loss = F.mse_loss(self.value(s), (q_min_v - lp_v))
        self.opt_value.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.opt_value.step()

        # ── Actor ─────────────────────────────────────────────────────────────
        a_pi, lp_pi = self.policy.sample(s)
        q_min_pi = torch.min(self.q1(s, a_pi), self.q2(s, a_pi))
        pi_loss  = (lp_pi - q_min_pi).mean()
        self.opt_policy.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.opt_policy.step()

        # ── Soft update of target V ───────────────────────────────────────────
        for tp, p in zip(self.value_target.parameters(), self.value.parameters()):
            tp.data.copy_((1 - args.tau) * tp.data + args.tau * p.data)

        self.num_updates += 1

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(directory, 'sac_policy.pth'))
        torch.save(self.value.state_dict(),  os.path.join(directory, 'sac_value.pth'))
        torch.save(self.q1.state_dict(),     os.path.join(directory, 'sac_q1.pth'))
        torch.save(self.q2.state_dict(),     os.path.join(directory, 'sac_q2.pth'))
        print(f'[SAC] Model saved → {directory}')

    def load(self, directory):
        self.policy.load_state_dict(
            torch.load(os.path.join(directory, 'sac_policy.pth'), map_location=device))
        self.value.load_state_dict(
            torch.load(os.path.join(directory, 'sac_value.pth'),  map_location=device))
        self.q1.load_state_dict(
            torch.load(os.path.join(directory, 'sac_q1.pth'),     map_location=device))
        self.q2.load_state_dict(
            torch.load(os.path.join(directory, 'sac_q2.pth'),     map_location=device))
        self.value_target.load_state_dict(self.value.state_dict())
        print(f'[SAC] Model loaded ← {directory}')


# ── Angle-sliced cluster helper ───────────────────────────────────────────────

def cluster_fire_points(fire_points, n_clusters=3):
    points = np.asarray(fire_points, dtype=np.float32)
    if len(points) == 0:
        empty = points.reshape(0, 2).astype(np.float32)
        return [empty, empty.copy(), empty.copy()]
    angles = np.arctan2(points[:, 1], points[:, 0])
    c0 = points[(angles >= -np.pi) & (angles < -np.pi / 3.0)]
    c1 = points[(angles >= -np.pi / 3.0) & (angles < np.pi / 3.0)]
    c2 = points[(angles >= np.pi / 3.0) & (angles <= np.pi)]
    return [c0.astype(np.float32), c1.astype(np.float32), c2.astype(np.float32)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Data ─────────────────────────────────────────────────────────────────
    if args.center_csv and os.path.exists(args.center_csv) and \
            args.points_file and os.path.exists(args.points_file):
        print(f'[SAC Circle1] Loading data from {args.points_file} …')
        lat_c, lon_c, radius, fire_points = load_circle_data(
            args.center_csv, args.points_file, circle_id=args.circle_id)
        print(f'  Centre: ({lat_c:.4f}°N, {lon_c:.4f}°E)  radius={radius:.0f} m  '
              f'fire points: {len(fire_points)}')
    else:
        print('[SAC Circle1] Using synthetic sample data …')
        (lat_c, lon_c, radius), fire_points = generate_sample_circle1_data()
        print(f'  Sample: radius={radius:.0f} m  fire_points={len(fire_points)}')

    clusters = cluster_fire_points(fire_points, args.num_uavs)
    print(f'[SAC Circle1] {len(clusters)} clusters:  '
          + '  '.join(f'UAV{i+1}={len(c)}pts' for i, c in enumerate(clusters)))

    envs = [
        UAVFireClusterEnv(
            fire_points=fire_points,
            radius=radius,
            num_clusters=len(clusters),
            cluster_index=idx,
            cluster_strategy='fixed',
            return_dict_obs=False,
            algorithm_name='SAC',
            env_name='Circle1',
        )
        for idx in range(len(clusters))
    ]
    state_dim  = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    print(f'[SAC Circle1] state_dim={state_dim}  action_dim={action_dim}')

    agent = SACAgent(state_dim, action_dim)
    if args.load:
        agent.load(args.save_dir)
    episode_logger = EpisodeCSVLogger('SAC', 'Circle1', args.log_dir)
    print(f'[SAC Circle1] Writing training log to: {os.path.abspath(episode_logger.path)}')

    running_rewards = [0.0] * len(envs)
    total_steps     = 0

    for episode in range(1, args.max_episodes + 1):
        env_idx   = (episode - 1) % len(envs)
        env       = envs[env_idx]
        state     = env_reset(env)
        ep_reward = 0.0

        for t in range(env.MAX_STEPS):
            if total_steps < args.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env_step(env, action)
            ep_reward += reward

            if args.render:
                env.render()

            agent.replay.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_steps += 1
            if done:
                break

        running_rewards[env_idx] = 0.95 * running_rewards[env_idx] + 0.05 * ep_reward

        if episode % args.log_interval == 0:
            cov = info['coverage_rate'] * 100
            print(f'Episode {episode:5d}  UAV{env_idx+1}  '
                  f'steps={t+1:4d}  ep_r={ep_reward:7.1f}  '
                  f'running_r={running_rewards[env_idx]:7.1f}  '
                  f'coverage={cov:.1f}%  updates={agent.num_updates}')
        episode_logger.log_episode(
            episode=episode,
            timesteps=total_steps,
            episode_reward=ep_reward,
            coverage_pct=info.get('coverage_rate', 0.0) * 100.0,
            collision=bool(info.get('collision', False)),
        )

        if episode % args.save_interval == 0:
            agent.save(args.save_dir)

    agent.save(args.save_dir)
    print('[SAC Circle1] Training complete.')

    print('\n[SAC Circle1] Evaluating all UAVs …')
    for i, env in enumerate(envs):
        state   = env_reset(env)
        total_r = 0.0
        for _ in range(env.MAX_STEPS):
            action  = agent.select_action(state, deterministic=True)
            state, reward, done, info = env_step(env, action)
            total_r += reward
            if done:
                break
        print(f'  UAV{i+1}: coverage={info["coverage_rate"]*100:.1f}%  '
              f'visited={info["visited_count"]}/{info["total_fire_points"]}  '
              f'total_reward={total_r:.1f}')


if __name__ == '__main__':
    main()
