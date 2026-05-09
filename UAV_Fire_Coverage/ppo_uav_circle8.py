"""
PPO training for Circle 8 — single-UAV fire coverage with obstacle avoidance.

The UAV must visit all fire points while avoiding elevation obstacles
(regions above 2000 m in the supplied DEM).

Usage
-----
# Train with sample data (no real files needed):
    python ppo_uav_circle8.py

# Train with your own data files:
    python ppo_uav_circle8.py \
        --center_csv  "/path/to/prepare/circle_8_center.csv" \
        --points_file "/path/to/prepare/circle_8_points.shp" \
        --elevation_tif "/path/to/prepare/elevation/your_dem.tif"

# Resume (load saved model):
    python ppo_uav_circle8.py --load
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
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_fire_obstacle_env import UAVFireObstacleEnv
from data_utils import (load_circle_data, load_elevation_obstacle_map,
                        generate_sample_circle8_data)
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

parser = argparse.ArgumentParser(description='PPO — Circle 8 fire coverage + obstacle avoidance')
parser.add_argument('--center_csv',
    default=os.path.join(PREPARE_DIR, 'circle_8_center.csv'),
    type=str, help='Circle-8 centre CSV (columns: circle_id, center_x, center_y, radius_m, diameter_m)')
parser.add_argument('--circle_id',
    default=None, type=int,
    help='circle_id value to select from the centre CSV (default: first row)')
parser.add_argument('--points_file',
    default=os.path.join(PREPARE_DIR, 'circle_8_points.shp'),
    type=str, help='Circle-8 fire-point SHP or CSV file')
parser.add_argument('--elevation_tif',
    default='',
    type=str, help='DEM GeoTIFF; pixels ≥ elev_threshold are obstacles')
parser.add_argument('--elev_threshold', default=2000.0, type=float)
parser.add_argument('--gamma',         default=0.99, type=float)
parser.add_argument('--lr_actor',      default=3e-4, type=float)
parser.add_argument('--lr_critic',     default=1e-3, type=float)
parser.add_argument('--clip_param',    default=0.2,  type=float)
parser.add_argument('--ppo_epoch',     default=10,   type=int)
parser.add_argument('--buffer_size',   default=2048, type=int)
parser.add_argument('--batch_size',    default=64,   type=int)
parser.add_argument('--max_grad_norm', default=0.5,  type=float)
parser.add_argument('--max_episodes',  default=2000, type=int)
parser.add_argument('--log_interval',  default=20,   type=int)
parser.add_argument('--save_interval', default=200,  type=int)
parser.add_argument('--render',        action='store_true')
parser.add_argument('--load',          action='store_true')
parser.add_argument('--save_dir',      default='./ppo_circle8_model', type=str)
parser.add_argument('--log_dir',       default=os.path.join(SCRIPT_DIR, 'logs'), type=str,
                    help='Directory for unified comparison CSV logs')
args = parser.parse_args()
if not args.elevation_tif:
    elev_dir = os.path.join(PREPARE_DIR, 'elevation')
    if os.path.isdir(elev_dir):
        tif_candidates = sorted(
            f for f in os.listdir(elev_dir) if f.lower().endswith(('.tif', '.tiff', '.zip'))
        )
        if tif_candidates:
            args.elevation_tif = os.path.join(elev_dir, tif_candidates[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


# ── Neural networks (same architecture as ppo_uav_circle1.py) ─────────────────

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
        )
        self.mu_head    = nn.Linear(128, action_dim)
        self.sigma_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x   = self.net(x)
        mu  = torch.tanh(self.mu_head(x))
        std = F.softplus(self.sigma_head(x)) + 1e-5
        return mu, std


class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── PPO agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor  = ActorNet(state_dim, action_dim).to(device).float()
        self.critic = CriticNet(state_dim).to(device).float()
        self.opt_a  = optim.Adam(self.actor.parameters(),  lr=args.lr_actor)
        self.opt_c  = optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.buffer = []
        self.ptr    = 0

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std = self.actor(s)
        dist   = Normal(mu, std)
        action = dist.sample()
        log_p  = dist.log_prob(action).sum(dim=-1)
        action = action.clamp(-1.0, 1.0)
        return action.cpu().numpy().flatten(), log_p.item()

    def store(self, transition):
        self.buffer.append(transition)
        self.ptr += 1
        return self.ptr % args.buffer_size == 0

    def update(self):
        s   = torch.FloatTensor(np.stack([t.s        for t in self.buffer])).to(device)
        a   = torch.FloatTensor(np.stack([t.a        for t in self.buffer])).to(device)
        r   = torch.FloatTensor(np.array([t.r        for t in self.buffer])).unsqueeze(1).to(device)
        s_  = torch.FloatTensor(np.stack([t.s_       for t in self.buffer])).to(device)
        alp = torch.FloatTensor(np.array([t.a_log_p  for t in self.buffer])).unsqueeze(1).to(device)

        r = (r - r.mean()) / (r.std() + 1e-7)
        with torch.no_grad():
            target_v = r + args.gamma * self.critic(s_)
        adv = (target_v - self.critic(s)).detach()

        buf_len = len(self.buffer)
        for _ in range(args.ppo_epoch):
            idx = np.random.choice(buf_len, min(args.batch_size, buf_len), replace=False)
            idx = torch.LongTensor(idx).to(device)

            mu, std = self.actor(s[idx])
            dist    = Normal(mu, std)
            new_lp  = dist.log_prob(a[idx]).sum(dim=-1, keepdim=True)
            ratio   = torch.exp(new_lp - alp[idx])

            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - args.clip_param,
                                        1 + args.clip_param) * adv[idx]
            entropy     = dist.entropy().mean()
            actor_loss  = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = F.smooth_l1_loss(self.critic(s[idx]), target_v[idx])

            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.opt_a.step()

            self.opt_c.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            self.opt_c.step()

        self.buffer.clear()
        self.ptr = 0

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, 'ppo_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'ppo_critic.pth'))
        print(f'[PPO] Model saved → {directory}')

    def load(self, directory):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, 'ppo_actor.pth'), map_location=device))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, 'ppo_critic.pth'), map_location=device))
        print(f'[PPO] Model loaded ← {directory}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    obstacle_map  = None
    resolution_m  = 50.0
    dem_query_metadata = None

    # ── Load or generate data ─────────────────────────────────────────────────
    if args.center_csv and os.path.exists(args.center_csv) and \
            args.points_file and os.path.exists(args.points_file):
        print(f'[PPO Circle8] Loading data from {args.points_file} …')
        lat_c, lon_c, radius, fire_points = load_circle_data(
            args.center_csv, args.points_file, circle_id=args.circle_id)
        print(f'  Centre: ({lat_c:.4f}°N, {lon_c:.4f}°E)  radius={radius:.0f} m  '
              f'fire points: {len(fire_points)}')

        if args.elevation_tif and os.path.exists(args.elevation_tif):
            print(f'[PPO Circle8] Loading elevation map: {args.elevation_tif}')
            try:
                obstacle_map, resolution_m, dem_query_metadata = load_elevation_obstacle_map(
                    args.elevation_tif, lat_c, lon_c,
                    region_radius_m=radius,
                    elevation_threshold=args.elev_threshold,
                    return_metadata=True,
                )
                n_obs = int(np.sum(obstacle_map))
                print(f'  Obstacle map: {obstacle_map.shape}  '
                      f'{n_obs} obstacle pixels ({n_obs/obstacle_map.size*100:.1f}%)')
            except Exception as e:
                print(f'  [WARNING] Could not load elevation map: {e}')
                print('  Proceeding without obstacles.')
    else:
        print('[PPO Circle8] Using synthetic sample data …')
        (lat_c, lon_c, radius), fire_points, obstacle_map, resolution_m = \
            generate_sample_circle8_data()
        n_obs = int(np.sum(obstacle_map))
        print(f'  Sample: radius={radius:.0f} m  fire_points={len(fire_points)}  '
              f'obstacle_pixels={n_obs}')

    # ── Environment ───────────────────────────────────────────────────────────
    env = UAVFireObstacleEnv(
        fire_points=fire_points,
        radius=radius,
        obstacle_map=obstacle_map,
        resolution_m=resolution_m,
        algorithm_name='PPO',
        env_name='Circle8',
        lat_center=lat_c,
        lon_center=lon_c,
        elevation_threshold=args.elev_threshold,
        dem_query_metadata=dem_query_metadata,
    )
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'[PPO Circle8] state_dim={state_dim}  action_dim={action_dim}')

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = PPOAgent(state_dim, action_dim)
    if args.load:
        agent.load(args.save_dir)
    episode_logger = EpisodeCSVLogger('PPO', 'Circle8', args.log_dir)
    print(f'[PPO Circle8] Writing training log to: {os.path.abspath(episode_logger.path)}')

    # ── Training ──────────────────────────────────────────────────────────────
    running_reward = 0.0
    total_steps = 0

    for episode in range(1, args.max_episodes + 1):
        state    = env_reset(env)
        ep_reward = 0.0

        for t in range(env.MAX_STEPS):
            action, log_p = agent.select_action(state)
            next_state, reward, done, info = env_step(env, action)
            ep_reward += reward

            if args.render:
                env.render()

            trans = Transition(state, action, log_p, reward, next_state)
            if agent.store(trans):
                agent.update()

            state = next_state
            total_steps += 1
            if done:
                break

        running_reward = 0.95 * running_reward + 0.05 * ep_reward

        if episode % args.log_interval == 0:
            collision = info.get('collision', False)
            cov = info['coverage_rate'] * 100
            print(f'Episode {episode:5d}  steps={t+1:4d}  '
                  f'ep_r={ep_reward:7.1f}  running_r={running_reward:7.1f}  '
                  f'coverage={cov:.1f}%  collision={collision}')
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
    print('[PPO Circle8] Training complete.')


if __name__ == '__main__':
    main()