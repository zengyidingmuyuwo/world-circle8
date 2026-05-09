import os
import sys
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UAV_DIR = os.path.join(ROOT, 'UAV_Fire_Coverage')
if UAV_DIR not in sys.path:
  sys.path.insert(0, UAV_DIR)

from global_planner import DronePlanner
from uav_fire_obstacle_env import UAVFireObstacleEnv


def test_global_planner_returns_waypoints():
  fire_points = np.array([[300.0, 100.0], [500.0, -200.0], [-200.0, -350.0]], dtype=np.float32)
  obstacle_map = np.zeros((128, 128), dtype=bool)
  planner = DronePlanner(obstacle_map=obstacle_map, resolution_m=50.0)
  result = planner.plan(np.array([0.0, 0.0], dtype=np.float32), fire_points)
  assert result.cost_matrix.shape == (4, 4)
  assert len(result.tsp_order) == 4
  assert len(result.waypoints) > 0


def test_obstacle_env_dict_obs_contains_vector():
  fire_points = np.array([[100.0, 100.0], [200.0, 0.0]], dtype=np.float32)
  env = UAVFireObstacleEnv(
      fire_points=fire_points,
      radius=2000.0,
      obstacle_map=np.zeros((64, 64), dtype=bool),
      resolution_m=50.0,
      return_dict_obs=True,
  )
  out = env.reset(seed=0)
  obs = out[0] if isinstance(out, tuple) else out
  assert isinstance(obs, dict)
  assert 'image' in obs and 'vector' in obs
  assert obs['vector'].shape == (4,)
  step_out = env.step(np.array([0.0], dtype=np.float32))
  info = step_out[-1]
  assert 'off_path' in info


def test_obstacle_env_uses_circle8_naming_and_tracks_wind():
  out_dir = os.path.abspath('trajectory_results')
  os.makedirs(out_dir, exist_ok=True)
  existing = set(os.listdir(out_dir))
  fire_points = np.array([[0.0, 0.0]], dtype=np.float32)
  env = UAVFireObstacleEnv(
      fire_points=fire_points,
      radius=2000.0,
      obstacle_map=np.zeros((64, 64), dtype=bool),
      resolution_m=50.0,
      algorithm_name='SAC',
      env_name='Circle8',
  )
  env.reset(seed=0)
  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env.heading = 0.0
  old_normal = np.random.normal
  np.random.normal = lambda loc=0.0, scale=1.0, size=None: (
      np.zeros(size, dtype=np.float32) if size is not None else 0.0
  )
  try:
    env.step(np.array([0.0], dtype=np.float32))
  finally:
    np.random.normal = old_normal
  assert len(env._wind_history) == 1
  created = [name for name in os.listdir(out_dir) if name not in existing]
  best_files = [name for name in created if name.startswith('best_SAC_Circle8_PID') and name.endswith('.png')]
  assert best_files
  for name in created:
    os.remove(os.path.join(out_dir, name))


def test_obstacle_env_reset_starts_at_center():
  fire_points = np.array([[100.0, 100.0], [200.0, 0.0]], dtype=np.float32)
  env = UAVFireObstacleEnv(
      fire_points=fire_points,
      radius=2000.0,
      obstacle_map=np.zeros((64, 64), dtype=bool),
      resolution_m=50.0,
      return_dict_obs=True,
      env_name='Circle8',
  )
  out = env.reset(seed=123)
  obs = out[0] if isinstance(out, tuple) else out
  assert isinstance(obs, dict)
  np.testing.assert_allclose(env.pos, np.array([0.0, 0.0], dtype=np.float32), atol=1e-8)