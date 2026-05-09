import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parent


def test_training_scripts_do_not_use_windows_absolute_paths():
  files = [
      ROOT / 'ppo_uav_circle1.py',
      ROOT / 'sac_uav_circle1.py',
      ROOT / 'ppo_uav_circle8.py',
      ROOT / 'sac_uav_circle8.py',
  ]
  for path in files:
    text = path.read_text(encoding='utf-8')
    assert not re.search(r'[A-Za-z]:[\\/]', text), f'Windows absolute path found in {path.name}'


def test_training_scripts_use_prepare_dir_defaults():
  files = [
      ROOT / 'ppo_uav_circle1.py',
      ROOT / 'sac_uav_circle1.py',
      ROOT / 'ppo_uav_circle8.py',
      ROOT / 'sac_uav_circle8.py',
  ]
  for path in files:
    text = path.read_text(encoding='utf-8')
    assert "PREPARE_DIR = os.path.join(BASE_DIR, 'prepare')" in text