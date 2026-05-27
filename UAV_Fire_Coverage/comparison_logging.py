import csv
import os
import time
from typing import Optional


class EpisodeCSVLogger:
    """Unified episode logger for cross-algorithm comparison."""

    def __init__(
        self,
        algorithm: str,
        scenario: str,
        output_dir: str,
    ):
        self.algorithm = str(algorithm).upper()
        self.scenario = str(scenario)
        self._start_time = time.time()
        self._path = self._make_path(output_dir)
        self._init_file()

    def _make_path(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'standard_comparison_log_{self.algorithm}_{self.scenario}.csv'
        return os.path.abspath(os.path.join(output_dir, filename))

    def _init_file(self) -> None:
        headers = ['Episode', 'Steps', 'Wall_time', 'Reward', 'Coverage_Rate']
        if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            return
        with open(self._path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    @property
    def path(self) -> str:
        return self._path

    def log_episode(
        self,
        episode: int,
        timesteps: int,
        episode_reward: float,
        coverage_pct: float,
        collision: bool = False,
        wall_time_sec: Optional[float] = None,
    ) -> None:
        if wall_time_sec is None:
            wall_time_sec = time.time() - self._start_time
        row = {
            'Episode': int(episode),
            'Steps': int(timesteps),
            'Wall_time': float(wall_time_sec),
            'Reward': float(episode_reward),
            'Coverage_Rate': float(coverage_pct),
        }
        with open(self._path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Episode', 'Steps', 'Wall_time', 'Reward', 'Coverage_Rate'])
            writer.writerow(row)