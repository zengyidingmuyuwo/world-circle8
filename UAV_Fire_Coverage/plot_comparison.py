#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


STYLE = {
    'PPO': {'color': '#1f77b4', 'linestyle': '-'},
    'SAC': {'color': '#2ca02c', 'linestyle': '--'},
    'DREAMER': {'color': '#d62728', 'linestyle': '-.'},
}


@dataclass
class RunSeries:
    algorithm: str
    scenario: str
    run_id: str
    episode: np.ndarray
    timesteps: np.ndarray
    wall_time_sec: np.ndarray
    episode_reward: np.ndarray
    coverage_pct: np.ndarray


def normalize_scenario(name: str) -> str:
    s = str(name).strip().lower().replace(' ', '')
    if 'circle1' in s or s.endswith('1'):
        return 'Circle1'
    if 'circle8' in s or s.endswith('8'):
        return 'Circle8'
    return str(name)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 2:
        return y.copy()
    y = y.astype(np.float64)
    valid = np.isfinite(y).astype(np.float64)
    y0 = np.where(np.isfinite(y), y, 0.0)
    kernel = np.ones(int(window), dtype=np.float64)
    num = np.convolve(y0, kernel, mode='same')
    den = np.convolve(valid, kernel, mode='same')
    den = np.maximum(den, 1.0)
    return num / den


def infer_scenario_from_path(path: Path) -> str:
    low = str(path).lower()
    if re.search(r'circle[_\-/ ]?1', low):
        return 'Circle1'
    if re.search(r'circle[_\-/ ]?8', low):
        return 'Circle8'
    if 'uavfire_circle1' in low:
        return 'Circle1'
    if 'uavfire_circle8' in low:
        return 'Circle8'
    return 'Unknown'


def parse_unified_csv(path: Path) -> RunSeries:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    def _pick(row, *keys, default=None):
        for k in keys:
            if k in row and row[k] not in (None, ''):
                return row[k]
        return default

    alg = str(_pick(rows[0], 'algorithm', 'Algorithm', default='UNKNOWN')).upper()
    scenario = normalize_scenario(_pick(rows[0], 'scenario', 'Scenario', default=infer_scenario_from_path(path)))
    run_id = _pick(rows[0], 'run_id', 'Run_ID', default=path.stem)
    rows.sort(key=lambda r: float(_pick(r, 'timesteps', 'Steps', default=0.0)))
    return RunSeries(
        algorithm=alg,
        scenario=scenario,
        run_id=run_id,
        episode=np.asarray([float(_pick(r, 'episode', 'Episode', default=i + 1)) for i, r in enumerate(rows)], dtype=np.float64),
        timesteps=np.asarray([float(_pick(r, 'timesteps', 'Steps', default=i + 1)) for i, r in enumerate(rows)], dtype=np.float64),
        wall_time_sec=np.asarray([float(_pick(r, 'wall_time_sec', 'Wall_time', default=i + 1)) for i, r in enumerate(rows)], dtype=np.float64),
        episode_reward=np.asarray([float(_pick(r, 'episode_reward', 'Reward', default=np.nan)) for r in rows], dtype=np.float64),
        coverage_pct=np.asarray([float(_pick(r, 'coverage_pct', 'Coverage_Rate', default=np.nan)) for r in rows], dtype=np.float64),
    )


def parse_dreamer_scores_jsonl(path: Path) -> RunSeries:
    xs_step, xs_time, ys_reward = [], [], []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            reward = obj.get('episode/score')
            if reward is None:
                continue
            xs_step.append(float(obj.get('step', len(xs_step) + 1)))
            xs_time.append(float(obj.get('time', obj.get('wall_time_sec', len(xs_step)))))
            ys_reward.append(float(reward))
    if not ys_reward:
        raise ValueError(f'No episode/score in {path}')
    scenario = normalize_scenario(infer_scenario_from_path(path))
    rid = path.parent.name
    n = len(ys_reward)
    return RunSeries(
        algorithm='DREAMER',
        scenario=scenario,
        run_id=rid,
        episode=np.arange(1, n + 1, dtype=np.float64),
        timesteps=np.asarray(xs_step, dtype=np.float64),
        wall_time_sec=np.asarray(xs_time, dtype=np.float64),
        episode_reward=np.asarray(ys_reward, dtype=np.float64),
        coverage_pct=np.full(n, np.nan, dtype=np.float64),
    )


def discover_runs(logs_dir: Path, repo_root: Path) -> List[RunSeries]:
    runs: List[RunSeries] = []
    for path in logs_dir.rglob('*.csv'):
        if path.name == 'comparison_results.csv':
            continue
        try:
            rows = sum(1 for _ in path.open('r', encoding='utf-8'))
        except Exception:
            continue
        if rows < 2:
            continue
        try:
            run = parse_unified_csv(path)
        except Exception:
            continue
        if run.scenario in ('Circle1', 'Circle8'):
            runs.append(run)

    has_dreamer = any(r.algorithm == 'DREAMER' for r in runs)
    if not has_dreamer:
        for path in repo_root.rglob('scores.jsonl'):
            try:
                run = parse_dreamer_scores_jsonl(path)
            except Exception:
                continue
            if run.scenario in ('Circle1', 'Circle8'):
                runs.append(run)
    return runs


def to_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    smooth_window: int,
) -> np.ndarray:
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y = moving_average(y, smooth_window)
    if len(x) == 1:
        out = np.full_like(x_grid, np.nan, dtype=np.float64)
        k = np.argmin(np.abs(x_grid - x[0]))
        out[k] = y[0]
        return out
    yi = np.interp(x_grid, x, y)
    yi[(x_grid < x[0]) | (x_grid > x[-1])] = np.nan
    return yi


def plot_one(
    runs: List[RunSeries],
    scenario: str,
    x_key: str,
    y_key: str,
    smooth_window: int,
    out_path: Path,
) -> None:
    selected = [r for r in runs if r.scenario == scenario]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    plotted = False
    for alg in ('PPO', 'SAC', 'DREAMER'):
        aruns = [r for r in selected if r.algorithm == alg]
        if not aruns:
            continue
        xmins, xmaxs = [], []
        for r in aruns:
            xv = getattr(r, x_key)
            yv = getattr(r, y_key)
            mask = np.isfinite(xv) & np.isfinite(yv)
            if np.sum(mask) < 2:
                continue
            xmins.append(float(np.nanmin(xv[mask])))
            xmaxs.append(float(np.nanmax(xv[mask])))
        if not xmins:
            continue
        xmin, xmax = min(xmins), max(xmaxs)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            continue
        x_grid = np.linspace(xmin, xmax, 240)
        ys = []
        for r in aruns:
            xv = getattr(r, x_key)
            yv = getattr(r, y_key)
            mask = np.isfinite(xv) & np.isfinite(yv)
            if np.sum(mask) < 2:
                continue
            ys.append(to_grid(xv[mask], yv[mask], x_grid, smooth_window))
        if not ys:
            continue
        mat = np.vstack(ys)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        style = STYLE.get(alg, {})
        ax.plot(
            x_grid, mean,
            label=f'{alg} (n={len(ys)})',
            color=style.get('color', None),
            linestyle=style.get('linestyle', '-'),
            linewidth=2.0,
        )
        ax.fill_between(
            x_grid, mean - std, mean + std,
            color=style.get('color', None), alpha=0.20, linewidth=0.0
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', frameon=True)
    ax.set_title(f'{scenario}: {y_key} vs {x_key}')
    ax.set_xlabel('Timesteps' if x_key == 'timesteps' else 'Wall-clock Time (sec)')
    ax.set_ylabel('Episode Reward' if y_key == 'episode_reward' else 'Coverage (%)')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize_metrics(runs: List[RunSeries], out_csv: Path) -> None:
    rows = []
    groups: Dict[Tuple[str, str], List[RunSeries]] = {}
    for run in runs:
        groups.setdefault((run.algorithm, run.scenario), []).append(run)

    for (alg, scenario), rr in sorted(groups.items()):
        conv_rewards, max_covs, steps_to90, total_times = [], [], [], []
        for run in rr:
            t = run.timesteps
            r = run.episode_reward
            c = run.coverage_pct
            w = run.wall_time_sec
            mask_r = np.isfinite(t) & np.isfinite(r)
            if np.any(mask_r):
                t1, r1 = t[mask_r], r[mask_r]
                cutoff = np.nanmax(t1) * 0.9
                tail = r1[t1 >= cutoff]
                if len(tail):
                    conv_rewards.append(float(np.nanmean(tail)))
                total_times.append(float(np.nanmax(w[np.isfinite(w)])) if np.any(np.isfinite(w)) else np.nan)
            mask_c = np.isfinite(t) & np.isfinite(c)
            if np.any(mask_c):
                t2, c2 = t[mask_c], c[mask_c]
                max_covs.append(float(np.nanmax(c2)))
                hit = np.where(c2 >= 90.0)[0]
                steps_to90.append(float(t2[hit[0]]) if len(hit) else np.nan)
            else:
                max_covs.append(np.nan)
                steps_to90.append(np.nan)

        mean_time_sec = float(np.nanmean(total_times)) if total_times else np.nan
        time_h = mean_time_sec / 3600.0 if np.isfinite(mean_time_sec) else np.nan
        rows.append({
            'Algorithm': alg,
            'Scenario': scenario,
            'Converged Reward': float(np.nanmean(conv_rewards)) if conv_rewards else np.nan,
            'Max Coverage': float(np.nanmean(max_covs)) if max_covs else np.nan,
            'Steps to 90% Coverage': float(np.nanmean(steps_to90)) if steps_to90 else np.nan,
            'Total Training Time (hours)': time_h,
            'Total Training Time (H:M:S)': (
                f'{int(mean_time_sec // 3600):02d}:{int((mean_time_sec % 3600) // 60):02d}:{int(mean_time_sec % 60):02d}'
                if np.isfinite(mean_time_sec) else 'N/A'
            ),
            'Num Runs': len(rr),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'Algorithm', 'Scenario', 'Converged Reward', 'Max Coverage',
            'Steps to 90% Coverage', 'Total Training Time (hours)',
            'Total Training Time (H:M:S)', 'Num Runs'
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description='Plot algorithm comparison curves and export summary table.'
    )
    parser.add_argument('--logs_dir', type=str, default=str(script_dir / 'logs'))
    parser.add_argument('--output_dir', type=str, default=str(script_dir / 'comparison_outputs'))
    parser.add_argument('--smooth_window', type=int, default=20)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(logs_dir, repo_root)
    if not runs:
        print(f'[plot_comparison] No compatible logs found in {logs_dir}')
        return

    scenarios = sorted(set(r.scenario for r in runs if r.scenario in ('Circle1', 'Circle8')))
    for scenario in scenarios:
        for x_key in ('timesteps', 'wall_time_sec'):
            for y_key in ('episode_reward', 'coverage_pct'):
                filename = f'{scenario.lower()}_{y_key}_vs_{x_key}.png'
                plot_one(
                    runs=runs,
                    scenario=scenario,
                    x_key=x_key,
                    y_key=y_key,
                    smooth_window=max(1, int(args.smooth_window)),
                    out_path=out_dir / filename,
                )

    summary_path = out_dir / 'comparison_results.csv'
    summarize_metrics(runs, summary_path)
    print(f'[plot_comparison] Saved figures and table to: {out_dir}')


if __name__ == '__main__':
    main()