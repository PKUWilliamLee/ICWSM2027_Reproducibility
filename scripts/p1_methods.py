"""Frozen numerical analysis functions used by the study."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd

POLICY_HORIZONS = ((0, '2021-08'), (6, '2022-02'), (12, '2022-08'), (24, '2023-08'), (36, '2024-08'))
CONVERGENCE_THRESHOLDS = {'pooled_direction_cosine_min': 0.95, 'min_horizon_direction_cosine_min': 0.9, 'mean_shift_norm_abs_tolerance': 0.01, 'mean_shift_norm_relative_tolerance': 0.1, 'paired_cosine_displacement_abs_tolerance': 0.005, 'paired_cosine_displacement_relative_tolerance': 0.1, 'max_horizon_displacement_abs_tolerance': 0.01, 'horizon_trajectory_cosine_min': 0.98}

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float('nan')
    return float(np.dot(a, b) / denom)

def relative_change(a: float, b: float) -> float:
    denom = max(abs(float(b)), 1e-12)
    return float(abs(float(a) - float(b)) / denom)

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.sum(np.asarray(values) * w[:, None], axis=0)

def weighted_scalar_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(np.sum(np.asarray(values) * w))

def metrics_for_n(paired: dict[str, Any], n: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = paired['pairs']
    delta = paired['delta']
    displacement = paired['displacement']
    weights = paired['weights']
    mean_delta = weighted_mean(delta, weights)
    pooled = {'N': n, 'mean_shift_norm': float(np.linalg.norm(mean_delta)), 'paired_cosine_displacement': weighted_scalar_mean(displacement, weights), 'mean_delta': mean_delta}
    horizon = {}
    for h, month in POLICY_HORIZONS:
        mask = pairs['policy_horizon_months'].astype(int).eq(h).to_numpy()
        h_delta = delta[mask]
        h_disp = displacement[mask]
        h_weights = weights[mask]
        h_mean_delta = weighted_mean(h_delta, h_weights)
        horizon[h] = {'horizon_months': h, 'simulation_month': month, 'mean_shift_norm': float(np.linalg.norm(h_mean_delta)), 'paired_cosine_displacement': weighted_scalar_mean(h_disp, h_weights), 'mean_delta': h_mean_delta}
    return (pooled, horizon)

def compare_n(small_n: int, large_n: int, all_metrics: Mapping[int, Any]) -> dict[str, Any]:
    small = all_metrics[small_n]
    large = all_metrics[large_n]
    s_pool = small['pooled']
    l_pool = large['pooled']
    pooled_direction = safe_cosine(s_pool['mean_delta'], l_pool['mean_delta'])
    s_norm = float(s_pool['mean_shift_norm'])
    l_norm = float(l_pool['mean_shift_norm'])
    norm_abs = abs(s_norm - l_norm)
    norm_rel = relative_change(s_norm, l_norm)
    norm_stable = norm_abs <= CONVERGENCE_THRESHOLDS['mean_shift_norm_abs_tolerance'] or norm_rel <= CONVERGENCE_THRESHOLDS['mean_shift_norm_relative_tolerance']
    s_disp = float(s_pool['paired_cosine_displacement'])
    l_disp = float(l_pool['paired_cosine_displacement'])
    disp_abs = abs(s_disp - l_disp)
    disp_rel = relative_change(s_disp, l_disp)
    disp_stable = disp_abs <= CONVERGENCE_THRESHOLDS['paired_cosine_displacement_abs_tolerance'] or disp_rel <= CONVERGENCE_THRESHOLDS['paired_cosine_displacement_relative_tolerance']
    horizon_direction = []
    s_traj = []
    l_traj = []
    h_disp_diffs = []
    for h, _ in POLICY_HORIZONS:
        s_h = small['horizon'][h]
        l_h = large['horizon'][h]
        horizon_direction.append(safe_cosine(s_h['mean_delta'], l_h['mean_delta']))
        s_traj.append(s_h['paired_cosine_displacement'])
        l_traj.append(l_h['paired_cosine_displacement'])
        h_disp_diffs.append(abs(s_h['paired_cosine_displacement'] - l_h['paired_cosine_displacement']))
    finite_horizon_direction = [x for x in horizon_direction if np.isfinite(x)]
    min_horizon_direction = min(finite_horizon_direction) if len(finite_horizon_direction) == len(POLICY_HORIZONS) else float('nan')
    trajectory_cosine = safe_cosine(np.asarray(s_traj, float), np.asarray(l_traj, float))
    max_horizon_disp_diff = float(max(h_disp_diffs))
    checks = {'pooled_direction': np.isfinite(pooled_direction) and pooled_direction >= CONVERGENCE_THRESHOLDS['pooled_direction_cosine_min'], 'min_horizon_direction': np.isfinite(min_horizon_direction) and min_horizon_direction >= CONVERGENCE_THRESHOLDS['min_horizon_direction_cosine_min'], 'mean_shift_norm': norm_stable, 'paired_displacement': disp_stable, 'max_horizon_displacement': max_horizon_disp_diff <= CONVERGENCE_THRESHOLDS['max_horizon_displacement_abs_tolerance'], 'horizon_trajectory': np.isfinite(trajectory_cosine) and trajectory_cosine >= CONVERGENCE_THRESHOLDS['horizon_trajectory_cosine_min']}
    return {'small_N': small_n, 'large_N': large_n, 'pooled_direction_cosine': pooled_direction, 'min_horizon_direction_cosine': min_horizon_direction, 'mean_shift_norm_small': s_norm, 'mean_shift_norm_large': l_norm, 'mean_shift_norm_abs_change': norm_abs, 'mean_shift_norm_relative_change': norm_rel, 'paired_displacement_small': s_disp, 'paired_displacement_large': l_disp, 'paired_displacement_abs_change': disp_abs, 'paired_displacement_relative_change': disp_rel, 'max_horizon_displacement_abs_change': max_horizon_disp_diff, 'horizon_trajectory_cosine': trajectory_cosine, **{f'check::{k}': bool(v) for k, v in checks.items()}, 'stable': bool(all(checks.values())), 'formal_selection_pair': (small_n, large_n) in {(100, 200), (200, 400)}}
