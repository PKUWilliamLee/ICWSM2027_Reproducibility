"""Frozen numerical analysis functions used by the study."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd

POLICY_HORIZONS = ((0, '2021-08'), (6, '2022-02'), (12, '2022-08'), (24, '2023-08'), (36, '2024-08'))
HORIZON_VALUES = tuple((h for h, _ in POLICY_HORIZONS))
HORIZON_MONTH_MAP = dict(POLICY_HORIZONS)
ALL_SCENARIOS = ('P0', 'P_NORM_ABC', 'P_ABC', 'P_A_ONLY', 'P_B_ONLY', 'P_C_ONLY', 'P_ABC_MINUS_A_STATE', 'P_ABC_MINUS_B_STATE', 'P_ABC_MINUS_C_STATE')
FULL_CONTRAST_ID = 'full_policy'
FULL_CONTRAST = ('P0', 'P_ABC')
PRIMARY_CONTRASTS = {'information': ('P0', 'P_NORM_ABC'), 'state_given_information': ('P_NORM_ABC', 'P_ABC'), 'sufficiency_A': ('P0', 'P_A_ONLY'), 'sufficiency_B': ('P0', 'P_B_ONLY'), 'sufficiency_C': ('P0', 'P_C_ONLY'), 'necessity_A': ('P_ABC_MINUS_A_STATE', 'P_ABC'), 'necessity_B': ('P_ABC_MINUS_B_STATE', 'P_ABC'), 'necessity_C': ('P_ABC_MINUS_C_STATE', 'P_ABC')}
N_PROFILES = 100
N_HORIZONS = 5
N_SCENARIOS = 9
EXPECTED_ROWS = 4500
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20260816
CROSSFIT_FOLDS = 5
CROSSFIT_SEED = 20260816
ALPHA = 0.05

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest().upper()

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float('nan')
    return float(np.dot(a, b) / denom)

def normalized_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise AssertionError('analysis_weight 必须全部有限且 > 0')
    return w / w.sum()

def weighted_mean_vector(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = normalized_weights(weights)
    return np.sum(np.asarray(values) * w[:, None], axis=0)

def weighted_mean_scalar(values: np.ndarray, weights: np.ndarray) -> float:
    w = normalized_weights(weights)
    return float(np.sum(np.asarray(values, dtype=float) * w))

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError('p_values 必须一维')
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q

def build_embedding_cube(index: pd.DataFrame, embeddings: np.ndarray):
    profile_ids = sorted(index['profile_sim_id'].astype(str).unique().tolist())
    if len(profile_ids) != N_PROFILES:
        raise AssertionError('profile count != 100')
    p_to_i = {p: i for i, p in enumerate(profile_ids)}
    h_to_i = {h: i for i, h in enumerate(HORIZON_VALUES)}
    s_to_i = {s: i for i, s in enumerate(ALL_SCENARIOS)}
    dims = embeddings.shape[1]
    cube = np.full((N_SCENARIOS, N_PROFILES, N_HORIZONS, dims), np.nan, dtype=np.float32)
    request_hash_cube = np.empty((N_SCENARIOS, N_PROFILES, N_HORIZONS), dtype=object)
    request_row_cube = np.empty_like(request_hash_cube, dtype=object)
    profile_weight = np.full(N_PROFILES, np.nan, dtype=float)
    for row in index.to_dict('records'):
        s = str(row['scenario_id'])
        p = str(row['profile_sim_id'])
        h = int(row['policy_horizon_months'])
        e = int(row['embedding_row'])
        si = s_to_i[s]
        pi = p_to_i[p]
        hi = h_to_i[h]
        cube[si, pi, hi] = embeddings[e]
        request_hash_cube[si, pi, hi] = str(row['request_hash'])
        request_row_cube[si, pi, hi] = str(row['request_row_id'])
        w = float(row['analysis_weight'])
        if np.isnan(profile_weight[pi]):
            profile_weight[pi] = w
        elif not np.isclose(profile_weight[pi], w, atol=1e-12, rtol=0.0):
            raise AssertionError(f'profile weight drift: {p}')
    if np.isnan(cube).any():
        raise AssertionError('embedding cube 有缺失')
    if np.any(pd.isna(request_hash_cube)):
        raise AssertionError('request_hash cube 有缺失')
    if np.any(~np.isfinite(profile_weight)):
        raise AssertionError('profile weights 有缺失')
    return {'profile_ids': profile_ids, 'weights': profile_weight, 'cube': cube, 'request_hash_cube': request_hash_cube, 'request_row_cube': request_row_cube, 'scenario_index': s_to_i, 'horizon_index': h_to_i}

def scenario_embeddings(cube_info: dict[str, Any], scenario: str) -> np.ndarray:
    return cube_info['cube'][cube_info['scenario_index'][scenario]]

def contrast_delta(cube_info: dict[str, Any], left: str, right: str) -> np.ndarray:
    return scenario_embeddings(cube_info, right) - scenario_embeddings(cube_info, left)

def contrast_displacement(cube_info: dict[str, Any], left: str, right: str) -> np.ndarray:
    a = scenario_embeddings(cube_info, left)
    b = scenario_embeddings(cube_info, right)
    return 1.0 - np.sum(a * b, axis=2)

def temporal_concat(delta: np.ndarray) -> np.ndarray:
    return delta.reshape(delta.shape[0], -1) / math.sqrt(N_HORIZONS)

def make_sign_matrix(n_profiles: int, n_permutations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, n_profiles), dtype=np.int8)
    signs = signs * 2 - 1
    return signs.astype(np.float32)

def signflip_test(profile_vectors: np.ndarray, weights: np.ndarray, signs: np.ndarray) -> dict[str, Any]:
    values = np.asarray(profile_vectors, dtype=np.float32)
    w = normalized_weights(weights).astype(np.float32)
    A = values * w[:, None]
    G = A @ A.T
    observed_sq = float(np.sum(G))
    observed_sq = max(observed_sq, 0.0)
    observed = math.sqrt(observed_sq)
    perm_sq = np.einsum('bi,ij,bj->b', signs, G, signs, optimize=True)
    perm_sq = np.maximum(perm_sq.astype(float), 0.0)
    perm_stats = np.sqrt(perm_sq)
    ge = int(np.sum(perm_stats >= observed - 1e-12))
    p = (ge + 1.0) / (len(signs) + 1.0)
    return {'T_observed': observed, 'p_value': float(p), 'perm_mean': float(perm_stats.mean()), 'perm_sd': float(perm_stats.std(ddof=1)), 'perm_q95': float(np.quantile(perm_stats, 0.95)), 'perm_q99': float(np.quantile(perm_stats, 0.99))}

def deterministic_folds(profile_ids: list[str], n_folds: int, seed: int) -> np.ndarray:
    ranked = sorted(enumerate(profile_ids), key=lambda item: sha256_text(f'P3_CROSSFIT|{seed}|{item[1]}'))
    folds = np.empty(len(profile_ids), dtype=int)
    for rank, (original_idx, _) in enumerate(ranked):
        folds[original_idx] = rank % n_folds
    return folds

def crossfit_directional_contribution(D_component: np.ndarray, D_full: np.ndarray, weights: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    w_global = normalized_weights(weights)
    component_projection = np.zeros(len(weights), dtype=float)
    full_projection = np.zeros(len(weights), dtype=float)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        train_full_mean = weighted_mean_vector(D_full[train], weights[train])
        norm = float(np.linalg.norm(train_full_mean))
        if norm <= 1e-12:
            raise RuntimeError(f'crossfit fold {fold}: train full direction 零向量')
        u = train_full_mean / norm
        component_projection[test] = D_component[test] @ u
        full_projection[test] = D_full[test] @ u
    numerator = float(np.sum(w_global * component_projection))
    denominator = float(np.sum(w_global * full_projection))
    if abs(denominator) <= 1e-12:
        raise RuntimeError('crossfit full held-out projection denominator ≈ 0')
    return {'crossfit_component_projection': numerator, 'crossfit_full_projection': denominator, 'crossfit_directional_contribution_to_full': numerator / denominator}

def evaluate_all(cube_info: dict[str, Any], signs: np.ndarray) -> dict[str, pd.DataFrame]:
    weights = cube_info['weights']
    profile_ids = cube_info['profile_ids']
    folds = deterministic_folds(profile_ids, CROSSFIT_FOLDS, CROSSFIT_SEED)
    contrast_specs = {FULL_CONTRAST_ID: FULL_CONTRAST, **PRIMARY_CONTRASTS}
    delta_cache = {cid: contrast_delta(cube_info, left, right) for cid, (left, right) in contrast_specs.items()}
    displacement_cache = {cid: contrast_displacement(cube_info, left, right) for cid, (left, right) in contrast_specs.items()}
    D_cache = {cid: temporal_concat(delta) for cid, delta in delta_cache.items()}
    D_full = D_cache[FULL_CONTRAST_ID]
    full_mean_D = weighted_mean_vector(D_full, weights)
    full_omnibus_magnitude = float(np.linalg.norm(full_mean_D))
    full_delta = delta_cache[FULL_CONTRAST_ID]
    repeated_weights = np.repeat(normalized_weights(weights) / N_HORIZONS, N_HORIZONS)
    pooled_full = weighted_mean_vector(full_delta.reshape(N_PROFILES * N_HORIZONS, full_delta.shape[2]), repeated_weights)
    full_pooled_magnitude = float(np.linalg.norm(pooled_full))
    main_rows = []
    for cid, (left, right) in contrast_specs.items():
        delta = delta_cache[cid]
        displacement = displacement_cache[cid]
        D = D_cache[cid]
        mean_D = weighted_mean_vector(D, weights)
        omnibus_magnitude = float(np.linalg.norm(mean_D))
        pooled_delta = weighted_mean_vector(delta.reshape(N_PROFILES * N_HORIZONS, delta.shape[2]), repeated_weights)
        pooled_magnitude = float(np.linalg.norm(pooled_delta))
        pooled_displacement = weighted_mean_scalar(displacement.reshape(-1), repeated_weights)
        test = signflip_test(D, weights, signs)
        alignment = safe_cosine(mean_D, full_mean_D)
        naive_contribution = float(np.dot(mean_D, full_mean_D / full_omnibus_magnitude)) / full_omnibus_magnitude if full_omnibus_magnitude > 1e-12 else float('nan')
        crossfit = crossfit_directional_contribution(D, D_full, weights, folds)
        main_rows.append({'contrast_id': cid, 'family': 'benchmark_not_in_primary_FDR' if cid == FULL_CONTRAST_ID else 'primary_8', 'left_scenario': left, 'right_scenario': right, 'omnibus_temporal_magnitude': omnibus_magnitude, 'relative_omnibus_magnitude_to_full': omnibus_magnitude / full_omnibus_magnitude if full_omnibus_magnitude > 1e-12 else float('nan'), 'pooled_effect_magnitude': pooled_magnitude, 'relative_pooled_magnitude_to_full': pooled_magnitude / full_pooled_magnitude if full_pooled_magnitude > 1e-12 else float('nan'), 'pooled_paired_cosine_displacement': pooled_displacement, 'temporal_alignment_with_full_direction': alignment, 'naive_directional_contribution_to_full': naive_contribution, **crossfit, 'T_observed': test['T_observed'], 'raw_p': test['p_value'], 'perm_mean': test['perm_mean'], 'perm_sd': test['perm_sd'], 'perm_q95': test['perm_q95'], 'perm_q99': test['perm_q99']})
    main = pd.DataFrame(main_rows)
    primary_mask = main['family'].eq('primary_8')
    primary_q = bh_fdr(main.loc[primary_mask, 'raw_p'].to_numpy(float))
    main['BH_q_primary_8'] = np.nan
    main.loc[primary_mask, 'BH_q_primary_8'] = primary_q
    main['reject_BH_0_05_primary'] = False
    main.loc[primary_mask, 'reject_BH_0_05_primary'] = primary_q < ALPHA
    horizon_rows = []
    for cid, (left, right) in contrast_specs.items():
        delta = delta_cache[cid]
        displacement = displacement_cache[cid]
        for hi, (h, month) in enumerate(POLICY_HORIZONS):
            h_delta = delta[:, hi, :]
            h_disp = displacement[:, hi]
            h_mean = weighted_mean_vector(h_delta, weights)
            h_full_mean = weighted_mean_vector(full_delta[:, hi, :], weights)
            h_test = signflip_test(h_delta, weights, signs)
            horizon_rows.append({'contrast_id': cid, 'family': 'benchmark_horizon' if cid == FULL_CONTRAST_ID else 'secondary_40', 'left_scenario': left, 'right_scenario': right, 'horizon_months': h, 'simulation_month': month, 'effect_magnitude': float(np.linalg.norm(h_mean)), 'paired_cosine_displacement': weighted_mean_scalar(h_disp, weights), 'alignment_with_full_horizon_direction': safe_cosine(h_mean, h_full_mean), 'T_observed': h_test['T_observed'], 'raw_p': h_test['p_value']})
    horizon = pd.DataFrame(horizon_rows)
    secondary_mask = horizon['family'].eq('secondary_40')
    secondary_q = bh_fdr(horizon.loc[secondary_mask, 'raw_p'].to_numpy(float))
    horizon['BH_q_secondary_40'] = np.nan
    horizon.loc[secondary_mask, 'BH_q_secondary_40'] = secondary_q
    horizon['reject_BH_0_05_secondary'] = False
    horizon.loc[secondary_mask, 'reject_BH_0_05_secondary'] = secondary_q < ALPHA
    component_rows = []
    for component in ('A', 'B', 'C'):
        suff = main.loc[main['contrast_id'].eq(f'sufficiency_{component}')].iloc[0]
        need = main.loc[main['contrast_id'].eq(f'necessity_{component}')].iloc[0]
        component_rows.append({'component': component, 'sufficiency_omnibus_magnitude': suff['omnibus_temporal_magnitude'], 'sufficiency_relative_to_full': suff['relative_omnibus_magnitude_to_full'], 'sufficiency_alignment': suff['temporal_alignment_with_full_direction'], 'sufficiency_crossfit_directional_contribution': suff['crossfit_directional_contribution_to_full'], 'sufficiency_raw_p': suff['raw_p'], 'sufficiency_BH_q': suff['BH_q_primary_8'], 'sufficiency_reject_BH_0_05': suff['reject_BH_0_05_primary'], 'necessity_omnibus_magnitude': need['omnibus_temporal_magnitude'], 'necessity_relative_to_full': need['relative_omnibus_magnitude_to_full'], 'necessity_alignment': need['temporal_alignment_with_full_direction'], 'necessity_crossfit_directional_contribution': need['crossfit_directional_contribution_to_full'], 'necessity_raw_p': need['raw_p'], 'necessity_BH_q': need['BH_q_primary_8'], 'necessity_reject_BH_0_05': need['reject_BH_0_05_primary']})
    component = pd.DataFrame(component_rows)
    fold_frame = pd.DataFrame({'profile_sim_id': profile_ids, 'analysis_weight': weights, 'crossfit_fold': folds})
    return {'main': main, 'horizon': horizon, 'component': component, 'folds': fold_frame}
