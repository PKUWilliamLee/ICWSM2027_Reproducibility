"""Frozen numerical analysis functions used by the study."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from pathlib import Path
import hashlib
import math
import re
import numpy as np
import pandas as pd

ARCHITECTURES = ('M0', 'M1', 'M2', 'M3')
POLICY_HORIZONS = {0: '2021-08', 6: '2022-02', 12: '2022-08', 24: '2023-08', 36: '2024-08'}
POLICY_HORIZON_MONTHS = tuple(POLICY_HORIZONS.values())
TOP_K_VALUES = (1, 5, 10)

def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return re.sub('\\s+', ' ', str(value)).strip()

def _deduplicate_real_horizon(real_index: pd.DataFrame, real_embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    work = real_index.copy()
    work['sample_id'] = work['sample_id'].astype(str)
    work['duplicate_group_id'] = work['duplicate_group_id'].fillna('').astype(str)
    work['group_key'] = [group if normalize_text(group) else f'sample::{sample}' for group, sample in zip(work['duplicate_group_id'], work['sample_id'])]
    work['embedding_row'] = pd.to_numeric(work['embedding_row'], errors='raise').astype(int)
    chosen = work.sort_values(['month', 'group_key', 'sample_id'], kind='mergesort').drop_duplicates(['month', 'group_key'], keep='first').reset_index(drop=True)
    embeddings = real_embeddings[chosen['embedding_row'].to_numpy(int)]
    return (chosen, np.asarray(embeddings, dtype=np.float32))

def _row_topk_mean(matrix: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k > matrix.shape[1]:
        raise ValueError(f'k={k} 不适用于矩阵列数 {matrix.shape[1]}')
    if k == matrix.shape[1]:
        return matrix.mean(axis=1)
    partitioned = np.partition(matrix, matrix.shape[1] - k, axis=1)
    return partitioned[:, -k:].mean(axis=1)

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) != len(w) or np.any(~np.isfinite(v)) or np.any(~np.isfinite(w)):
        raise ValueError('weighted mean 输入非法')
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError('weighted mean 权重非法')
    return float(np.sum(v * w) / np.sum(w))

def _semantic_scores(synthetic: pd.DataFrame, synthetic_embeddings: np.ndarray, real_index: pd.DataFrame, real_embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_unique, real_unique_embeddings = _deduplicate_real_horizon(real_index, real_embeddings)
    syn_lookup = {str(rid): i for i, rid in enumerate(synthetic['request_row_id'].astype(str))}
    real_unique['real_local_row'] = np.arange(len(real_unique), dtype=int)
    item_parts: list[pd.DataFrame] = []
    horizon_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        arch = synthetic.loc[synthetic['architecture_id'].eq(architecture)].copy()
        arch['month'] = pd.to_datetime(arch['simulation_month']).dt.to_period('M').astype(str)
        for month in POLICY_HORIZON_MONTHS:
            syn_month = arch.loc[arch['month'].eq(month)].copy()
            real_month = real_unique.loc[real_unique['month'].eq(month)].copy()
            syn_indices = [syn_lookup[str(v)] for v in syn_month['request_row_id'].astype(str)]
            real_indices = real_month['real_local_row'].to_numpy(int)
            syn_emb = synthetic_embeddings[np.asarray(syn_indices, dtype=int)]
            real_emb = real_unique_embeddings[real_indices]
            cosine = np.clip(syn_emb @ real_emb.T, -1.0, 1.0).astype(np.float32)
            syn_weights = pd.to_numeric(syn_month['analysis_weight'], errors='raise').to_numpy(float)
            s2r_base_item = cosine.mean(axis=1)
            r2s_base_item = cosine.mean(axis=0)
            for k in TOP_K_VALUES:
                s2r_item = _row_topk_mean(cosine, k)
                r2s_item = _row_topk_mean(cosine.T, k)
                s2r = _weighted_mean(s2r_item, syn_weights)
                r2s = float(np.mean(r2s_item))
                s2r_base = _weighted_mean(s2r_base_item, syn_weights)
                r2s_base = float(np.mean(r2s_base_item))
                bidirectional = 0.5 * (s2r + r2s)
                random_baseline = 0.5 * (s2r_base + r2s_base)
                excess = bidirectional - random_baseline
                horizon_rows.append({'architecture_id': architecture, 'month': month, 'top_k': k, 'synthetic_to_real_topk': s2r, 'real_to_synthetic_topk': r2s, 'bidirectional_fidelity': bidirectional, 'synthetic_to_real_random_baseline': s2r_base, 'real_to_synthetic_random_baseline': r2s_base, 'random_same_horizon_baseline': random_baseline, 'excess_fidelity': excess, 'direction_gap': abs(s2r - r2s), 'synthetic_n': len(syn_month), 'real_unique_n': len(real_month)})
                syn_items = syn_month[['request_row_id', 'profile_sim_id', 'analysis_weight']].copy()
                syn_items['architecture_id'] = architecture
                syn_items['month'] = month
                syn_items['direction'] = 'synthetic_to_real'
                syn_items['top_k'] = k
                syn_items['topk_score'] = s2r_item
                syn_items['random_baseline_score'] = s2r_base_item
                syn_items['excess_score'] = s2r_item - s2r_base_item
                syn_items['unit_id'] = syn_items['profile_sim_id'].astype(str)
                item_parts.append(syn_items)
                real_items = real_month[['sample_id', 'group_key']].copy()
                real_items['architecture_id'] = architecture
                real_items['month'] = month
                real_items['direction'] = 'real_to_synthetic'
                real_items['top_k'] = k
                real_items['topk_score'] = r2s_item
                real_items['random_baseline_score'] = r2s_base_item
                real_items['excess_score'] = r2s_item - r2s_base_item
                real_items['analysis_weight'] = 1.0
                real_items['request_row_id'] = ''
                real_items['profile_sim_id'] = ''
                real_items['unit_id'] = real_items['group_key'].astype(str)
                item_parts.append(real_items)
    item = pd.concat(item_parts, ignore_index=True, sort=False)
    horizon = pd.DataFrame(horizon_rows)
    architecture = horizon.groupby(['architecture_id', 'top_k'], as_index=False).agg(synthetic_to_real_topk=('synthetic_to_real_topk', 'mean'), real_to_synthetic_topk=('real_to_synthetic_topk', 'mean'), bidirectional_fidelity=('bidirectional_fidelity', 'mean'), random_same_horizon_baseline=('random_same_horizon_baseline', 'mean'), excess_fidelity=('excess_fidelity', 'mean'), direction_gap=('direction_gap', 'mean')).sort_values(['top_k', 'excess_fidelity'], ascending=[True, False]).reset_index(drop=True)
    return (item, horizon, architecture)
