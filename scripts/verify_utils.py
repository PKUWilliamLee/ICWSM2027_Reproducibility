"""Validation utilities for frozen numerical inputs and reproduced results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], columns: list[str] | None = None, atol: float = 2e-8, rtol: float = 1e-6) -> dict:
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise AssertionError('Duplicate comparison keys: ' + ', '.join(keys))
    if len(actual) != len(expected): raise AssertionError(f'Row count mismatch: {len(actual)} != {len(expected)}')
    merged = actual.merge(expected, on=keys, how='outer', suffixes=('_actual', '_expected'), indicator=True, validate='one_to_one')
    if not merged['_merge'].eq('both').all(): raise AssertionError('Comparison key sets differ')
    if columns is None: columns = [c for c in expected.select_dtypes(include='number') if c not in keys and c in actual]
    errors = {}
    for col in columns:
        a = pd.to_numeric(merged[col + '_actual']).to_numpy(float)
        b = pd.to_numeric(merged[col + '_expected']).to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)): raise AssertionError('Missing-value pattern mismatch: ' + col)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True, err_msg=col)
        valid = np.isfinite(a) & np.isfinite(b)
        errors[col] = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None
    return {'rows_compared': len(actual), 'absolute_differences': errors}
def array(path: Path, rows: int) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if x.shape != (rows, 1024) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise AssertionError('Invalid frozen embedding array: ' + path.name)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-4, rtol=0)
    return x
def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
def verify_inputs() -> dict:
    manifest = json.loads((ROOT / 'input_checksums.json').read_text(encoding='utf-8'))
    failures = []
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected: failures.append(relative)
    if failures: raise AssertionError('Input checksum mismatch: ' + ', '.join(failures))
    return {'checked_files': len(manifest), 'mismatches': 0}
if __name__ == '__main__': print(json.dumps(verify_inputs(), indent=2))
