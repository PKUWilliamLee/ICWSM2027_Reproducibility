"""Optionally re-encode saved generated text with a pre-existing local BGE snapshot.

The default prints usage only. This script never calls a hosted generation API,
never downloads a model and never replaces the frozen vectors used for verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from verify_utils import ROOT, sha256, write_json
from validate_saved_texts import load_texts


def directory_digest(directory: Path) -> str:
    lines = [f'{p.relative_to(directory).as_posix()}|{p.stat().st_size}|{sha256(p).upper()}'
             for p in sorted(directory.rglob('*')) if p.is_file()]
    if not lines:
        raise ValueError('The local model directory is empty')
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encode', action='store_true', help='Explicit opt-in to local model inference')
    parser.add_argument('--dataset', choices=['m', 'p1', 'p3'], default='p3')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--run-id', default='local_bge_check')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if not args.encode:
        parser.print_help()
        print('\nNo model was loaded; no network request was sent.')
        return 0
    if args.model_dir is None or not args.model_dir.is_dir() or args.batch_size < 1:
        parser.error('--encode requires an existing --model-dir and a positive batch size')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        parser.error('Invalid run ID')
    params = json.loads((ROOT / 'data/semantic/analysis_parameters.json').read_text())
    fingerprint = directory_digest(args.model_dir)
    if fingerprint != params['BGE_local_directory_manifest_sha256']:
        raise ValueError('Local model files do not match the recorded snapshot. Preserve the frozen vectors; do not substitute a different encoder.')
    dest = ROOT / 'outputs/local_encoding' / args.run_id
    if dest.exists():
        raise FileExistsError('Choose a new run ID; local-encoding outputs are never overwritten')
    index, texts, corpus = load_texts(args.dataset)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(str(args.model_dir.resolve()), device=args.device,
                                local_files_only=True, trust_remote_code=False)
    emb = np.asarray(model.encode(texts, batch_size=args.batch_size,
                                  show_progress_bar=True, convert_to_numpy=True,
                                  normalize_embeddings=True), dtype=np.float32)
    if emb.shape != (len(texts), 1024) or not np.isfinite(emb).all():
        raise ValueError('Unexpected embedding output')
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4, rtol=0)
    frozen_name = {'m': 'm_generated_embeddings.npy', 'p1': 'p1_embeddings_N200.npy', 'p3': 'p3_embeddings.npy'}[args.dataset]
    frozen = np.load(ROOT / 'data/semantic' / frozen_name, allow_pickle=False)
    dest.mkdir(parents=True, exist_ok=False)
    np.save(dest / 'embeddings.npy', emb, allow_pickle=False)
    index.to_csv(dest / 'index.csv', index=False)
    write_json(dest / 'encoding_report.json', {**corpus, 'model_id': params['BGE_model_id'],
        'revision': params['BGE_revision'], 'local_snapshot_sha256': fingerprint,
        'maximum_coordinate_difference': float(np.max(np.abs(emb - frozen))),
        'generation_api_calls': 0, 'frozen_vectors_replaced': False})
    print('Local re-encoding completed. The default numerical inputs were not replaced.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
