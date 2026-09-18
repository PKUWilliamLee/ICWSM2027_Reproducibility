"""Check saved generated text against the original embedding-corpus fingerprints."""
from __future__ import annotations
import hashlib
import json
import re
import pandas as pd
from verify_utils import ROOT, write_json


def load_texts(dataset: str) -> tuple[pd.DataFrame, list[str], dict]:
    manifest = json.loads((ROOT / 'data/semantic/text_encoding_manifest.json').read_text())
    spec = manifest[dataset]
    frame = pd.read_csv(ROOT / spec['file'], keep_default_na=False)
    index = pd.read_csv(ROOT / spec['index'])
    if len(frame) != spec['rows'] or frame.request_row_id.duplicated().any():
        raise AssertionError('Generated-text roster differs: ' + dataset)
    if frame.request_row_id.tolist() != index.request_row_id.tolist():
        raise AssertionError('Generated text and embedding row order differ: ' + dataset)
    for col in ['title', 'content']:
        if not frame[col].map(lambda s: isinstance(s, str) and bool(s.strip())).all():
            raise AssertionError('Generated text contains an empty field: ' + dataset)
    if dataset == 'm':
        texts = [re.sub(r'\s+', ' ', a).strip() + '\n' + re.sub(r'\s+', ' ', b).strip()
                 for a, b in zip(frame.title, frame.content)]
    else:
        texts = [a + '\n' + b for a, b in zip(frame.title, frame.content)]
    corpus = '\n'.join(f'{rid}|{t}' for rid, t in zip(frame.request_row_id, texts))
    actual = hashlib.sha256(corpus.encode('utf-8')).hexdigest().upper()
    if actual != spec['corpus_sha256']:
        raise AssertionError('Generated-text corpus does not match the original embedding metadata: ' + dataset)
    return index, texts, {'rows': len(texts), 'corpus_sha256': actual}


def main() -> None:
    report = {k: load_texts(k)[2] for k in ['m', 'p1', 'p3']}
    write_json(ROOT / 'outputs/saved_text_checks.json', report)
    print('Saved generated texts match all three original embedding-corpus fingerprints.')


if __name__ == '__main__':
    main()
