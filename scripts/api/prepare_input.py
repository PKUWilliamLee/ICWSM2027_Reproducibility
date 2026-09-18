"""Convert an authorized prepared CSV/Parquet prompt plan to the optional JSONL interface.

No model is run. Only prepared user_prompt text is accepted; private profile and
memory construction are not silently approximated from generated output text.
"""
from __future__ import annotations
import argparse
import json
import re
import uuid
from pathlib import Path
from request_io import ROOT, check_no_credentials, prepare_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--profile', help='Fixed request profile; otherwise use each row architecture_id')
    parser.add_argument('--cohort-column', help='Optional boolean membership column, e.g. in_N100')
    parser.add_argument('--name', default='prepared_requests')
    args = parser.parse_args(argv)
    if args.input is None:
        parser.print_help()
        return 0
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.name):
        parser.error('Invalid output name')
    import pandas as pd
    if args.input.suffix.lower() == '.parquet':
        frame = pd.read_parquet(args.input)
    elif args.input.suffix.lower() == '.csv':
        frame = pd.read_csv(args.input, keep_default_na=False)
    else:
        parser.error('Input must be CSV or Parquet')
    if args.cohort_column:
        col = frame[args.cohort_column].astype(str).str.lower()
        if not col.isin(['true','false','1','0']).all():
            raise ValueError('Cohort membership must be boolean')
        frame = frame.loc[col.isin(['true','1'])]
    idcol = next((k for k in ['record_id','request_row_id','sample_id'] if k in frame), None)
    if idcol is None or 'user_prompt' not in frame:
        raise ValueError('Input must contain a record/request/sample ID and complete user_prompt text')
    out = ROOT / 'outputs/prepared_api' / (args.name + '.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError('Output already exists; choose a new name')
    rows = []
    for record in frame.to_dict('records'):
        profile = args.profile or record.get('architecture_id')
        obj = {'record_id': str(record[idcol]), 'profile': profile, 'user_prompt': record['user_prompt']}
        if 'allowed_concerns' in record:
            obj['allowed_concerns'] = record['allowed_concerns']
        check_no_credentials(obj)
        rows.append(obj)
    # Validate the entire local plan before committing an output file.
    temp = out.with_name(out.stem + '.' + uuid.uuid4().hex + '.tmp.jsonl')
    try:
        with temp.open('x', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        plan = prepare_plan(temp)
        with out.open('xb') as target:
            target.write(temp.read_bytes())
    finally:
        temp.unlink(missing_ok=True)
    print(f'Prepared {len(plan)} rows locally: {out.relative_to(ROOT).as_posix()}')
    print('No API call was sent. Review the private prompts before any voluntary live run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
