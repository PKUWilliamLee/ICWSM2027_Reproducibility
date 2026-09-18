# Historical API timing provenance

All calendar times below are in UTC. This note documents the original saved study responses, not a new API run. Model identifiers are reported as stored in the source records; they are not claims about present endpoint availability.

## M-series: completion timestamps

The original `m_series_generation_v4/m_series_api_log_v4.parquet` contains 2,000 valid rows and 2,000 unique request hashes. Every hash matches a row in the distributed `data/frozen_api_outputs/m_series_synthetic_v4.parquet`. Its recorded model is `deepseek-v4-pro`.

The `completed_at_utc` field ranges from `2026-08-13T15:35:53.634834+00:00` to `2026-08-14T03:22:05.902532+00:00`. There are 1,958 retained response completions on 13 August and 42 on 14 August (UTC). All 2,000 timestamps are present. The decoded minimum and maximum were independently checked against the original Parquet column statistics.

The exported log does not contain `started_at_utc`. Although it contains `elapsed_seconds`, no inferred start time is substituted for a directly recorded start field in the README. The interval includes pauses and does not imply continuous execution.

[Minimal M-series timing extract](provenance/m_series_response_times.csv) preserves the request hash, completion time, elapsed duration, and returned model only. It contains no prompts, response IDs, source messages, or credentials.

## P1: retained cache request intervals

The original `p1_n_convergence_v1/api_cache/` records contain `response_metadata.started_at_utc` and `response_metadata.completed_at_utc`. The 1,600 successful retained records match all distinct hashes in both the final N=200 request plan and the distributed N=200 generated-output Parquet. One response can supply more than one logical row when canonical request hashes agree.

| Nested cohort | Logical rows | Distinct retained requests | Earliest retained start (UTC) | Latest retained completion (UTC) |
|---|---:|---:|---|---|
| N=50 | 500 | 400 | 2026-08-15 11:57:14.629852 | 2026-08-15 12:14:00.682750 |
| N=100 | 1,000 | 800 | 2026-08-15 11:57:14.629852 | 2026-08-15 12:30:43.164630 |
| N=200 | 2,000 | 1,600 | 2026-08-15 11:57:14.629852 | 2026-08-15 13:20:19.163654 |

These are cumulative nested-cohort windows, not three independent runs with repeated starts. They cover successful retained records, not all earlier failures or retries. [Minimal P1 timing extract](provenance/p1_retained_request_times.csv) also records nested-cohort membership and whether the request was reused in the final P-series output.

## Final P-series: cache reuse and freeze provenance

The distributed `p2_full9_source_index_N100_v1.csv` contains 4,500 logical rows and 3,644 distinct request hashes:

| Source attribution | Logical rows | Distinct request hashes within group |
|---|---:|---:|
| P1-stage rows using P1 cache | 1,000 | 800 |
| P2-stage rows also using P1 cache | 656 | 356 |
| P2-stage rows using P2 cache | 2,844 | 2,844 |

The 356 hashes in the second row overlap the 800 in the first row. Across the first two rows there are **800**, not 1,156, distinct P1 requests. All 1,656 P1-backed logical rows were matched to those original cache records, dated from `2026-08-15T11:57:14.629852+00:00` to `2026-08-15T12:30:43.164630+00:00` (earliest retained start to latest completion).

The original `p2_generation_freeze_contract_v1.json` records `created_at_utc = 2026-08-16T05:12:36.802805+00:00`; its `full9_sha256` matches the distributed final generated-output Parquet. This establishes the output artifact's freeze time. The verified P2 output exports and executed notebooks do not supply per-request calendar times for the additional 2,844 P2 requests. Their exact start/end window is therefore left unspecified. It would be incorrect to treat the freeze timestamp as the last API response or to describe all 4,500 rows as new requests on 16 August.

## Retrieval queries: artifact evidence only

The original `queries_400_v8.parquet` has 1,200 query rows for 400 profiles. Its SHA-256 matches `files.queries.sha256` in the original D1.3B bundle manifest, whose `created_at_utc` is `2026-08-12T09:03:59.252107+00:00`. The output identifies 90 query rows imported from 30 pilot profiles and 1,110 query rows generated for 370 additional profiles. It has no per-request timestamp column. The bundle creation time is not assigned as a common call date to these two groups.

## Relevance screening: latency is not a calendar date

The six released evaluation prediction tables retain `elapsed_sec`, `parsed_ok`, and token counts for 6,000 predictions. They do not contain absolute request timestamps. Their historical average latencies remain reproducible, but neither those durations nor the source inventory's filenames, backup names, filesystem times, or upload dates establish exact API-call dates. Calendar dates for the separate full-corpus screening run are also not established by the verified records.

## Machine-readable evidence and unchanged numerical inputs

[llm_api_timing_evidence.json](llm_api_timing_evidence.json) records the source hashes, coverage checks, exact boundaries, and unavailable fields as `null`. The timing CSVs are column projections of original response/cache metadata. They are supplementary documentation and are not substituted for any frozen numerical inputs. This documentation update changes no model outputs, embeddings, parameters, numerical code, API-execution safeguards, or reference results. No new hosted request was made to produce it.
