# Data manifest

## Empirical aggregates

`data/derived/empirical/monthly_theme_panel.csv` contains the 180-month panel used for numerical re-fitting. Other empirical CSVs contain the frozen comparison results or aggregate co-occurrence data. Focal model coefficients and effects are checked against these references; all fitted terms are exported anew.

## Semantic evaluation

All NPY files contain float32 arrays with 1,024 coordinates per row and are loaded with `allow_pickle=False`.

- `m_generated_embeddings.npy`: 2,000 rows. The corresponding index contains request ID, anonymous profile ID, architecture, month, weight and embedding row.
- `m_reference_embeddings.npy`: 5,426 rows. The index contains month, embedding row and replacement reference/group IDs. The replacements preserve the original lexical ordering and duplicate membership; source sample IDs and topic columns are omitted.
- `p1_embeddings_N200.npy`: 2,000 rows for the two convergence scenarios. N=50/N=100 membership was recovered from the original frozen cohort embedding-index tables. Subsets use the final N=200 representation, matching the original convergence analysis.
- `p3_embeddings.npy`: 4,500 rows for nine scenarios and five horizons. The array hash matches the original P3 embedding contract.
- `p3_index.csv`: a compact column projection of the original 4,500-row Wave3 embedding index. The original index was supplied and verified against the saved P3 corpus metadata. The projection retains request ID/hash, scenario, profile, month, weight, horizon and embedding row; title/content are provided separately in `p3_generated_text.csv`. It is no longer reconstructed from other tables.
- `p3_folds.csv`: the original saved 100-profile fold assignment.

The 2026-09-17 numerical audit confirmed that this indexed representation reproduces the stored P3 main table, secondary horizon results, randomization p-values and BH q-values. The retained projection and the generated-text CSV preserve the original numerical row alignment without publishing unused index columns.

## Evaluation predictions

The classifier test file contains saved seven-label gold values, probabilities, predictions, and salted anonymous row IDs. It corresponds to configuration 5 in the original grid-search test-metrics record. No thresholds are optimized on this test file.

The six relevance prediction files retain saved gold/predicted labels and selected numeric fields. Original message text and free-form model explanations were removed during collection. The final projections include elapsed time, parse success and token counts. Table 4(A) latency is recalculated as a mean of the original per-request elapsed times, not measured with a new API run.

## Frozen generation outputs

The original M-series and P2 Parquet outputs are included byte-for-byte, along with the N=200 convergence output. They are provided as stored output records, not as evidence of a new hosted-API run in this audit. The offline pipeline does not require a Parquet decoder. SHA-256 checks protect the retained bytes. Companion CSVs contain request IDs and the original generated title/body strings; their text-to-encoding corpus fingerprints are checked by the default pipeline. The Parquet records have not been re-decoded in this packaging pass.

## Source calculations

`m_methods.py`, `p1_methods.py`, and `p3_methods.py` contain the pure numerical functions extracted from the recovered original calculation modules/notebooks. Top-level experiment actions, API execution code, author-specific paths and internal manuscript-writing directions are excluded from these numerical modules.

This distribution does not reconstruct private source messages, train the classifier or reassess literal-copy risks in its default workflow. Optional hosted-generation code is separate and opt-in; it issues prepared prompts with recorded settings rather than guaranteeing identical outputs. New timing measurements would be different observations from the historical elapsed times used in the paper.

## Optional API inputs and credentials

`data/prompts/` contains the supplied experimental prompt text. `data/api/request_profiles.json` contains recorded request parameters without key values. `examples/api/` contains illustrative synthetic inputs, not the private experimental prompt roster. `scripts/api/` is the new cost-controlled execution interface; no historical notebook with top-level paid actions is included as a runnable default. Prepared private plans and all new live-run artifacts remain outside the frozen inputs and are ignored by Git.

## Text-to-encoding verification

`data/semantic/text_encoding_manifest.json` records the original corpus fingerprints for M, P1 and P3. Text joins are `title + newline + content` for P1/P3 and whitespace-normalized title/body joined by a newline for M. These rules were retained from the corresponding source encoders. The three CSV projections were checked against those fingerprints; they were not reconstructed from the numerical values in the paper.

## Historical API timing

`docs/llm_api_timing_evidence.json` and `docs/provenance/` record timing metadata recovered from original logs and valid request caches. The M-series extract retains completion timestamps; the P1 extract retains direct start and completion timestamps. These extracts contain no source prompts, response IDs, or credentials. The README separately labels query-bundle and P2 freeze times as artifact timestamps; missing request dates are not replaced with file modification times. These documentation assets do not alter the frozen numerical inputs.
