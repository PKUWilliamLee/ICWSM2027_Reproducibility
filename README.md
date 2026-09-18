# ICWSM 2027 reproducibility materials

This repository provides the numerical analysis code, saved model outputs, and indexed intermediate data for **Do You Hear the People Complain? Understanding and Simulating Public Appeals on E-petition Forums during China's Education Reform**.

### Scope and restricted source data

The complete CFPS source microdata and the raw public-appeal texts are not redistributed because they contain or may contain sensitive or personally identifying information and are subject to privacy and access restrictions. Accordingly, this repository is scoped to reproducing the paper's **analytical figures, inferential/statistical tables, and validation results** from privacy-preserving derived data, saved model outputs, and indexed intermediate data. Purely descriptive tables and schematic/conceptual figures are outside the numerical reproduction target.

## Reproduction and optional API generation

**The default reproduction workflow is offline and does not require an API key.** For the language-model experiments, reproduction starts from the saved outputs used in the study and their corresponding frozen BGE embeddings. The workflow checks the saved generated text against the original embedding-corpus fingerprints, then recomputes the downstream statistics, tables, and figure inputs. It does not request new language-model responses.

We also provide optional code for reconnecting to the hosted APIs and issuing generation or screening requests using prepared prompts and the recorded request configurations. All credential examples use placeholders. Users may supply their own keys through environment variables and explicitly enable a bounded live run. **Providing a key alone does not enable API calls:** the runner defaults to a local dry run, and the accompanying notebook's Run All executes only dry-run examples.

A new API run is a new set of observations, not an exact replay of the saved study outputs. Responses and timing may differ, and hosted model aliases or serving implementations may change. Running the hosted models again is not necessary to execute the provided numerical reproduction. Newly generated responses are written to a separate directory and never replace the study's frozen inputs.

### Quick start: reproduce the reported numerical results

The default workflow is designed to run on a recent Python 3 installation rather than requiring one exact patch version. The release was directly verified with **Python 3.13.5 on Linux** and subsequently completed successfully with **Python 3.12 on Windows**. Python **3.11–3.13** is the intended compatibility range for the supplied code and dependency ranges. `requirements.txt` therefore uses compatible version ranges rather than exact pins. For archival purposes, `requirements-lock.txt` records the exact package versions used in the Linux release verification. Dependency installation needs internet access; the reproduction command itself does not.

Windows PowerShell, from the extracted repository directory:

```powershell
# See which Python 3 versions are already installed (optional).
py -0p

# Use an installed Python 3.11, 3.12, or 3.13. `py -3` normally selects
# an available Python 3 installation and does not require Python 3.13 specifically.
py -3 -m venv ..\.venv-icwsm
& ..\.venv-icwsm\Scripts\python.exe -m pip install --upgrade pip
& ..\.venv-icwsm\Scripts\python.exe -m pip install -r requirements.txt
& ..\.venv-icwsm\Scripts\python.exe .\scripts\reproduce_all.py
```

If several Python versions are installed and `py -3` chooses a version outside the intended range, replace it with an installed version such as `py -3.12` or `py -3.11`.

Linux/macOS:

```bash
python3 --version
python3 -m venv ../.venv-icwsm
../.venv-icwsm/bin/python -m pip install --upgrade pip
../.venv-icwsm/bin/python -m pip install -r requirements.txt
../.venv-icwsm/bin/python scripts/reproduce_all.py
```

To reproduce the **exact package set used in our release verification** instead of using compatible versions, install `requirements-lock.txt` in the virtual environment:

```bash
python -m pip install -r requirements-lock.txt
```

The numerical verification checks the results against frozen reference values, so a compatible dependency environment is acceptable if it finishes with `offline_numerical_checks_passed`; exact package-version identity is not itself a success criterion.

For the 10,000-draw sign-flip tests, deterministic effect sizes, vectors, fold assignments, and the permutation sign matrix are checked normally. Monte-Carlo p-values are compared at their discrete resolution: a reproduced p-value may differ from the archived value by at most **one permutation exceedance count** (`1 / 10001`) to accommodate a floating-point tie at the observed-statistic boundary across BLAS/platform implementations. BH-adjusted values must remain within the corresponding one-count propagation bound, and all 0.05 rejection decisions must be identical. This tolerance does not permit a different seed, sign matrix, or substantive inferential result.

The release intentionally does **not** ship a pre-populated `outputs/` directory. The default reproduction command creates `outputs/` from scratch. The final status is written to `outputs/verification_report.json`; a successful run reports `offline_numerical_checks_passed`. A clean run produces 11 PDF panels, 11 PNG panels and 22 CSV tables. Input files are verified by SHA-256 before and after the run; keyed numerical comparisons stop on discrepancies. Pre-release verification evidence from the authors' packaging environment is kept separately under `verification/` and is not used as input by the reproduction workflow.

### Quick start: inspect the API examples without sending requests

The API runner uses Python's standard library. Its dry run needs neither an API key nor a provider SDK.

```bash
python scripts/api/run_generation.py
python scripts/api/run_generation.py --input examples/api/relevance.jsonl
python scripts/api/run_generation.py --input examples/api/query.jsonl
```

All three commands are dry runs. They only validate the local input plan, prompts and request parameters. `notebooks/API_examples.ipynb` provides the same checks; all of its executable cells are offline.

The illustrative API input files are synthetic demonstrations of the input interface, **not the private study prompt roster and not the saved experimental outputs**. The actual saved responses are under `data/frozen_api_outputs/`.

See **[the API guide](docs/API_GUIDE.md)** for credential placeholders, explicit paid-run switches, request caps, resume behavior, complete input-plan formats and the recorded model configurations. Do not use live-generation commands to perform the ordinary numerical reproduction.

## Historical LLM API timing (UTC)

The following times were recovered from the original response logs, retained request caches, and artifact records. **Request timestamps and artifact-freeze timestamps are distinguished below.** The simulation horizons (August 2021 through August 2024) describe the modeled conditions, not when the APIs were called. File modification, ZIP-entry, and upload times are not used as evidence of API execution dates.

| Experimental stage | Evidence-supported time in UTC | Interpretation and coverage |
|---|---|---|
| M-series architecture generation, 2,000 saved outputs (`deepseek-v4-pro`) | Recorded response completions: **2026-08-13 15:35:53.634834 to 2026-08-14 03:22:05.902532** | All 2,000 output request hashes match the original API log. The log retains `completed_at_utc`; it does not retain a separate request-start field. This is a completion-time range, not a start-to-finish runtime. |
| P1 sample-size convergence, through N=200 (`deepseek-v4-pro`) | Earliest retained request start: **2026-08-15 11:57:14.629852**; latest retained completion: **2026-08-15 13:20:19.163654** | Direct `started_at_utc` and `completed_at_utc` fields from 1,600 successful retained cache records, covering 2,000 logical output rows. Incremental stages reuse earlier responses. |
| Final nine-scenario P-series outputs, 4,500 rows (`deepseek-v4-pro`) | The final output artifact was frozen at **2026-08-16 05:12:36.802805**. Reused P1 responses have recorded requests on **2026-08-15**. | The source index identifies 1,656 logical rows using 800 distinct P1 cache records and 2,844 additional distinct P2 requests. Per-request calendar times for those additional P2 requests were not recovered from the verified exports. The freeze time is **not** a last-call timestamp; it does not mean all 4,500 responses were newly generated on that date. |
| Profile-conditioned retrieval-query generation (`deepseek-v4-flash`) | The 400-profile/1,200-query output is hash-linked to a bundle manifest created at **2026-08-12 09:03:59.252107**. | This is an artifact timestamp only. The output includes 30 imported pilot query sets and 370 additional query sets; exact request dates are not established by the verified records. |
| Relevance-model comparison | June 15, 2026, approximately 03:49–09:12 (local timestamps recorded by the original notebook; timezone offset was not stored) | The six 1,000-message evaluation configurations were run on this date. The retained notebook records timestamped API-log filenames and per-request timing information. |
| Full-corpus relevance screening | June 15, 2026, approximately 11:49–17:09 (local timestamps recorded by the original notebook; timezone offset was not stored) | Qwen3.6 Flash no-thinking was used for the 325,665-message screening. The main run was followed by API repair passes for failed/invalid batches. A later manual merge at approximately 18:20 was not an API call. |

These windows describe retained responses, not every pilot request, failed attempt, retry, or billable call. Because requests can run concurrently and stages can be paused, their elapsed durations must not be summed and presented as the end-to-end experiment runtime. P3 evaluation and BGE encoding are local computations and do not add hosted generative-API calls to this timeline.

Exact values, source hashes, record coverage, and minimal timing extracts are provided in [the timing provenance note](docs/LLM_API_TIMING.md) and [the machine-readable timing evidence](docs/llm_api_timing_evidence.json). The reproduction workflow remains offline; no new API run was made to reconstruct this timeline.

## Numerical scope

| Paper result | Starting data | Recalculation |
|---|---|---|
| Figure 2(a) | Monthly counts | Annual count-weighted theme shares |
| Figure 2(b), Table 12 | Aggregate co-occurrence shares | Aggregate export and rendering; no message-level recount |
| Table 2, Figure 3, Tables 5–11 | 180-month panel | 49 fractional-logit fits, HAC uncertainty, contrasts and sensitivity specifications |
| Table 3, Figure 4 | 4,500 saved response texts and indexed frozen BGE vectors | Text-to-encoding fingerprint checks; paired semantic contrasts; 10,000 profile-block sign flips; BH correction; five-fold projections |
| Table 4(A) | Six sets of 1,000 saved predictions and per-request timing/token records | Accuracy, precision, recall, F1, confusion counts, parse success, historical mean/summed time and token totals |
| Table 4(B) | 2,250 saved gold/probability/predicted-label rows | Per-label and aggregate classifier metrics |
| Table 13 | 2,000 generated-response vectors and 5,426 reference vectors | Bidirectional Top-1/5/10 scores and same-horizon random-pair baselines |
| Table 14 | Final N=200 vectors and fixed nested cohort membership | N=50/100/200 contrasts and both convergence comparisons |

The classifier predictions correspond to `best_grid6` configuration 5, identified in the saved test-metrics record. No classifier is retrained and no threshold is retuned by the default workflow. Historical API latency is recomputed from the saved measurements, not estimated using today's endpoint.

Table 13's literal-reuse/copy-risk note requires separate message-level evidence and is not reevaluated by this command. Description-only workflows, example-appeal tables, prompt templates and scenario definitions are not statistical estimates to be refitted. Private source-record preparation and classifier training are outside this package's default reproduction entry point.

## Saved responses and local text encoding

The study's generated titles and bodies are available as ordinary CSV files:

```text
data/frozen_api_outputs/m_generated_text.csv
data/frozen_api_outputs/p1_generated_text.csv
data/frozen_api_outputs/p3_generated_text.csv
```

Each CSV is indexed by the original request ID. `validate_saved_texts.py` applies the recorded text-joining rule and checks the original corpus fingerprint. The original Parquet response records are also retained. Reading those Parquet files requires the optional decoder in `requirements-parquet-optional.txt`; the default workflow and the CSV text checks do not.

For efficiency and numerical comparability, the default workflow uses the stored BGE vectors rather than re-encoding the text. **Checking a corpus fingerprint is not equivalent to rerunning the encoder.** Optional local encoding code is provided in `scripts/encode_saved_outputs.py`; it requires the already-downloaded, matching local model snapshot, checks its directory fingerprint, and sets local-only loading. It does not download weights or contact an embedding API. Re-encoded arrays go to a new output directory, not over the frozen inputs. See [the API guide](docs/API_GUIDE.md#optional-local-re-encoding) for details.

## Outputs, conventions and data

Results are created under `outputs/` when the reproduction command is run; `outputs/` is intentionally absent from a fresh clone/release and is ignored by Git. Read-only inputs are under `data/`, and numerical comparison targets are under `reference/`. The API runner uses only `outputs/api_runs/<run-id>/`. `.env` files, new API outputs, and private prepared prompt plans are ignored by Git. `.env.example` documents placeholder names and is not automatically loaded.

P3 retains the original profile weights, five horizons, seed 20260816, 10,000 permutation draws and five-fold assignment. Inference is conditional on the saved generation run. The quadratic ITSA model uses `(time_in_months / 12)^2`; the presentation table divides that term's coefficient and uncertainty bounds by 144 to express month-squared units, without changing fitted predictions.

Figure output filenames are inherited from the original rendering code:

| Paper figure | Output filename |
|---|---|
| Figure 2(a), seven themes | `fig1a_1_...` through `fig1a_7_...` |
| Figure 2(b) | `fig1b_cooccurrence_dumbbell` |
| Figure 3 | `fig2_itsa_effects` |
| Figure 4(a) | `fig3a_information_state` |
| Figure 4(b) | `fig3b_component_ratios` |

Final plot inputs use the current recalculated values; frozen summary targets are used for verification, not substituted for the calculations. Font availability can affect plot appearance; pixel-identical figures are not required for numerical comparison.

See [DATA_MANIFEST.md](DATA_MANIFEST.md) for the data representations and source-to-output boundaries. Identifiable raw source messages and raw survey records are not included. The full private profile, memory and prompt-construction inputs must not be inferred from generated responses or replaced by the illustrative examples when describing a study rerun.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use mocked transports and do not contact providers. They check safe defaults with a configured dummy key, explicit authorization, request caps, reuse/resume, interrupted requests, rejection of malformed inputs, M3-specific settings, secret-safe errors and redirect refusal. The release verification files distinguish these tests from the numerical run. The default numerical workflow has been completed successfully on both Linux/Python 3.13.5 and Windows/Python 3.12. Live hosted-model generation and full local BGE re-encoding were not rerun as part of release verification.
