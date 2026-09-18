# Optional API execution

## 1. Choose the appropriate workflow

**Reproducing the paper:** run `python scripts/reproduce_all.py`. No generation API or key is involved. The package preserves the model responses used in the study and the indexed BGE embeddings for downstream evaluation.

**Generating new responses:** use `scripts/api/run_generation.py` with a prepared input plan, your own credential, and explicit paid-run authorization. New outputs are not expected to replay the stored responses exactly. The script is a controlled re-entry interface using recorded prompts and HTTP parameters, not a claim that today's hosted service reproduces the historical environment.

The execution interface is intentionally separated from the offline pipeline. Neither `reproduce_all.py` nor the notebook's Run All launches paid work. No global environment switch or stored key silently changes this behavior.

## 2. Inspect without credentials or network access

```bash
python scripts/api/run_generation.py --input examples/api/simulation.jsonl --dry-run
python scripts/api/run_generation.py --input examples/api/relevance.jsonl --dry-run
python scripts/api/run_generation.py --input examples/api/query.jsonl --dry-run
```

Omitting `--dry-run` has the same safe behavior unless `--execute` is explicitly supplied. Dry runs read local prompts, validate input and count unique requests; they do not construct an HTTP request, read key values, test authentication or check model availability.

The files contain four M0–M3 demonstration rows, six relevance-configuration demonstration rows and one query demonstration row. Their user inputs are illustrative synthetic examples. They are not empirical records, the study's complete prompt roster or replacements for the frozen model-output files.

## 3. Supply a key only for a voluntary live run

`.env.example` contains only placeholders:

```text
DEEPSEEK_API_KEY_SIMULATION=YOUR_API_KEY_HERE
DEEPSEEK_API_KEY=YOUR_API_KEY_HERE
DASHSCOPE_API_KEY=YOUR_API_KEY_HERE
```

The runner reads the appropriate environment variable only after paid execution is explicitly authorized. It never reads `.env` automatically, never prints a key or its prefix, and never includes credentials in a cache fingerprint. No key should be placed in source code, JSONL/CSV inputs, notebooks or Git commits.

For an interactive PowerShell session, entering the key without displaying it or writing the literal key into command history can be done as follows:

```powershell
$secureKey = Read-Host "API key" -AsSecureString
$env:DEEPSEEK_API_KEY_SIMULATION = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
```

Alternatively, the placeholder-only form below illustrates the variable name. Replace the placeholder locally; do not commit the resulting command or file:

```powershell
$env:DEEPSEEK_API_KEY_SIMULATION = "YOUR_API_KEY_HERE"
```

For a Bash session:

```bash
read -rs -p 'API key: ' DEEPSEEK_API_KEY_SIMULATION
export DEEPSEEK_API_KEY_SIMULATION
printf '\n'
```

Relevance configurations use `DEEPSEEK_API_KEY` or `DASHSCOPE_API_KEY`, as recorded in `data/api/request_profiles.json`. Merely setting any of these variables still does not authorize a call.

## 4. Explicitly authorize a small live run

**The following command is deliberately not executed in the notebook. It can incur provider charges when a real key is supplied.**

```bash
python scripts/api/run_generation.py --input examples/api/simulation.jsonl --execute --confirm I_ACCEPT_API_COSTS --max-requests 1 --run-id example_live_01
```

All of these are required: `--execute`, the exact confirmation phrase, a positive `--max-requests`, a valid key, and a new run ID. Placeholders, zero/negative caps and missing authorization fail before a network request. `--max-requests` limits **new wire attempts per invocation**, not the number of logical rows, tokens, dollars or the provider invoice. An interrupted request can still have been billed by the server.

The executor makes one HTTP attempt at a time and stops at the cap. It does not silently retry a timeout, remove an unsupported parameter and call again, follow an HTTP redirect, or generate alternative responses until one passes. Model-specific token limits remain explicit in the configuration. This bounded sequential executor differs operationally from the historical concurrent experiment runner; it retains the recorded request bodies while prioritizing deliberate cost control for new runs.

Outputs are created under:

```text
outputs/api_runs/example_live_01/
  run.json
  results.jsonl
  status.json
  events.jsonl       # present if an error/event is recorded
  cache/
```

These files are ignored by Git. They may contain generated text, so review them before any further release. They never overwrite `data/`, `reference/` or the published numerical results.

## 5. Resume, caching and failures

A second invocation with the same run ID is refused unless `--resume` is specified. Inputs, prompts, configurations and the stored plan fingerprint must still match.

```bash
python scripts/api/run_generation.py --input examples/api/simulation.jsonl --execute --confirm I_ACCEPT_API_COSTS --max-requests 1 --run-id example_live_01 --resume
```

The cap applies afresh to this invocation. Completed rows are skipped, and identical request bodies reuse the stored validated response. Separate logical rows can share one cached response. Cache identity includes the endpoint, request body and response schema, but no credential.

A timeout or interrupted attempt is recorded before any later retry. Resume does not silently repeat it. After reviewing the error and accepting that it may already have been billed, a user can add `--retry-failed` together with `--resume` to explicitly allow another attempt; it still counts against the cap. Content-invalid responses are retained as rejected final visible text and are not resampled on resume. No fallback text is fabricated.

If the process is terminated without normal cleanup, `.lock` may remain. Confirm the prior process has stopped before removing that lock manually. Do not delete the cache or attempt records to bypass an uncertain request; preserve them for inspection.

Errors record an exception type and HTTP status where available, not the provider's raw error body or key. The saved completion retains visible content, parsed output and usage metadata. Hosted hidden reasoning text is not retained.

## 6. Prepared inputs for larger or new experiments

The runner accepts UTF-8 CSV or JSONL. Each row requires:

| Field | Meaning |
|---|---|
| `record_id` | Unique logical-row ID; `request_row_id` or `sample_id` is also accepted |
| `profile` | A request configuration from `request_profiles.json`; simulation `architecture_id` M0–M3 can be used instead |
| `user_prompt` | The complete prepared user message; blank, null and `nan` prompts are rejected |
| `allowed_concerns` | Required for query generation; JSON list of allowed axis codes |

Example structure:

```json
{"record_id":"example_M1","profile":"M1","user_prompt":"Complete authorized prompt text goes here."}
```

For P-series generation use profile `M1` and provide the already prepared prompts for the intended policy scenarios. For a full study-style rerun, retain the original cohort membership, profile/time pairing, scenario definitions, generalized profile fields, history-selection rules and approved prompt text. The executor does not reconstruct private survey data, historical messages or household states from generated output text.

The helper can project a prepared original-format prompt table to this interface, without calling an API:

```bash
python scripts/api/prepare_input.py --input private_inputs/prompt_plan.csv --name prepared_plan
python scripts/api/run_generation.py --input outputs/prepared_api/prepared_plan.jsonl --dry-run
```

CSV rows can supply `architecture_id`. Use `--profile M1` to fix the request profile, or `--cohort-column in_N100` when the prepared table contains a boolean membership column. Parquet conversion additionally needs `requirements-parquet-optional.txt`.

The complete original private request plans are not distributed here. Readers wishing to regenerate responses must supply an authorized prepared plan or use the clearly labeled demonstration inputs. This does not affect reproduction of the reported downstream results from the released saved outputs and numerical intermediates.

## 7. Recorded settings and implementation boundaries

`data/api/request_profiles.json` preserves the settings read from the supplied experiment sources; it is not a catalogue of currently available models.

| Profile | Recorded request behavior |
|---|---|
| M0–M2 | `deepseek-v4-pro`, thinking enabled/high, JSON output, max 4096 tokens; no temperature/top-p sent |
| M3 | Same backbone/thinking mode, separate system template and visible reflection/planning schema, max 6144 tokens |
| Query v8 | `deepseek-v4-flash`, thinking disabled, temperature 0, top-p 1, max 1024 tokens |
| DeepSeek relevance, thinking/default | Original model alias, temperature 0, max 800 tokens; no explicit thinking switch in that notebook configuration |
| DeepSeek relevance, no-thinking | Same model alias, thinking disabled, temperature 0, max 500 tokens |
| Qwen/GLM relevance, no-thinking | Original aliases and recorded compatible endpoint, `enable_thinking=false`, temperature 0, max 500 tokens |

The three request families retain their supplied system prompt text. The query input generator is represented by its prepared user-template interface; this example executor checks allowed axes and JSON structure rather than claiming to reproduce all upstream field-derivation and manual audit steps. Simulation output checks cover the architecture-specific fields, length limits and the direct identifier/meta-language/salutation checks used by the supplied runner. Automated checks do not establish factual validity of a new response.

The source fingerprints are in `API_SOURCE_NOTES.json`. Recorded provider model IDs and service behavior were not validated with live requests during packaging. If a provider no longer supports an alias or parameter, review its current documentation and record a separate configuration for the new experiment; do not silently substitute another model and label its output as the original run.

## Optional local re-encoding

The default workflow verifies the saved generated text and uses its frozen encodings. To examine text-to-vector encoding explicitly, supply the local `BAAI/bge-large-zh-v1.5` snapshot corresponding to revision `d7b4914a04f195eb4c1520d297963449e8720e95` and install the optional local-encoding dependencies:

```bash
python -m pip install -r requirements-encoding-optional.txt
python scripts/encode_saved_outputs.py --encode --dataset p3 --model-dir private_inputs/bge_snapshot --run-id p3_encoding_check
```

The script verifies the model directory fingerprint recorded in the original measurement contract, then loads with `local_files_only=True` and remote code disabled. It writes new vectors and a comparison report under `outputs/local_encoding/`. It does not download model files or replace the frozen arrays. Dependency versions and hardware can affect newly computed floating-point values. No claim of a new, fully executed encoder run is made for this release: the complete model snapshot was not present in the packaging environment. CSV text selection, joining and corpus-fingerprint checks were executed.

Implementation references: Python's `urllib.request` redirect-handler documentation and Sentence Transformers' `SentenceTransformer` constructor/encoding documentation. These support the transport/local-loading interface; the scientific request settings above come from the supplied experiment sources.
