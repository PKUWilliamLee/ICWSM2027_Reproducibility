"""Optional hosted-model calls. The default is a credential-free, network-free dry run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from request_io import ROOT, PROFILES, canonical, digest, prepare_plan, preview, parse_completion

COST_CONFIRMATION = "I_ACCEPT_API_COSTS"
RUNS = ROOT / "outputs/api_runs"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class RunStopped(RuntimeError):
    """An execution was stopped without exposing request text or credentials."""


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical(value) + b"\n")
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("ab") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def credential(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value or re.search(r"YOUR.*KEY|PLACEHOLDER|REPLACE.*KEY|<.*>", value, re.I):
        raise RunStopped(f"Set a real credential in {env_name} before requesting a paid run")
    if any(ch.isspace() for ch in value):
        raise RunStopped("Credential contains whitespace; no request has been sent for this row")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def http_once(url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
    """One wire attempt; no redirects, automatic retries, or parameter fallback."""
    req = urllib.request.Request(url, data=canonical(body), method="POST",
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds local size limit")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Response is not a JSON object")
        return payload


def safe_error(exc: BaseException) -> dict[str, Any]:
    # Provider exception text can contain request bodies, keys or private text.
    record: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        record["http_status"] = int(exc.code)
    return record


def execute_plan(plan: list[dict[str, Any]], *, execute: bool = False,
                 confirmation: str = "", max_requests: int | None = None,
                 run_id: str | None = None, resume: bool = False,
                 retry_failed: bool = False,
                 transport: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run a bounded prepared plan, or return a dry-run summary.

    Merely configuring a key or importing this module cannot authorize live calls.
    Failed/in-flight attempts require a further explicit retry opt-in. Content-invalid
    responses are never automatically regenerated.
    """
    if not execute:
        return preview(plan)
    if confirmation != COST_CONFIRMATION:
        raise RunStopped("Paid execution requires --confirm I_ACCEPT_API_COSTS")
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
        raise RunStopped("Paid execution requires a positive --max-requests HTTP-attempt cap")
    if not run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise RunStopped("Choose a run-id using letters, digits, dots, underscores or hyphens")
    if retry_failed and not resume:
        raise RunStopped("--retry-failed must be used with --resume")
    run = RUNS / run_id
    if not run.resolve().is_relative_to(RUNS.resolve()):
        raise RunStopped("Run directory is outside outputs/api_runs")
    # Check the first key before creating a paid-run directory.
    credential(plan[0]["api_key_env"])
    RUNS.mkdir(parents=True, exist_ok=True)
    if resume:
        if not run.is_dir():
            raise RunStopped("Cannot resume a run that does not exist")
    else:
        try:
            run.mkdir(exist_ok=False)
        except FileExistsError:
            raise RunStopped("This run-id already exists. Use a new ID or explicitly --resume") from None
    lock = run / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RunStopped("Run is locked. Do not start a second process; see interrupted-run guidance") from None
    os.close(fd)
    calls = 0
    try:
        manifest = {"plan_sha256": digest(plan), "schema": "safe-api-run-v1"}
        meta = run / "run.json"
        if resume:
            if not meta.is_file() or json.loads(meta.read_text()) != manifest:
                raise RunStopped("Prepared inputs, prompts or configuration differ from this run")
        else:
            write_json(meta, manifest)
        cache = run / "cache"
        cache.mkdir(exist_ok=True)
        events = run / "events.jsonl"
        results = run / "results.jsonl"
        completed: set[str] = set()
        records_by_id = {x["record_id"]: x for x in plan}
        if results.exists():
            try:
                for line in results.read_text(encoding="utf-8").splitlines():
                    entry = json.loads(line)
                    rid = entry["record_id"]
                    if rid in completed or rid not in records_by_id or entry["request_hash"] != records_by_id[rid]["request_hash"]:
                        raise ValueError("Invalid result index")
                    # Verify previously completed cache instead of blindly trusting a results file.
                    cpath = cache / (entry["request_hash"] + ".json")
                    c = json.loads(cpath.read_text())
                    if c["integrity"] != digest(c["record"]) or c["record"]["response"] != entry["response"]:
                        raise ValueError("Changed cache/result")
                    completed.add(rid)
            except (ValueError, KeyError, OSError):
                raise RunStopped("Stored results are corrupt or incomplete; preserve the directory for inspection") from None
        send = transport or http_once
        for row in plan:
            if row["record_id"] in completed:
                continue
            h = row["request_hash"]
            cpath = cache / (h + ".json")
            attempt_path = cache / (h + ".attempt.json")
            cached = cpath.is_file()
            if cached:
                envelope = json.loads(cpath.read_text(encoding="utf-8"))
                saved = envelope["record"]
                if envelope.get("integrity") != digest(saved) or saved.get("request_hash") != h:
                    raise RunStopped("Cached result failed integrity checking")
            else:
                if calls >= max_requests:
                    break
                if attempt_path.exists():
                    previous = json.loads(attempt_path.read_text())
                    if previous.get("status") == "content_invalid":
                        raise RunStopped("A content-invalid response exists. It will not be resampled on resume")
                    if not retry_failed:
                        raise RunStopped("A failed or interrupted attempt may already have been billed; --retry-failed is required to attempt it again")
                key = credential(row["api_key_env"])
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                marker = {"status": "in_flight", "request_hash": h, "started_at_utc": started}
                write_json(attempt_path, marker)
                calls += 1
                t0 = time.perf_counter()
                try:
                    raw = send(row["request"]["url"], row["request"]["body"], key, row["timeout_seconds"])
                except Exception as exc:
                    marker.update({"status": "transport_or_response_error", "wire_attempts_this_invocation": calls, **safe_error(exc)})
                    write_json(attempt_path, marker)
                    append_jsonl(events, marker)
                    raise RunStopped("HTTP attempt failed; execution stopped. It may have been billed. No automatic retry or fallback was made") from None
                finally:
                    key = ""
                elapsed = time.perf_counter() - t0
                try:
                    response = parse_completion(raw, row["request"])
                except Exception as exc:
                    marker.update({"status": "content_invalid", **safe_error(exc)})
                    write_json(attempt_path, marker)
                    append_jsonl(events, marker)
                    # Keep final visible content privately, not hidden reasoning or full HTTP responses.
                    choices = raw.get("choices") or []
                    visible = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], dict) else None
                    write_json(cache / (h + ".rejected.json"), {"request_hash": h, "content": visible, **safe_error(exc)})
                    raise RunStopped("Completion failed output checks. The final visible content was retained privately; no replacement was generated") from None
                saved = {"request_hash": h, "response": response, "started_at_utc": started,
                         "elapsed_sec": elapsed, "endpoint": row["request"]["url"]}
                write_json(cpath, {"record": saved, "integrity": digest(saved)})
                write_json(attempt_path, {**marker, "status": "completed"})
            append_jsonl(results, {"record_id": row["record_id"], "request_hash": h,
                                   "profile": row["profile"], "cache_hit": cached,
                                   "response": saved["response"], "elapsed_sec": saved["elapsed_sec"]})
            completed.add(row["record_id"])
        report = {"mode": "paid_execution", "wire_attempts_this_invocation": calls,
                  "logical_rows_complete": len(completed), "logical_rows_total": len(plan),
                  "status": "complete" if len(completed) == len(plan) else "paused_at_request_cap",
                  "run_directory": "outputs/api_runs/" + run_id}
        write_json(run / "status.json", report)
        return report
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/api/simulation.jsonl")
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate locally (the default)")
    mode.add_argument("--execute", action="store_true", help="Request explicitly authorized paid HTTP calls")
    parser.add_argument("--confirm", default="", help="Must be I_ACCEPT_API_COSTS for --execute")
    parser.add_argument("--max-requests", type=int, help="Required positive cap on new HTTP attempts per invocation")
    parser.add_argument("--run-id", help="New output directory name under outputs/api_runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly reattempt prior failed/uncertain wire requests")
    args = parser.parse_args(argv)
    try:
        plan = prepare_plan(args.input, args.profiles)
        result = execute_plan(plan, execute=args.execute, confirmation=args.confirm,
                              max_requests=args.max_requests, run_id=args.run_id,
                              resume=args.resume, retry_failed=args.retry_failed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except RunStopped as exc:
        print("STOPPED: " + str(exc), file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError, TypeError):
        print("Input/configuration validation failed. Check paths, record IDs, user_prompt text and the documented schema. No input text is printed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
