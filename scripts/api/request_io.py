"""Prepare recorded Chat Completions requests without creating clients or network traffic."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "data/api/request_profiles.json"
SCHEMA_VERSION = "review-api-examples-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"nan", "none", "null"}:
        raise ValueError(f"{field} must be a non-empty text value")
    return value


def check_no_credentials(value: Any) -> None:
    """Reject credential-bearing fields; credentials belong only in environment variables."""
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in {"api_key", "authorization", "access_token", "secret", "password", "headers"}:
                raise ValueError("Credential-bearing fields are not accepted in request files")
            check_no_credentials(v)
    elif isinstance(value, list):
        for v in value:
            check_no_credentials(v)
    elif isinstance(value, str) and re.search(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b", value):
        raise ValueError("A possible credential was found in request text; remove it before use")


def endpoint(base_url: str) -> str:
    u = urlsplit(base_url)
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError("base_url must be an HTTPS service URL without embedded credentials, query or fragment")
    return base_url.rstrip("/") + "/chat/completions"


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("The request input file does not exist")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raise ValueError("Prepared API inputs must be CSV or JSONL; see docs/API_GUIDE.md")
    if not records:
        raise ValueError("The input plan is empty")
    return records


def prepare_plan(path: Path, profiles_path: Path = PROFILES) -> list[dict[str, Any]]:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    check_no_credentials(profiles)
    plan, seen = [], set()
    for number, record in enumerate(read_records(path), 1):
        if not isinstance(record, dict):
            raise ValueError(f"Input row {number} is not an object")
        check_no_credentials(record)
        rid = record.get("record_id", record.get("request_row_id", record.get("sample_id")))
        rid = text(rid, f"Row {number} record_id")
        if rid in seen:
            raise ValueError("Duplicate record_id in input plan")
        seen.add(rid)
        name = record.get("profile") or record.get("architecture_id")
        if name not in profiles:
            raise ValueError(f"Row {number} has no recognized request profile")
        cfg = profiles[name]
        user = text(record.get("user_prompt"), f"Row {number} user_prompt")
        prompt_path = (ROOT / cfg["system_file"]).resolve()
        if not prompt_path.is_relative_to(ROOT / "data/prompts"):
            raise ValueError("System prompts must be stored in data/prompts")
        system = prompt_path.read_text(encoding="utf-8")
        if cfg["task"] == "simulation":
            if not system.startswith("SYSTEM\n") or "\nUSER\n" not in system:
                raise ValueError("Simulation template lacks SYSTEM / USER separators")
            system = system.removeprefix("SYSTEM\n").split("\nUSER\n", 1)[0].strip()
        body = dict(cfg["body"])
        if "messages" in body or "extra_body" in body:
            raise ValueError("Profile body must contain HTTP parameters, not messages or SDK wrappers")
        if body.get("stream") is not False or not isinstance(body.get("max_tokens"), int) or body["max_tokens"] <= 0:
            raise ValueError("A positive token limit and stream=false are required")
        text(body.get("model"), "model")
        body["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        allowed = record.get("allowed_concerns", [])
        if isinstance(allowed, str):
            allowed = json.loads(allowed) if allowed.strip() else []
        if cfg["task"] == "query" and (not isinstance(allowed, list) or not allowed or any(not isinstance(a, str) or not a for a in allowed)):
            raise ValueError("Query rows require an explicit list of allowed_concerns")
        env = cfg["api_key_env"]
        if not isinstance(env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", env):
            raise ValueError("Invalid credential environment-variable name")
        timeout = float(cfg.get("timeout_seconds", 300))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid HTTP timeout")
        request = {"task": cfg["task"], "architecture_id": cfg.get("architecture_id"),
                   "url": endpoint(cfg["base_url"]), "body": body,
                   "schema_version": SCHEMA_VERSION, "allowed_concerns": allowed}
        plan.append({"record_id": rid, "profile": name, "request": request,
                     "request_hash": digest(request), "api_key_env": env,
                     "timeout_seconds": timeout})
    return plan


def preview(plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {"mode": "dry_run", "network_calls": 0, "logical_rows": len(plan),
            "unique_requests": len({r["request_hash"] for r in plan}),
            "request_profiles": sorted({r["profile"] for r in plan}),
            "credential_environment_names": sorted({r["api_key_env"] for r in plan}),
            "plan_sha256": digest(plan),
            "message": "Prepared locally. No credentials were read; no HTTP request was sent."}


def normalized(s: Any) -> str:
    if not isinstance(s, str):
        raise ValueError("Expected a response string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def string_list(v: Any, maximum_items: int, maximum_chars: int) -> list[str]:
    if not isinstance(v, list) or len(v) > maximum_items:
        raise ValueError("Invalid response list")
    result = [normalized(x) for x in v]
    if any(not x or len(x) > maximum_chars for x in result):
        raise ValueError("Invalid response list item")
    return result


def validate_payload(payload: Any, request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object response")
    task = request["task"]
    if task == "simulation":
        m3 = request["architecture_id"] == "M3"
        keys = {"title", "content"} | ({"reflection", "planning"} if m3 else set())
        if set(payload) != keys:
            raise ValueError("Simulation output keys do not match the selected architecture")
        out = {k: normalized(payload[k]) for k in ["title", "content"]}
        if not (1 <= len(out["title"]) <= 50 and 20 <= len(out["content"]) <= 600):
            raise ValueError("Simulation title or content length is outside the recorded limits")
        if m3:
            rfields = {"persistent_constraints": 4, "salient_concerns": 3,
                       "current_policy_tensions": 3, "unresolved_uncertainty": 3}
            if not isinstance(payload["reflection"], dict) or set(payload["reflection"]) != set(rfields):
                raise ValueError("Invalid M3 reflection fields")
            out["reflection"] = {k: string_list(payload["reflection"][k], n, 30) for k, n in rfields.items()}
            planning = payload["planning"]
            if not isinstance(planning, dict) or set(planning) != {"focal_problem", "requested_action", "supporting_facts", "prohibited_claims"}:
                raise ValueError("Invalid M3 planning fields")
            out["planning"] = {k: normalized(planning[k]) for k in ["focal_problem", "requested_action"]}
            if any(not x or len(x) > 120 for x in out["planning"].values()):
                raise ValueError("Invalid M3 planning text")
            out["planning"].update({k: string_list(planning[k], 3, 40) for k in ["supporting_facts", "prohibited_claims"]})
        final = out["title"] + "\n" + out["content"]
        all_text = json.dumps(out, ensure_ascii=False)
        pii = [r"(?<!\d)1[3-9]\d{9}(?!\d)", r"(?<!\d)\d{17}[\dXx](?!\d)",
               r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", r"https?://\S+|www\.\S+", r"(?<!\d)\d{15,19}(?!\d)"]
        if any(re.search(p, all_text, re.I) for p in pii):
            raise ValueError("A direct identifier pattern was detected in the response")
        if re.search(r"智能体|模拟|提示词|主题|分类器|情景|ITSA|RoBERTa|Topic[1-7]", final, re.I):
            raise ValueError("The response contains disallowed study-meta language")
        salutation = r"尊敬的(?:学校|校方|校)领导|(?:学校|校方|校)领导(?:您好|好)|校长(?:您好|好)|尊敬的(?:老师|教师)|(?:老师|教师)(?:您好|好)|尊敬的(?:培训机构|机构)负责人|(?:培训机构|机构)负责人(?:您好|好)"
        if re.search(salutation, final, re.I) or (m3 and re.search(salutation, out['planning']['requested_action'], re.I)):
            raise ValueError("The response is addressed directly to a non-government recipient")
        return out
    if task == "relevance":
        label = payload.get("label", payload.get("relevance_label", payload.get("pred_label")))
        if isinstance(label, str):
            label = {"相关": 1, "是": 1, "yes": 1, "1": 1, "不相关": 0, "否": 0, "no": 0, "0": 0}.get(label.strip().lower())
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("The relevance label is not binary")
        conf = payload.get("confidence", payload.get("conf"))
        if conf is not None:
            conf = float(conf)
            if not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError("Invalid confidence value")
        reason = payload.get("reason", payload.get("rationale", "")) or ""
        if not isinstance(reason, str):
            raise ValueError("Invalid relevance reason")
        return {"pred_label": int(label), "confidence": conf, "reason": reason.replace("\n", " ").replace("\r", " ")[:200]}
    if task == "query":
        queries = payload.get("queries")
        if not isinstance(queries, list) or not 1 <= len(queries) <= 3:
            raise ValueError("Query output must contain one to three entries")
        seen = set()
        for q in queries:
            if not isinstance(q, dict) or q.get("concern") not in request["allowed_concerns"]:
                raise ValueError("Query concern is not in allowed_concerns")
            text(q.get("query_text"), "query_text")
            if q["concern"] in seen:
                raise ValueError("Duplicate concern in query output")
            seen.add(q["concern"])
        return payload
    raise ValueError("Unknown response task")


def parse_completion(raw: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected one completion choice")
    choice = choices[0]
    if choice.get("finish_reason") not in (None, "stop"):
        raise ValueError("Completion did not finish normally; automatic regeneration is disabled")
    content = text(choice.get("message", {}).get("content"), "completion content")
    candidate = content
    if request["task"] == "relevance":
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.S)
            if match is None:
                raise ValueError("Response has no parseable JSON object") from None
            payload = json.loads(match.group(0))
    else:
        payload = json.loads(candidate)
    parsed = validate_payload(payload, request)
    # Do not retain hosted hidden-reasoning text or HTTP headers.
    return {"content": content, "parsed": parsed,
            "returned_model": raw.get("model"), "response_id": raw.get("id"),
            "system_fingerprint": raw.get("system_fingerprint"),
            "usage": raw.get("usage"), "finish_reason": choice.get("finish_reason")}
