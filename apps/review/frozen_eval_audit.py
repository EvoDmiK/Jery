"""Independent leakage and label review auditor for frozen-eval-v1 candidate packets and manifests.

Python 3.9+ standard library only.
Audits frozen-eval candidate/frozen packets (cases.jsonl, manifest.json, and optional
AI independent proposals JSON) against the pinned ten-tool catalog and all historical
datasets (hermes-routing-v1, mcp-routing-v2, hermes-ten-tools-contrast-source-v1, and
diagnostics probes) without mutating any raw or candidate files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import unicodedata

try:
    if __package__:
        from .audit import validate_output_summary_path
    else:
        from audit import validate_output_summary_path
except ImportError:
    from apps.review.audit import validate_output_summary_path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = DEFAULT_ROOT / "datasets/hermes-ten-tools-v2/tools.json"
DEFAULT_HISTORICAL_PATHS: Tuple[Path, ...] = (
    DEFAULT_ROOT / "datasets/hermes-routing-v1/train.jsonl",
    DEFAULT_ROOT / "datasets/hermes-routing-v1/validation.jsonl",
    DEFAULT_ROOT / "datasets/hermes-routing-v1/test.jsonl",
    DEFAULT_ROOT / "datasets/mcp-routing-v2/train.jsonl",
    DEFAULT_ROOT / "datasets/mcp-routing-v2/validation.jsonl",
    DEFAULT_ROOT / "datasets/mcp-routing-v2/test.jsonl",
    DEFAULT_ROOT / "datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl",
    DEFAULT_ROOT / "datasets/diagnostics/laya-ko-en-request-v1.json",
)

EXPECTED_SCHEMA_VERSION = "frozen-eval-v1"
VALID_PACKET_STATUSES = ("candidate", "frozen")
VALID_DECISIONS = ("select_tools", "no_tool", "clarify", "unsupported")
VALID_REVIEW_STATUSES = ("needs_human_review", "human_reviewed", "excluded", "needs_review")
VALID_SOURCE_KINDS = ("real", "synthetic")
VALID_ACTOR_KINDS = ("human_user", "human", "user", "cron", "worker", "system", "agent", "automated")
NON_HUMAN_ACTOR_KINDS = ("cron", "worker", "system", "agent", "automated")
TARGET_TEN_TOOLS_TOTAL = 100
TARGET_PER_TOOL = 10

AUTOMATED_REQUEST_PATTERNS = (
    re.compile(r"(?i)\b(?:orca\s+orchestration|worker_done|heartbeat|--dispatch-id|--task-id)\b"),
    re.compile(r"(?i)^\[?(?:cron|system|scheduled|heartbeat|auto-trigger)\b"),
    re.compile(r"(?i)\b(?:You are working inside Orca|You are a dispatched worker)\b"),
)

UNSANITIZED_PII_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----")),
    ("bearer_or_api_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]+|\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_\-]+")),
    ("credential_assignment", re.compile(r"(?im)(?:authorization|cookie|password|passwd|secret|token|credentials?|(?:api|access|refresh|auth)[_-]?(?:key|token))\s*[\"\x27]?\s*[:=]\s*[^\s\[][^\n]*")),
    ("raw_email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("raw_filesystem_path", re.compile(r"(?:[A-Za-z]:\\|(?<!\w)/(?:Users|Volumes|home|var|tmp|etc|opt)/)[^\s<>\"\x27,;]+")),
)

POST_CALL_CONTEXT_KEYS = {
    "tool_calls",
    "tool_call_id",
    "observed_choice",
    "observed_choices",
    "outcome",
    "execution_result",
    "post_call_result",
}


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def dump_canonical(value: Any) -> str:
    """Deterministic JSON serialization for hashing and signatures."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text_signature(text: str) -> str:
    """Compute normalized alphanumeric casefolded NFKC signature for exact/normalized duplicate checks."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in norm if ch.isalnum())


def normalize_state_signature(request: str, context: Sequence[Dict[str, Any]]) -> str:
    """Compute normalized signature of model input state (request + pre-decision context)."""
    canonical_ctx = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in context
        if isinstance(m, dict)
    ]
    return normalize_text_signature(dump_canonical({"request": request, "context": canonical_ctx}))


def tokenize_for_similarity(text: str) -> Set[str]:
    """Extract normalized word and character 3-gram tokens for near-duplicate detection."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    words = set(re.findall(r"[a-z0-9가-힣]+", norm))
    compact = "".join(ch for ch in norm if ch.isalnum())
    char_ngrams = {compact[i : i + 3] for i in range(max(0, len(compact) - 2))} if len(compact) >= 3 else set()
    return words | char_ngrams


def load_catalog_spec(catalog_path: Path) -> Dict[str, Any]:
    """Load and validate the pinned 10-tool catalog."""
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing catalog file: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools_list = catalog.get("tools", [])
    if not isinstance(tools_list, list) or not tools_list:
        raise ValueError(f"Invalid or empty tools list in catalog: {catalog_path}")
    tools_by_id: Dict[str, Dict[str, Any]] = {}
    action_enums: Dict[str, Optional[Set[str]]] = {}
    action_required: Dict[str, bool] = {}
    for tool in tools_list:
        tid = tool.get("id")
        if not isinstance(tid, str) or not tid:
            raise ValueError(f"Invalid tool id in catalog: {tool}")
        tools_by_id[tid] = tool
        schema = tool.get("input_schema", {}) if isinstance(tool.get("input_schema"), dict) else {}
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        act_prop = props.get("action")
        if isinstance(act_prop, dict) and isinstance(act_prop.get("enum"), list):
            action_enums[tid] = set(act_prop["enum"])
        else:
            action_enums[tid] = None
        req = schema.get("required", [])
        action_required[tid] = isinstance(req, list) and ("action" in req)

    return {
        "path": str(catalog_path),
        "sha256": compute_sha256(catalog_path),
        "catalog_version": catalog.get("catalog_version"),
        "source_catalog_version": catalog.get("source_catalog_version"),
        "tool_ids": list(tools_by_id.keys()),
        "tools_by_id": tools_by_id,
        "action_enums": action_enums,
        "action_required": action_required,
    }


def load_historical_corpus(paths: Sequence[Path]) -> Dict[str, Any]:
    """Load all historical train/validation/test/contrast/diagnostic rows for leakage checking."""
    ids: Dict[str, List[str]] = {}
    scenarios: Dict[str, List[str]] = {}
    req_signatures: Dict[str, List[Dict[str, str]]] = {}
    state_signatures: Dict[str, List[Dict[str, str]]] = {}
    tokenized_requests: List[Dict[str, Any]] = []
    file_fingerprints: Dict[str, str] = {}
    total_records = 0

    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing required historical dataset file for leakage audit: {path}")
        rel_name = str(path)
        try:
            rel_name = str(path.resolve().relative_to(DEFAULT_ROOT.resolve()))
        except ValueError:
            pass
        file_fingerprints[rel_name] = compute_sha256(path)

        if path.suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                total_records += 1
                rid = str(row.get("id") or row.get("source_id") or f"{rel_name}:{line_no}")
                sid = str(row.get("scenario_id") or row.get("session_group") or "")
                split = str(row.get("split") or row.get("source_split") or "unknown")
                if "state" in row and isinstance(row["state"], dict):
                    req = str(row["state"].get("request", ""))
                    ctx = row["state"].get("context", [])
                else:
                    req = str(row.get("request", ""))
                    ctx = row.get("context", [])
                if not isinstance(ctx, list):
                    ctx = []

                ids.setdefault(rid, []).append(f"{rel_name}:{split}")
                if sid:
                    scenarios.setdefault(sid, []).append(f"{rel_name}:{split}")

                r_sig = normalize_text_signature(req)
                s_sig = normalize_state_signature(req, ctx)
                meta_ref = {"file": rel_name, "split": split, "id": rid, "scenario_id": sid, "request": req}
                if r_sig:
                    req_signatures.setdefault(r_sig, []).append(meta_ref)
                if s_sig:
                    state_signatures.setdefault(s_sig, []).append(meta_ref)
                toks = tokenize_for_similarity(req)
                if toks:
                    tokenized_requests.append({
                        "tokens": toks,
                        "req_sig": r_sig,
                        "meta": meta_ref,
                    })
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
            for idx, item in enumerate(pairs):
                if not isinstance(item, dict):
                    continue
                total_records += 1
                rid = str(item.get("source_id") or f"{rel_name}:{idx}")
                req = str(item.get("english_request") or item.get("request") or "")
                ids.setdefault(rid, []).append(f"{rel_name}:diagnostic")
                r_sig = normalize_text_signature(req)
                meta_ref = {"file": rel_name, "split": "diagnostic", "id": rid, "scenario_id": "", "request": req}
                if r_sig:
                    req_signatures.setdefault(r_sig, []).append(meta_ref)
                toks = tokenize_for_similarity(req)
                if toks:
                    tokenized_requests.append({
                        "tokens": toks,
                        "req_sig": r_sig,
                        "meta": meta_ref,
                    })

    dedup_tokenized: List[Dict[str, Any]] = []
    seen_sigs: Set[str] = set()
    for entry in tokenized_requests:
        sig = entry["req_sig"]
        if sig and sig not in seen_sigs:
            seen_sigs.add(sig)
            dedup_tokenized.append(entry)

    corpus_fingerprint = hashlib.sha256(dump_canonical(file_fingerprints).encode("utf-8")).hexdigest()
    return {
        "file_fingerprints": file_fingerprints,
        "corpus_fingerprint": corpus_fingerprint,
        "total_records": total_records,
        "ids": ids,
        "scenarios": scenarios,
        "req_signatures": req_signatures,
        "state_signatures": state_signatures,
        "tokenized_requests": dedup_tokenized,
    }


def _check_choice_against_catalog(
    choice: Any,
    candidate_tools: Sequence[str],
    catalog_spec: Dict[str, Any],
) -> List[str]:
    """Validate a single {tool, action} choice entry against candidate_tools and catalog schema."""
    errors: List[str] = []
    if not isinstance(choice, dict):
        return ["choice must be an object"]
    extra_keys = set(choice.keys()) - {"tool", "action"}
    if extra_keys:
        errors.append(f"choice has unexpected keys: {sorted(extra_keys)}")
    tool = choice.get("tool")
    if not isinstance(tool, str) or not tool:
        errors.append("choice missing non-empty string 'tool'")
        return errors
    if tool not in catalog_spec["tools_by_id"]:
        errors.append(f"choice tool '{tool}' not in pinned catalog")
        return errors
    if tool not in candidate_tools:
        errors.append(f"choice tool '{tool}' not in row candidate_tools")

    action = choice.get("action")
    enum_set = catalog_spec["action_enums"].get(tool)
    is_req = catalog_spec["action_required"].get(tool, False)
    if enum_set is not None:
        if action is None:
            if is_req:
                errors.append(f"tool '{tool}' requires an action from {sorted(enum_set)}, got None")
        elif not isinstance(action, str) or action not in enum_set:
            errors.append(f"tool '{tool}' action '{action}' not in catalog enum {sorted(enum_set)}")
    else:
        if action is not None:
            errors.append(f"tool '{tool}' has no action property in catalog schema, but got action='{action}'")
    return errors


def _scan_privacy_and_feature_hygiene(
    row_id: str,
    request: str,
    context: Sequence[Any],
    catalog_tool_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Check pre-decision feature boundary and unsanitized PII/secret patterns."""
    boundary_violations: List[Dict[str, Any]] = []
    privacy_warnings: List[Dict[str, Any]] = []

    if not isinstance(request, str) or not request.strip():
        boundary_violations.append({"id": row_id, "issue": "empty_or_non_string_request"})
        request_str = ""
    else:
        request_str = request

    texts_to_scan: List[Tuple[str, str]] = [("request", request_str)]
    if not isinstance(context, list):
        boundary_violations.append({"id": row_id, "issue": "context_not_a_list"})
    else:
        for idx, msg in enumerate(context):
            if not isinstance(msg, dict):
                boundary_violations.append({"id": row_id, "issue": f"context[{idx}]_not_an_object"})
                continue
            leaked_keys = set(msg.keys()) & POST_CALL_CONTEXT_KEYS
            if leaked_keys:
                boundary_violations.append({
                    "id": row_id,
                    "issue": f"context[{idx}]_contains_post_call_fields:{sorted(leaked_keys)}",
                })
            extra_ctx_keys = set(msg.keys()) - {"role", "content"}
            if extra_ctx_keys:
                boundary_violations.append({
                    "id": row_id,
                    "issue": f"context[{idx}]_contains_non_feature_keys:{sorted(extra_ctx_keys)}",
                })
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant", "tool"):
                boundary_violations.append({
                    "id": row_id,
                    "issue": f"context[{idx}]_invalid_role:{role}",
                })
            if not isinstance(content, str):
                boundary_violations.append({
                    "id": row_id,
                    "issue": f"context[{idx}]_non_string_content",
                })
            else:
                texts_to_scan.append((f"context[{idx}]", content))

    for field_name, text in texts_to_scan:
        for tid in catalog_tool_ids:
            if tid in text:
                boundary_violations.append({
                    "id": row_id,
                    "issue": f"tool_id_leaked_in_{field_name}:{tid}",
                })
        for pat_name, pat in UNSANITIZED_PII_PATTERNS:
            if pat.search(text):
                privacy_warnings.append({
                    "id": row_id,
                    "field": field_name,
                    "pattern": pat_name,
                })

    return boundary_violations, privacy_warnings


def audit_ai_proposals(
    proposals_path: Optional[Path],
    case_rows_by_id: Dict[str, Dict[str, Any]],
    catalog_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit optional independent AI label proposals JSON."""
    if proposals_path is None:
        return {
            "present": False,
            "path": None,
            "sha256": None,
            "proposal_count": 0,
            "violations": [],
            "proposer_counts": {},
            "disposition_counts": {},
            "confidence_counts": {},
            "agreement_summary": {
                "cases_with_proposals": 0,
                "unanimous_select_tools_count": 0,
                "disputed_or_ambiguous_count": 0,
            },
        }

    proposals_path = Path(proposals_path)
    if not proposals_path.exists():
        raise FileNotFoundError(f"Missing AI proposals file: {proposals_path}")

    raw = json.loads(proposals_path.read_text(encoding="utf-8"))
    violations: List[str] = []
    proposer_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()

    # Support either {"proposals": {...}} or direct keyed dict {"case_id": {... or [...]}}
    if isinstance(raw, dict) and "proposals" in raw and isinstance(raw["proposals"], dict):
        top_review = raw.get("review_status") or raw.get("human_review_status")
        if top_review == "human_reviewed":
            violations.append("AI proposals file top-level status claims human_reviewed")
        if raw.get("execution_verified") is True:
            violations.append("AI proposals file top-level claims execution_verified=true")
        entries_map = raw["proposals"]
    elif isinstance(raw, dict):
        entries_map = raw
    else:
        raise ValueError("AI proposals JSON must be an object keyed by case id or containing 'proposals'")

    unanimous_select = 0
    disputed_or_ambiguous = 0

    for cid, entry in sorted(entries_map.items()):
        if cid not in case_rows_by_id:
            violations.append(f"Proposal references unknown case id: {cid}")
            continue
        case_row = case_rows_by_id[cid]
        cand_tools = case_row.get("candidate_tools", [])
        proposal_list = entry if isinstance(entry, list) else [entry]
        if not proposal_list:
            violations.append(f"Case {cid} has empty proposal list")
            continue

        proposed_tools_for_case: Set[str] = set()
        proposed_decisions_for_case: Set[str] = set()

        for idx, prop in enumerate(proposal_list):
            if not isinstance(prop, dict):
                violations.append(f"Case {cid} proposal[{idx}] is not an object")
                continue
            for req_key in ("confidence", "disposition", "rationale", "proposer"):
                if not isinstance(prop.get(req_key), str) or not str(prop.get(req_key)).strip():
                    violations.append(f"Case {cid} proposal[{idx}] missing non-empty '{req_key}'")
            proposer = str(prop.get("proposer", "unknown"))
            disposition = str(prop.get("disposition", "unknown"))
            confidence = str(prop.get("confidence", "unknown"))
            proposer_counts[proposer] += 1
            disposition_counts[disposition] += 1
            confidence_counts[confidence] += 1

            if prop.get("review_status") == "human_reviewed" or prop.get("human_reviewed") is True:
                violations.append(f"Case {cid} proposal[{idx}] illegally marks human_reviewed")
            if prop.get("execution_verified") is True:
                violations.append(f"Case {cid} proposal[{idx}] illegally marks execution_verified=true")
            if any(k in prop for k in ("model_prediction", "predicted_tool", "model_logits", "model_probabilities")):
                violations.append(f"Case {cid} proposal[{idx}] contains model prediction fields (prohibited in adjudication)")

            dec = prop.get("decision")
            if dec is not None:
                if dec not in VALID_DECISIONS:
                    violations.append(f"Case {cid} proposal[{idx}] has invalid decision '{dec}'")
                else:
                    proposed_decisions_for_case.add(dec)

            choices = prop.get("acceptable_choices")
            if choices is not None:
                if not isinstance(choices, list):
                    violations.append(f"Case {cid} proposal[{idx}] acceptable_choices is not a list")
                else:
                    for ch in choices:
                        ch_errs = _check_choice_against_catalog(ch, cand_tools, catalog_spec)
                        for err in ch_errs:
                            violations.append(f"Case {cid} proposal[{idx}] invalid choice: {err}")
                        if isinstance(ch, dict) and isinstance(ch.get("tool"), str):
                            proposed_tools_for_case.add(ch["tool"])

        if proposed_decisions_for_case == {"select_tools"} and len(proposed_tools_for_case) == 1 and "ambiguous" not in {
            str(p.get("disposition", "")).lower() for p in proposal_list if isinstance(p, dict)
        }:
            unanimous_select += 1
        else:
            disputed_or_ambiguous += 1

    return {
        "present": True,
        "path": str(proposals_path),
        "sha256": compute_sha256(proposals_path),
        "proposal_count": len(entries_map),
        "violations": violations,
        "proposer_counts": dict(proposer_counts),
        "disposition_counts": dict(disposition_counts),
        "confidence_counts": dict(confidence_counts),
        "agreement_summary": {
            "cases_with_proposals": len(entries_map),
            "unanimous_select_tools_count": unanimous_select,
            "disputed_or_ambiguous_count": disputed_or_ambiguous,
        },
    }


def run_frozen_eval_audit(
    packet_dir: Optional[Path] = None,
    *,
    cases_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    proposals_path: Optional[Path] = None,
    catalog_path: Path = DEFAULT_CATALOG,
    historical_paths: Sequence[Path] = DEFAULT_HISTORICAL_PATHS,
    near_duplicate_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Run deterministic leakage, schema, source authenticity, and label review audit on a frozen-eval packet."""
    catalog_spec = load_catalog_spec(catalog_path)
    historical = load_historical_corpus(historical_paths)

    if packet_dir is not None:
        packet_dir = Path(packet_dir)
        if cases_path is None:
            cases_path = packet_dir / "cases.jsonl"
        if manifest_path is None:
            manifest_path = packet_dir / "manifest.json"
        if proposals_path is None and (packet_dir / "proposals.json").exists():
            proposals_path = packet_dir / "proposals.json"

    missing_prerequisites: List[str] = []
    if cases_path is None or not Path(cases_path).exists():
        missing_prerequisites.append(f"Missing sanitized candidate cases file: {cases_path or 'cases.jsonl'}")
    if manifest_path is None or not Path(manifest_path).exists():
        missing_prerequisites.append(f"Missing frozen-eval manifest file: {manifest_path or 'manifest.json'}")

    if missing_prerequisites:
        return {
            "audit_version": "frozen-eval-audit-v1",
            "packet_status": "missing_prerequisites",
            "label_review_gate": "blocked_missing_inputs",
            "evaluation_gate": "refused",
            "provisional_ai_agreement_available": False,
            "missing_prerequisites": missing_prerequisites,
            "catalog": {
                "catalog_version": catalog_spec["catalog_version"],
                "sha256": catalog_spec["sha256"],
                "tool_count": len(catalog_spec["tool_ids"]),
            },
            "historical_corpus": {
                "corpus_fingerprint": historical["corpus_fingerprint"],
                "total_records": historical["total_records"],
                "file_fingerprints": historical["file_fingerprints"],
            },
            "blockers": [{"code": "MISSING_PREREQUISITES", "message": msg} for msg in missing_prerequisites],
            "warnings": [],
        }

    cases_path = Path(cases_path)
    manifest_path = Path(manifest_path)
    cases_sha256 = compute_sha256(cases_path)
    manifest_sha256 = compute_sha256(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt JSONL at {cases_path}:{line_no}: {exc}") from exc

    blockers: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    # 1. Manifest structure and integrity checks
    schema_version = manifest.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        blockers.append({
            "code": "MANIFEST_SCHEMA_MISMATCH",
            "message": f"Expected schema_version='{EXPECTED_SCHEMA_VERSION}', got '{schema_version}'",
        })

    packet_status = manifest.get("status")
    if packet_status not in VALID_PACKET_STATUSES:
        blockers.append({
            "code": "INVALID_PACKET_STATUS",
            "message": f"Manifest status must be one of {VALID_PACKET_STATUSES}, got '{packet_status}'",
        })

    manifest_catalog_version = manifest.get("catalog_version")
    if manifest_catalog_version != catalog_spec["catalog_version"]:
        blockers.append({
            "code": "CATALOG_VERSION_MISMATCH",
            "message": (
                f"Manifest catalog_version '{manifest_catalog_version}' does not match "
                f"pinned catalog '{catalog_spec['catalog_version']}'"
            ),
        })

    # Verify manifest SHA-256 bindings
    manifest_hashes = manifest.get("sha256", {}) if isinstance(manifest.get("sha256"), dict) else {}
    recorded_cases_sha = (
        manifest.get("input_sha256")
        or manifest.get("cases_sha256")
        or manifest_hashes.get("cases.jsonl")
        or manifest_hashes.get("input")
    )
    if not recorded_cases_sha:
        blockers.append({
            "code": "MISSING_CASES_SHA256",
            "message": "Manifest must record SHA-256 of cases.jsonl (via sha256['cases.jsonl'] or cases_sha256/input_sha256)",
        })
    elif recorded_cases_sha != cases_sha256:
        blockers.append({
            "code": "CASES_SHA256_MISMATCH",
            "message": f"cases.jsonl SHA-256 mismatch: manifest has {recorded_cases_sha}, actual is {cases_sha256}",
        })

    recorded_catalog_sha = manifest.get("catalog_sha256") or manifest_hashes.get("tools.json")
    if recorded_catalog_sha and recorded_catalog_sha != catalog_spec["sha256"]:
        blockers.append({
            "code": "CATALOG_SHA256_MISMATCH",
            "message": f"tools.json SHA-256 mismatch: manifest has {recorded_catalog_sha}, actual is {catalog_spec['sha256']}",
        })

    for req_manifest_field in ("candidate_order_seed", "source_snapshot", "leak_audit_fingerprint"):
        if req_manifest_field not in manifest or manifest.get(req_manifest_field) in (None, "", {}):
            warnings.append({
                "code": f"MISSING_MANIFEST_{req_manifest_field.upper()}",
                "message": f"Manifest is missing recommended provenance field '{req_manifest_field}'",
            })

    # 2. Row-by-row audit: IDs, scenarios, source authenticity, leakage, labels, catalog capability
    seen_ids: Set[str] = set()
    duplicate_packet_ids: List[str] = []
    seen_req_sigs: Dict[str, str] = {}
    seen_state_sigs: Dict[str, str] = {}
    duplicate_packet_requests: List[Dict[str, str]] = []
    duplicate_packet_states: List[Dict[str, str]] = []

    historical_id_hits: List[Dict[str, Any]] = []
    historical_scenario_hits: List[Dict[str, Any]] = []
    historical_exact_req_hits: List[Dict[str, Any]] = []
    historical_exact_state_hits: List[Dict[str, Any]] = []
    historical_near_dup_hits: List[Dict[str, Any]] = []

    source_kind_counts: Counter[str] = Counter()
    actor_kind_counts: Counter[str] = Counter()
    non_human_actor_rows: List[Dict[str, str]] = []
    invalid_source_rows: List[Dict[str, str]] = []
    synthetic_rows: List[str] = []

    review_status_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    tool_counts_all_resolved: Counter[str] = Counter()
    tool_counts_real_human_reviewed: Counter[str] = Counter()
    observed_tool_counts: Counter[str] = Counter()

    unresolved_label_rows: List[str] = []
    observed_promoted_without_review: List[str] = []
    execution_verified_true_rows: List[str] = []
    catalog_capability_violations: List[Dict[str, str]] = []
    candidate_set_violations: List[Dict[str, str]] = []
    multi_tool_single_forced_warnings: List[Dict[str, str]] = []
    boundary_violations: List[Dict[str, Any]] = []
    privacy_warnings: List[Dict[str, Any]] = []

    case_rows_by_id: Dict[str, Dict[str, Any]] = {}
    pinned_tool_set = set(catalog_spec["tool_ids"])

    for idx, row in enumerate(rows):
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            blockers.append({"code": "MISSING_ROW_ID", "message": f"Row {idx} is missing a valid string 'id'"})
            rid = f"__missing_id_{idx}__"
        if rid in seen_ids:
            duplicate_packet_ids.append(rid)
        seen_ids.add(rid)
        case_rows_by_id[rid] = row

        sid = row.get("scenario_id") or row.get("session_group")
        if not isinstance(sid, str) or not sid.strip():
            blockers.append({"code": "MISSING_SCENARIO_ID", "message": f"Row {rid} is missing 'scenario_id'"})
            sid = ""

        # Check ID / scenario_id against historical corpus
        if rid in historical["ids"]:
            historical_id_hits.append({"id": rid, "historical_sources": historical["ids"][rid]})
        if sid and sid in historical["scenarios"]:
            historical_scenario_hits.append({"id": rid, "scenario_id": sid, "historical_sources": historical["scenarios"][sid]})

        # Feature hygiene & privacy check
        req = row.get("request", "")
        ctx = row.get("context", [])
        b_viol, p_warn = _scan_privacy_and_feature_hygiene(rid, req, ctx, catalog_spec["tool_ids"])
        boundary_violations.extend(b_viol)
        privacy_warnings.extend(p_warn)

        # Normalized request & state signatures
        req_str = req if isinstance(req, str) else ""
        ctx_list = ctx if isinstance(ctx, list) else []
        r_sig = normalize_text_signature(req_str)
        s_sig = normalize_state_signature(req_str, ctx_list)

        if r_sig:
            if r_sig in seen_req_sigs:
                duplicate_packet_requests.append({"id": rid, "first_id": seen_req_sigs[r_sig]})
            else:
                seen_req_sigs[r_sig] = rid
            if r_sig in historical["req_signatures"]:
                historical_exact_req_hits.append({
                    "id": rid,
                    "request": req_str,
                    "matches": historical["req_signatures"][r_sig][:3],
                })

        if s_sig:
            if s_sig in seen_state_sigs:
                duplicate_packet_states.append({"id": rid, "first_id": seen_state_sigs[s_sig]})
            else:
                seen_state_sigs[s_sig] = rid
            if s_sig in historical["state_signatures"]:
                historical_exact_state_hits.append({
                    "id": rid,
                    "matches": historical["state_signatures"][s_sig][:3],
                })

        # Near-duplicate lexical/3-gram similarity check against historical corpus
        row_tokens = tokenize_for_similarity(req_str)
        if row_tokens and r_sig not in historical["req_signatures"]:
            best_sim = 0.0
            best_match: Optional[Dict[str, str]] = None
            for h_item in historical["tokenized_requests"]:
                h_tokens = h_item["tokens"]
                union_len = len(row_tokens | h_tokens)
                if not union_len:
                    continue
                sim = len(row_tokens & h_tokens) / union_len
                if sim > best_sim:
                    best_sim = sim
                    best_match = h_item["meta"]
            if best_sim >= near_duplicate_threshold and best_match is not None:
                historical_near_dup_hits.append({
                    "id": rid,
                    "request": req_str,
                    "similarity": round(best_sim, 4),
                    "historical_match": best_match,
                })

        # Candidate tools & catalog version check
        cand_tools = row.get("candidate_tools")
        if not isinstance(cand_tools, list) or len(cand_tools) != 10 or len(set(cand_tools)) != 10 or set(cand_tools) != pinned_tool_set:
            candidate_set_violations.append({
                "id": rid,
                "issue": "candidate_tools must be the exact 10 pinned tools without duplicates",
            })
            cand_tools = list(catalog_spec["tool_ids"])

        row_cat_ver = row.get("tool_catalog_version")
        if row_cat_ver != catalog_spec["catalog_version"]:
            candidate_set_violations.append({
                "id": rid,
                "issue": f"row tool_catalog_version '{row_cat_ver}' != pinned '{catalog_spec['catalog_version']}'",
            })

        # Source provenance check (real vs synthetic, human_user vs cron/worker/system)
        src = row.get("source")
        src_kind: Optional[str] = None
        actor_kind = "unspecified"
        if isinstance(src, dict):
            src_kind = src.get("kind")
            raw_actor = src.get("actor") or src.get("origin") or src.get("role_kind") or src.get("channel_kind")
            if isinstance(raw_actor, str) and raw_actor.strip():
                actor_kind = raw_actor.strip().lower()
            if src.get("automated") is True or src.get("is_cron") is True or src.get("is_worker") is True:
                actor_kind = "automated"
            prov = src.get("provenance") or src.get("session_ref") or src.get("log_ref") or src.get("origin_id")
            if src_kind == "real" and (not isinstance(prov, str) or not prov.strip()):
                invalid_source_rows.append({"id": rid, "issue": "real source dict missing pseudonymous provenance/session_ref"})
        elif isinstance(src, str):
            if src in VALID_SOURCE_KINDS:
                src_kind = src
            invalid_source_rows.append({"id": rid, "issue": f"source must be structured object with kind=real|synthetic and provenance, got string '{src}'"})
        else:
            invalid_source_rows.append({"id": rid, "issue": "missing or non-object source field"})

        if src_kind not in VALID_SOURCE_KINDS:
            invalid_source_rows.append({"id": rid, "issue": f"invalid source.kind '{src_kind}' (expected real|synthetic)"})
            src_kind_key = "invalid"
        else:
            src_kind_key = src_kind
        source_kind_counts[src_kind_key] += 1
        if src_kind_key == "synthetic":
            synthetic_rows.append(rid)

        # Also inspect request text for obvious automated/worker/cron prompt signatures
        for auto_pat in AUTOMATED_REQUEST_PATTERNS:
            if auto_pat.search(req_str):
                actor_kind = "automated_pattern_detected"
                break

        actor_kind_counts[actor_kind] += 1
        if actor_kind in NON_HUMAN_ACTOR_KINDS or actor_kind == "automated_pattern_detected":
            non_human_actor_rows.append({"id": rid, "actor": actor_kind})

        # Observed choice preservation check
        obs_choice = row.get("observed_choice")
        obs_choices = row.get("observed_choices")
        observed_list: List[Dict[str, Any]] = []
        if isinstance(obs_choice, dict):
            observed_list.append(obs_choice)
        elif isinstance(obs_choices, list):
            observed_list.extend(c for c in obs_choices if isinstance(c, dict))
        for oc in observed_list:
            otool = oc.get("tool")
            if isinstance(otool, str):
                observed_tool_counts[otool] += 1

        # Review status and execution_verified check
        rev_status = row.get("review_status")
        if rev_status not in VALID_REVIEW_STATUSES:
            blockers.append({
                "code": "INVALID_REVIEW_STATUS",
                "message": f"Row {rid} has invalid review_status '{rev_status}'",
            })
            rev_status_key = "invalid"
        else:
            rev_status_key = rev_status
        review_status_counts[rev_status_key] += 1

        exec_ver = row.get("execution_verified")
        if not isinstance(exec_ver, bool):
            blockers.append({
                "code": "INVALID_EXECUTION_VERIFIED",
                "message": f"Row {rid} execution_verified must be boolean, got {type(exec_ver).__name__}",
            })
        elif exec_ver is True:
            execution_verified_true_rows.append(rid)

        # Decision and acceptable_choices check
        dec = row.get("decision")
        acc_choices = row.get("acceptable_choices")
        if not isinstance(acc_choices, list):
            blockers.append({
                "code": "INVALID_ACCEPTABLE_CHOICES",
                "message": f"Row {rid} acceptable_choices must be a list",
            })
            acc_choices = []

        if dec is None:
            decision_counts["null"] += 1
            unresolved_label_rows.append(rid)
            if acc_choices:
                catalog_capability_violations.append({
                    "id": rid,
                    "issue": "decision is null but acceptable_choices is non-empty",
                })
            if rev_status == "human_reviewed":
                blockers.append({
                    "code": "NULL_DECISION_MARKED_HUMAN_REVIEWED",
                    "message": f"Row {rid} has decision=null but review_status='human_reviewed'",
                })
        elif dec not in VALID_DECISIONS:
            decision_counts["invalid"] += 1
            blockers.append({
                "code": "INVALID_DECISION",
                "message": f"Row {rid} has invalid decision '{dec}'",
            })
        else:
            decision_counts[dec] += 1
            if dec == "select_tools":
                if not acc_choices:
                    unresolved_label_rows.append(rid)
                    catalog_capability_violations.append({
                        "id": rid,
                        "issue": "decision='select_tools' requires non-empty acceptable_choices",
                    })
                else:
                    row_tools: Set[str] = set()
                    for ch in acc_choices:
                        errs = _check_choice_against_catalog(ch, cand_tools, catalog_spec)
                        for e in errs:
                            catalog_capability_violations.append({"id": rid, "issue": e})
                        if isinstance(ch, dict) and isinstance(ch.get("tool"), str):
                            row_tools.add(ch["tool"])
                    if len(row_tools) == 1:
                        only_tool = next(iter(row_tools))
                        tool_counts_all_resolved[only_tool] += 1
                        if (
                            src_kind_key == "real"
                            and actor_kind not in NON_HUMAN_ACTOR_KINDS
                            and actor_kind != "automated_pattern_detected"
                            and rev_status == "human_reviewed"
                        ):
                            tool_counts_real_human_reviewed[only_tool] += 1
                    elif len(row_tools) > 1 and rev_status not in ("excluded", "needs_review", "needs_human_review"):
                        multi_tool_single_forced_warnings.append({
                            "id": rid,
                            "issue": f"Multiple distinct tools in acceptable_choices {sorted(row_tools)} for ten-tool single-choice benchmark",
                        })
            else:
                if acc_choices:
                    catalog_capability_violations.append({
                        "id": rid,
                        "issue": f"decision='{dec}' must have empty acceptable_choices, got {len(acc_choices)}",
                    })

            # Check if observed_choice was copied into acceptable_choices while still needs_human_review
            if rev_status == "needs_human_review" and acc_choices and observed_list:
                observed_promoted_without_review.append(rid)

            # Check if observed_choices had multiple distinct tools (parallel multi-tool call) but row was forced into single tool
            obs_distinct_tools = {oc.get("tool") for oc in observed_list if isinstance(oc.get("tool"), str)}
            if len(obs_distinct_tools) > 1 and dec == "select_tools" and rev_status == "human_reviewed":
                multi_tool_single_forced_warnings.append({
                    "id": rid,
                    "issue": f"Observed parallel multi-tool call {sorted(obs_distinct_tools)} was marked human_reviewed single select_tools",
                })

    # 3. Aggregate blockers from row-level findings
    if duplicate_packet_ids:
        blockers.append({
            "code": "DUPLICATE_PACKET_IDS",
            "message": f"Found {len(duplicate_packet_ids)} duplicate row IDs inside packet: {duplicate_packet_ids[:5]}",
        })
    if duplicate_packet_requests:
        blockers.append({
            "code": "DUPLICATE_PACKET_REQUESTS",
            "message": f"Found {len(duplicate_packet_requests)} normalized duplicate requests inside packet",
        })
    if historical_id_hits:
        blockers.append({
            "code": "HISTORICAL_ID_LEAKAGE",
            "message": f"Found {len(historical_id_hits)} row IDs colliding with historical datasets",
        })
    if historical_scenario_hits:
        blockers.append({
            "code": "HISTORICAL_SCENARIO_LEAKAGE",
            "message": f"Found {len(historical_scenario_hits)} scenario_ids colliding with historical datasets",
        })
    if historical_exact_req_hits or historical_exact_state_hits:
        blockers.append({
            "code": "HISTORICAL_EXACT_REQUEST_LEAKAGE",
            "message": (
                f"Found {len(historical_exact_req_hits)} normalized request collisions and "
                f"{len(historical_exact_state_hits)} state collisions against historical datasets"
            ),
        })
    if historical_near_dup_hits:
        warnings.append({
            "code": "HISTORICAL_NEAR_DUPLICATE_REQUESTS",
            "message": (
                f"Found {len(historical_near_dup_hits)} near-duplicate requests "
                f"(similarity >= {near_duplicate_threshold}) against historical datasets"
            ),
        })
    if invalid_source_rows:
        blockers.append({
            "code": "INVALID_SOURCE_PROVENANCE",
            "message": f"Found {len(invalid_source_rows)} rows with missing or invalid source metadata",
        })
    if synthetic_rows:
        blockers.append({
            "code": "SYNTHETIC_SOURCE_IN_REAL_BENCHMARK",
            "message": f"Found {len(synthetic_rows)} synthetic rows; synthetic data must never substitute for real requests",
        })
    if non_human_actor_rows:
        blockers.append({
            "code": "NON_HUMAN_ACTOR_SOURCE",
            "message": f"Found {len(non_human_actor_rows)} rows from cron/worker/system/automated sources rather than human user requests",
        })
    if candidate_set_violations:
        blockers.append({
            "code": "CANDIDATE_SET_OR_CATALOG_VIOLATION",
            "message": f"Found {len(candidate_set_violations)} rows violating fixed 10-tool candidate set or catalog version",
        })
    if catalog_capability_violations:
        blockers.append({
            "code": "CATALOG_CAPABILITY_VIOLATION",
            "message": f"Found {len(catalog_capability_violations)} rows with choices/actions unsupported by catalog schema",
        })
    if boundary_violations:
        blockers.append({
            "code": "FEATURE_BOUNDARY_OR_POST_TOOL_LEAKAGE",
            "message": f"Found {len(boundary_violations)} pre-decision context or tool-ID feature boundary violations",
        })
    if privacy_warnings:
        blockers.append({
            "code": "UNSANITIZED_PII_OR_SECRET_DETECTED",
            "message": f"Found {len(privacy_warnings)} potential unsanitized PII/secret/path matches in request/context",
        })
    if observed_promoted_without_review:
        warnings.append({
            "code": "UNREVIEWED_LABEL_PREFILLED",
            "message": f"Found {len(observed_promoted_without_review)} rows with prefilled acceptable_choices while still needs_human_review",
        })

    # 4. Verify manifest declared counts against actual counts
    manifest_counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    if "by_decision" in manifest_counts and isinstance(manifest_counts["by_decision"], dict):
        if dict(decision_counts) != manifest_counts["by_decision"]:
            blockers.append({
                "code": "MANIFEST_DECISION_COUNTS_MISMATCH",
                "message": f"Manifest counts.by_decision {manifest_counts['by_decision']} != actual {dict(decision_counts)}",
            })
    if "by_review_status" in manifest_counts and isinstance(manifest_counts["by_review_status"], dict):
        if dict(review_status_counts) != manifest_counts["by_review_status"]:
            blockers.append({
                "code": "MANIFEST_REVIEW_COUNTS_MISMATCH",
                "message": f"Manifest counts.by_review_status {manifest_counts['by_review_status']} != actual {dict(review_status_counts)}",
            })

    # 5. Audit optional AI proposals
    ai_proposals_report = audit_ai_proposals(proposals_path, case_rows_by_id, catalog_spec)
    if ai_proposals_report["violations"]:
        blockers.append({
            "code": "AI_PROPOSALS_CONTRACT_VIOLATION",
            "message": f"Found {len(ai_proposals_report['violations'])} contract violations in AI proposals file",
        })

    # 6. Compute per-tool quotas and shortages (target: 10 per fixed tool = 100 total)
    per_tool_quota: Dict[str, Dict[str, int]] = {}
    total_real_human_reviewed_shortage = 0
    total_resolved_shortage = 0
    for tid in catalog_spec["tool_ids"]:
        resolved_cnt = tool_counts_all_resolved.get(tid, 0)
        real_rev_cnt = tool_counts_real_human_reviewed.get(tid, 0)
        obs_cnt = observed_tool_counts.get(tid, 0)
        shortage_rev = max(0, TARGET_PER_TOOL - real_rev_cnt)
        shortage_res = max(0, TARGET_PER_TOOL - resolved_cnt)
        total_real_human_reviewed_shortage += shortage_rev
        total_resolved_shortage += shortage_res
        per_tool_quota[tid] = {
            "target": TARGET_PER_TOOL,
            "real_human_reviewed_count": real_rev_cnt,
            "resolved_label_count": resolved_cnt,
            "observed_choice_count": obs_cnt,
            "real_human_reviewed_shortage": shortage_rev,
            "resolved_label_shortage": shortage_res,
        }

    # 7. Determine orthogonal statuses:
    #    - packet_status: candidate | frozen (file immutability only, never implies label review)
    #    - label_review_gate: pending_initial_labels | needs_human_review | ai_provisional_only | human_reviewed_complete
    #    - evaluation_gate: refused | provisional_exploratory_only | ready_for_reviewed_benchmark
    all_rows_count = len(rows)
    has_hard_blockers = len(blockers) > 0
    all_labels_resolved = (all_rows_count > 0) and (len(unresolved_label_rows) == 0)
    all_human_reviewed = (
        all_rows_count == TARGET_TEN_TOOLS_TOTAL
        and all_labels_resolved
        and review_status_counts.get("human_reviewed", 0) == TARGET_TEN_TOOLS_TOTAL
        and total_real_human_reviewed_shortage == 0
    )

    if all_rows_count == 0:
        label_review_gate = "empty_packet"
    elif len(unresolved_label_rows) > 0:
        if ai_proposals_report["present"] and ai_proposals_report["agreement_summary"]["unanimous_select_tools_count"] > 0:
            label_review_gate = "unresolved_gold_with_ai_proposals_needs_human_review"
        else:
            label_review_gate = "pending_initial_labels_needs_human_review"
    elif all_human_reviewed:
        label_review_gate = "human_reviewed_complete"
    else:
        label_review_gate = "needs_human_review"

    if has_hard_blockers or not all_labels_resolved or packet_status != "frozen":
        evaluation_gate = "refused"
    elif all_human_reviewed:
        evaluation_gate = "ready_for_reviewed_benchmark"
    else:
        evaluation_gate = "provisional_exploratory_only_not_human_reviewed"

    provisional_ai_ready = (
        not has_hard_blockers
        and ai_proposals_report["present"]
        and ai_proposals_report["agreement_summary"]["unanimous_select_tools_count"] > 0
    )

    return {
        "audit_version": "frozen-eval-audit-v1",
        "schema_version": schema_version,
        "packet_status": packet_status,
        "packet_is_immutable_frozen": packet_status == "frozen",
        "freeze_implies_human_verified": False,
        "label_review_gate": label_review_gate,
        "evaluation_gate": evaluation_gate,
        "provisional_ai_agreement_available": provisional_ai_ready,
        "files": {
            "cases_path": str(cases_path),
            "cases_sha256": cases_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "proposals_path": ai_proposals_report["path"],
            "proposals_sha256": ai_proposals_report["sha256"],
        },
        "catalog": {
            "catalog_version": catalog_spec["catalog_version"],
            "source_catalog_version": catalog_spec["source_catalog_version"],
            "sha256": catalog_spec["sha256"],
            "tool_count": len(catalog_spec["tool_ids"]),
        },
        "historical_corpus": {
            "corpus_fingerprint": historical["corpus_fingerprint"],
            "total_records": historical["total_records"],
            "file_fingerprints": historical["file_fingerprints"],
        },
        "counts": {
            "total_rows": all_rows_count,
            "unique_ids": len(seen_ids),
            "unique_scenarios": len({r.get("scenario_id") for r in rows if r.get("scenario_id")}),
            "by_source_kind": dict(source_kind_counts),
            "by_actor_kind": dict(actor_kind_counts),
            "by_review_status": dict(review_status_counts),
            "by_decision": dict(decision_counts),
            "unresolved_label_count": len(unresolved_label_rows),
            "execution_verified_true_count": len(execution_verified_true_rows),
        },
        "quota_and_shortage": {
            "target_total": TARGET_TEN_TOOLS_TOTAL,
            "target_per_tool": TARGET_PER_TOOL,
            "real_human_reviewed_total": sum(tool_counts_real_human_reviewed.values()),
            "real_human_reviewed_shortage": total_real_human_reviewed_shortage,
            "resolved_label_total": sum(tool_counts_all_resolved.values()),
            "resolved_label_shortage": total_resolved_shortage,
            "per_tool": per_tool_quota,
        },
        "leakage": {
            "duplicate_packet_ids": duplicate_packet_ids,
            "duplicate_packet_requests": duplicate_packet_requests,
            "duplicate_packet_states": duplicate_packet_states,
            "historical_id_hits": historical_id_hits,
            "historical_scenario_hits": historical_scenario_hits,
            "historical_exact_request_hits": historical_exact_req_hits,
            "historical_exact_state_hits": historical_exact_state_hits,
            "historical_near_duplicate_hits": historical_near_dup_hits,
            "near_duplicate_threshold": near_duplicate_threshold,
            "zero_exact_leakage": (
                len(duplicate_packet_ids) == 0
                and len(duplicate_packet_requests) == 0
                and len(historical_id_hits) == 0
                and len(historical_scenario_hits) == 0
                and len(historical_exact_req_hits) == 0
                and len(historical_exact_state_hits) == 0
            ),
        },
        "source_authenticity": {
            "invalid_source_rows": invalid_source_rows,
            "synthetic_rows": synthetic_rows,
            "non_human_actor_rows": non_human_actor_rows,
        },
        "label_and_catalog_audit": {
            "unresolved_label_rows": unresolved_label_rows,
            "observed_promoted_without_review": observed_promoted_without_review,
            "execution_verified_true_rows": execution_verified_true_rows,
            "candidate_set_violations": candidate_set_violations,
            "catalog_capability_violations": catalog_capability_violations,
            "multi_tool_single_forced_warnings": multi_tool_single_forced_warnings,
        },
        "feature_and_privacy_hygiene": {
            "boundary_violations": boundary_violations,
            "privacy_warnings": privacy_warnings,
        },
        "ai_proposals": ai_proposals_report,
        "blockers": blockers,
        "warnings": warnings,
    }


def validate_safe_output_path(output_path: Path, protected_paths: Sequence[Optional[Path]]) -> Path:
    """Ensure output JSON never overwrites or aliases any input dataset, catalog, or packet file."""
    output_resolved = Path(output_path).resolve()
    validate_output_summary_path(output_path, DEFAULT_ROOT / "datasets")
    for p in protected_paths:
        if p is None:
            continue
        p_path = Path(p)
        if not p_path.exists():
            continue
        p_resolved = p_path.resolve()
        if p_resolved.is_dir():
            try:
                output_resolved.relative_to(p_resolved)
                raise ValueError(f"Refusing to write audit output inside protected directory: {p_path}")
            except ValueError as exc:
                if "Refusing to write audit output" in str(exc):
                    raise
        else:
            if output_resolved == p_resolved:
                raise ValueError(f"Refusing to overwrite input file with audit output: {p_path}")
            if output_path.exists():
                try:
                    if os.path.samefile(output_path, p_path):
                        raise ValueError(f"Audit output path aliases existing input file: {p_path}")
                except OSError:
                    pass
    return output_path


def format_console_summary(report: Dict[str, Any]) -> str:
    """Render concise human-readable audit summary."""
    lines = [
        "=" * 76,
        f"  JERY FROZEN-EVAL AUDITOR ({report.get('audit_version')})",
        "=" * 76,
        f"  Packet Status        : {report.get('packet_status')}",
        f"  Label Review Gate    : {report.get('label_review_gate')}",
        f"  Evaluation Gate      : {report.get('evaluation_gate')}",
        f"  Catalog Version      : {report.get('catalog', {}).get('catalog_version')}",
        f"  Historical Corpus FP : {report.get('historical_corpus', {}).get('corpus_fingerprint')}",
    ]
    if "counts" in report:
        c = report["counts"]
        q = report["quota_and_shortage"]
        lines.extend([
            f"  Total Rows           : {c['total_rows']} (Unresolved Gold: {c['unresolved_label_count']})",
            f"  Source Kinds         : {c['by_source_kind']}",
            f"  Actor Kinds          : {c['by_actor_kind']}",
            f"  Review Statuses      : {c['by_review_status']}",
            f"  Real Human-Reviewed  : {q['real_human_reviewed_total']}/{q['target_total']} (Shortage: {q['real_human_reviewed_shortage']})",
        ])
    if report.get("blockers"):
        lines.append(f"\n  [BLOCKERS ({len(report['blockers'])})]")
        for b in report["blockers"]:
            lines.append(f"  - [{b['code']}] {b['message']}")
    if report.get("warnings"):
        lines.append(f"\n  [WARNINGS ({len(report['warnings'])})]")
        for w in report["warnings"]:
            lines.append(f"  - [{w['code']}] {w['message']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, help="Directory containing cases.jsonl and manifest.json")
    parser.add_argument("--cases", type=Path, help="Explicit path to cases.jsonl")
    parser.add_argument("--manifest", type=Path, help="Explicit path to manifest.json")
    parser.add_argument("--proposals", type=Path, help="Explicit path to optional AI proposals JSON")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to pinned 10-tool catalog")
    parser.add_argument("--near-dup-threshold", type=float, default=0.55, help="Near-duplicate token Jaccard threshold")
    parser.add_argument("--output-json", type=Path, help="Write deterministic JSON audit report to path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report to stdout")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code when blockers are present")
    args = parser.parse_args(argv)

    try:
        report = run_frozen_eval_audit(
            packet_dir=args.packet_dir,
            cases_path=args.cases,
            manifest_path=args.manifest,
            proposals_path=args.proposals,
            catalog_path=args.catalog,
            near_duplicate_threshold=args.near_dup_threshold,
        )
        if args.output_json:
            protected = [
                args.packet_dir,
                args.cases,
                args.manifest,
                args.proposals,
                args.catalog,
                *DEFAULT_HISTORICAL_PATHS,
            ]
            target = validate_safe_output_path(args.output_json, protected)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        sys.stderr.write(f"Frozen-eval audit error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_console_summary(report))

    if args.strict and report.get("blockers"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
