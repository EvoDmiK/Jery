"""Independent leakage, source-authenticity, and label-review auditor for frozen-eval packets.

Python 3.9+ stdlib only. Reuses apps/review helpers to audit candidate/frozen packets
(cases.jsonl, candidate-manifest.json or manifest.json, and optional AI proposals JSON)
against the pinned 10-tool catalog and all historical corpora without mutating inputs.
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
        from .audit import compute_sha256, validate_output_summary_path
        from .routing_v2_audit import load_json_lines
    else:
        from audit import compute_sha256, validate_output_summary_path
        from routing_v2_audit import load_json_lines
except ImportError:
    from apps.review.audit import compute_sha256, validate_output_summary_path
    from apps.review.routing_v2_audit import load_json_lines

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

VALID_SCHEMA_VERSIONS = ("frozen-eval-v1", "frozen-eval-candidates-v1")
VALID_PACKET_STATUSES = ("candidate", "frozen")
VALID_DECISIONS = ("select_tools", "no_tool", "clarify", "unsupported")
VALID_REVIEW_STATUSES = ("needs_human_review", "human_reviewed", "excluded", "needs_review")
VALID_SOURCE_KINDS = ("real", "synthetic")
NON_HUMAN_ACTOR_KINDS = ("cron", "worker", "system", "agent", "automated")
TARGET_TEN_TOOLS_TOTAL = 100
TARGET_PER_TOOL = 10

AUTOMATED_REQUEST_PATTERNS = (
    re.compile(r"(?i)\b(?:orca\s+orchestration|worker_done|heartbeat|--dispatch-id|--task-id)\b"),
    re.compile(r"(?i)^\[?(?:cron|system|scheduled|heartbeat|auto-trigger|IMPORTANT\s+MCP)\b"),
    re.compile(r"(?i)\b(?:You are working inside Orca|You are a dispatched worker|task-preserved|gateway injected)\b"),
)

UNSANITIZED_PII_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----")),
    ("bearer_or_api_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]+|\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_\-]+")),
    ("credential_assignment", re.compile(r"(?im)(?:authorization|cookie|password|passwd|secret|token|credentials?|(?:api|access|refresh|auth)[_-]?(?:key|token))\s*[\"\x27]?\s*[:=]\s*[^\s\[][^\n]*")),
    ("raw_email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("raw_filesystem_path", re.compile(r"(?:[A-Za-z]:\\|(?<!\w)/(?:Users|Volumes|home|var|tmp|etc|opt)/)[^\s<>\"\x27,;]+")),
    ("bracket_speaker_nickname", re.compile(r"^\[(?!(?:url|path|relative_path|document|email|phone|user_id|personal_id|ip|private|secret)_[0-9a-f]{16,64}\])[^\]\n]{1,40}\]\s*")),
)

POST_CALL_CONTEXT_KEYS = {
    "tool_calls", "tool_call_id", "observed_choice", "observed_choices",
    "outcome", "execution_result", "post_call_result",
}


def dump_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text_signature(text: str) -> str:
    masked = re.sub(
        r"\[(?P<kind>url|path|relative_path|document|email|phone|user_id|personal_id|ip|private|secret)_[0-9a-f]{16,64}\]",
        r"[\g<kind>]",
        text,
        flags=re.I,
    )
    norm = unicodedata.normalize("NFKC", masked).casefold()
    return "".join(ch for ch in norm if ch.isalnum())


def normalize_state_signature(request: str, context: Sequence[Dict[str, Any]]) -> str:
    ctx = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in context if isinstance(m, dict)]
    return normalize_text_signature(dump_canonical({"request": request, "context": ctx}))


def tokenize_for_similarity(text: str) -> Set[str]:
    norm = unicodedata.normalize("NFKC", text).casefold()
    words = set(re.findall(r"[a-z0-9가-힣]+", norm))
    compact = "".join(ch for ch in norm if ch.isalnum())
    ngrams = {compact[i : i + 3] for i in range(max(0, len(compact) - 2))} if len(compact) >= 3 else set()
    return words | ngrams


def load_catalog_spec(catalog_path: Path) -> Dict[str, Any]:
    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools_by_id, action_enums, action_required = {}, {}, {}
    for tool in catalog.get("tools", []):
        tid = tool["id"]
        tools_by_id[tid] = tool
        schema = tool.get("input_schema") or {}
        act_prop = (schema.get("properties") or {}).get("action")
        action_enums[tid] = set(act_prop["enum"]) if isinstance(act_prop, dict) and isinstance(act_prop.get("enum"), list) else None
        action_required[tid] = "action" in (schema.get("required") or [])
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
    ids: Dict[str, List[str]] = {}
    scenarios: Dict[str, List[str]] = {}
    req_sigs: Dict[str, List[Dict[str, str]]] = {}
    state_sigs: Dict[str, List[Dict[str, str]]] = {}
    tokenized: List[Dict[str, Any]] = []
    seen_tok_sigs: Set[str] = set()
    file_fps: Dict[str, str] = {}
    total = 0

    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing historical file: {path}")
        try:
            rel = str(path.resolve().relative_to(DEFAULT_ROOT.resolve()))
        except ValueError:
            rel = str(path)
        file_fps[rel] = compute_sha256(path)

        records = (
            load_json_lines(path)
            if path.suffix == ".jsonl"
            else (json.loads(path.read_text(encoding="utf-8")).get("pairs") or [])
        )
        for idx, row in enumerate(records):
            if not isinstance(row, dict):
                continue
            total += 1
            rid = str(row.get("id") or row.get("source_id") or f"{rel}:{idx}")
            sid = str(row.get("scenario_id") or row.get("session_group") or "")
            split = str(row.get("split") or row.get("source_split") or ("diagnostic" if path.suffix == ".json" else "unknown"))
            state = row.get("state") if isinstance(row.get("state"), dict) else row
            req = str(state.get("request") or state.get("english_request") or "")
            ctx = state.get("context") if isinstance(state.get("context"), list) else []

            ids.setdefault(rid, []).append(f"{rel}:{split}")
            if sid:
                scenarios.setdefault(sid, []).append(f"{rel}:{split}")
            r_sig = normalize_text_signature(req)
            s_sig = normalize_state_signature(req, ctx) if path.suffix == ".jsonl" else ""
            meta = {"file": rel, "split": split, "id": rid, "scenario_id": sid}
            if r_sig:
                req_sigs.setdefault(r_sig, []).append(meta)
            if s_sig:
                state_sigs.setdefault(s_sig, []).append(meta)
            toks = tokenize_for_similarity(req)
            if toks and r_sig and r_sig not in seen_tok_sigs:
                seen_tok_sigs.add(r_sig)
                tokenized.append({"tokens": toks, "req_sig": r_sig, "meta": meta})

    return {
        "file_fingerprints": file_fps,
        "corpus_fingerprint": hashlib.sha256(dump_canonical(file_fps).encode("utf-8")).hexdigest(),
        "total_records": total,
        "ids": ids,
        "scenarios": scenarios,
        "req_signatures": req_sigs,
        "state_signatures": state_sigs,
        "tokenized_requests": tokenized,
    }


def _check_choice(choice: Any, candidate_tools: Sequence[str], catalog_spec: Dict[str, Any]) -> List[str]:
    if not isinstance(choice, dict):
        return ["choice must be an object"]
    errs: List[str] = []
    extra = set(choice.keys()) - {"tool", "action"}
    if extra:
        errs.append(f"unexpected keys in choice: {sorted(extra)}")
    tool = choice.get("tool")
    if not isinstance(tool, str) or tool not in catalog_spec["tools_by_id"]:
        return errs + [f"unknown catalog tool '{tool}'"]
    if tool not in candidate_tools:
        errs.append(f"tool '{tool}' not in candidate_tools")
    action = choice.get("action")
    enum_set = catalog_spec["action_enums"].get(tool)
    if enum_set is not None:
        if action is None and catalog_spec["action_required"].get(tool):
            errs.append(f"tool '{tool}' requires action from {sorted(enum_set)}")
        elif action is not None and action not in enum_set:
            errs.append(f"tool '{tool}' invalid action '{action}'")
    elif action is not None:
        errs.append(f"tool '{tool}' does not accept action, got '{action}'")
    return errs


def audit_ai_proposals(
    proposals_path: Optional[Path],
    case_rows_by_id: Dict[str, Dict[str, Any]],
    catalog_spec: Dict[str, Any],
) -> Dict[str, Any]:
    if proposals_path is None:
        return {
            "present": False, "path": None, "sha256": None, "proposal_count": 0, "violations": [],
            "proposer_counts": {}, "disposition_counts": {}, "confidence_counts": {}, "decision_counts": {},
            "eligible_single_tool_counts": {},
            "agreement_summary": {"cases_with_proposals": 0, "unanimous_select_tools_count": 0, "disputed_or_ambiguous_count": 0},
        }
    proposals_path = Path(proposals_path)
    raw = json.loads(proposals_path.read_text(encoding="utf-8"))
    violations: List[str] = []
    if isinstance(raw, dict) and isinstance(raw.get("proposals"), dict):
        if raw.get("review_status") == "human_reviewed" or raw.get("human_review_status") == "human_reviewed":
            violations.append("AI proposals top-level claims human_reviewed")
        if raw.get("execution_verified") is True:
            violations.append("AI proposals top-level claims execution_verified=true")
        entries_map = raw["proposals"]
    elif isinstance(raw, dict):
        entries_map = raw
    else:
        raise ValueError("Proposals JSON must be an object keyed by case_id")

    proposer_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    eligible_tool_counts: Counter[str] = Counter()
    unanimous_select = 0
    disputed = 0

    for cid, entry in sorted(entries_map.items()):
        if cid not in case_rows_by_id:
            violations.append(f"Unknown case_id in proposals: {cid}")
            continue
        cand_tools = case_rows_by_id[cid].get("candidate_tools", [])
        plist = entry if isinstance(entry, list) else [entry]
        if not plist:
            violations.append(f"Empty proposal list for {cid}")
            continue
        case_tools: Set[str] = set()
        case_decisions: Set[str] = set()
        case_dispositions: Set[str] = set()
        for idx, prop in enumerate(plist):
            if not isinstance(prop, dict):
                violations.append(f"{cid}[{idx}] not an object")
                continue
            for k in ("proposer", "disposition", "confidence", "rationale"):
                if not isinstance(prop.get(k), str) or not prop[k].strip():
                    violations.append(f"{cid}[{idx}] missing '{k}'")
            proposer_counts[str(prop.get("proposer", "unknown"))] += 1
            disp = str(prop.get("disposition", "unknown"))
            disposition_counts[disp] += 1
            case_dispositions.add(disp.lower())
            confidence_counts[str(prop.get("confidence", "unknown"))] += 1

            if prop.get("review_status") == "human_reviewed" or prop.get("human_reviewed") is True:
                violations.append(f"{cid}[{idx}] marks human_reviewed")
            if prop.get("execution_verified") is True:
                violations.append(f"{cid}[{idx}] marks execution_verified=true")
            if any(k in prop for k in ("model_prediction", "predicted_tool", "model_logits", "model_probabilities")):
                violations.append(f"{cid}[{idx}] contains prohibited model prediction field")

            dec = prop.get("decision")
            if dec is not None:
                if dec not in VALID_DECISIONS:
                    violations.append(f"{cid}[{idx}] invalid decision '{dec}'")
                else:
                    case_decisions.add(dec)
                    decision_counts[dec] += 1
            choices = prop.get("acceptable_choices")
            if choices is not None:
                if not isinstance(choices, list):
                    violations.append(f"{cid}[{idx}] acceptable_choices not a list")
                else:
                    for ch in choices:
                        for err in _check_choice(ch, cand_tools, catalog_spec):
                            violations.append(f"{cid}[{idx}] {err}")
                        if isinstance(ch, dict) and isinstance(ch.get("tool"), str):
                            case_tools.add(ch["tool"])

        if case_decisions == {"select_tools"} and len(case_tools) == 1 and not (case_dispositions & {"ambiguous", "exclude", "needs_context", "unsupported"}):
            unanimous_select += 1
            eligible_tool_counts[next(iter(case_tools))] += 1
        else:
            disputed += 1

    return {
        "present": True,
        "path": str(proposals_path),
        "sha256": compute_sha256(proposals_path),
        "proposal_count": len(entries_map),
        "violations": violations,
        "proposer_counts": dict(proposer_counts),
        "disposition_counts": dict(disposition_counts),
        "confidence_counts": dict(confidence_counts),
        "decision_counts": dict(decision_counts),
        "eligible_single_tool_counts": dict(eligible_tool_counts),
        "agreement_summary": {
            "cases_with_proposals": len(entries_map),
            "unanimous_select_tools_count": unanimous_select,
            "disputed_or_ambiguous_count": disputed,
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
    catalog_spec = load_catalog_spec(catalog_path)
    historical = load_historical_corpus(historical_paths)

    if packet_dir is not None:
        packet_dir = Path(packet_dir)
        if cases_path is None:
            cases_path = packet_dir / "cases.jsonl"
        if manifest_path is None:
            cand_m = packet_dir / "candidate-manifest.json"
            manifest_path = cand_m if cand_m.exists() else (packet_dir / "manifest.json")
        if proposals_path is None and (packet_dir / "proposals.json").exists():
            proposals_path = packet_dir / "proposals.json"

    missing = []
    if cases_path is None or not Path(cases_path).exists():
        missing.append(f"Missing sanitized candidate cases file: {cases_path or 'cases.jsonl'}")
    if manifest_path is None or not Path(manifest_path).exists():
        missing.append(f"Missing frozen-eval manifest file: {manifest_path or 'manifest.json'}")
    if missing:
        return {
            "audit_version": "frozen-eval-audit-v1",
            "packet_status": "missing_prerequisites",
            "label_review_gate": "blocked_missing_inputs",
            "evaluation_gate": "refused",
            "provisional_ai_agreement_available": False,
            "missing_prerequisites": missing,
            "catalog": {"catalog_version": catalog_spec["catalog_version"], "sha256": catalog_spec["sha256"], "tool_count": len(catalog_spec["tool_ids"])},
            "historical_corpus": {"corpus_fingerprint": historical["corpus_fingerprint"], "total_records": historical["total_records"], "file_fingerprints": historical["file_fingerprints"]},
            "blockers": [{"code": "MISSING_PREREQUISITES", "message": m} for m in missing],
            "warnings": [],
        }

    cases_path, manifest_path = Path(cases_path), Path(manifest_path)
    cases_sha, manifest_sha = compute_sha256(cases_path), compute_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_json_lines(cases_path)

    blockers: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    schema_version = manifest.get("schema_version")
    if schema_version not in VALID_SCHEMA_VERSIONS:
        blockers.append({"code": "MANIFEST_SCHEMA_MISMATCH", "message": f"Unexpected schema_version '{schema_version}'"})
    packet_status = manifest.get("status")
    if packet_status not in VALID_PACKET_STATUSES:
        blockers.append({"code": "INVALID_PACKET_STATUS", "message": f"Invalid status '{packet_status}'"})
    if manifest.get("catalog_version") != catalog_spec["catalog_version"]:
        blockers.append({"code": "CATALOG_VERSION_MISMATCH", "message": f"Catalog version mismatch: {manifest.get('catalog_version')}"})

    m_hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
    rec_cases_sha = manifest.get("cases_sha256") or manifest.get("input_sha256") or m_hashes.get("cases.jsonl")
    if not rec_cases_sha:
        blockers.append({"code": "MISSING_CASES_SHA256", "message": "Manifest missing cases_sha256"})
    elif rec_cases_sha != cases_sha:
        blockers.append({"code": "CASES_SHA256_MISMATCH", "message": "cases.jsonl SHA-256 mismatch"})

    rec_cat_sha = manifest.get("catalog_sha256") or m_hashes.get("tools.json")
    if rec_cat_sha and rec_cat_sha != catalog_spec["sha256"]:
        blockers.append({"code": "CATALOG_SHA256_MISMATCH", "message": "tools.json SHA-256 mismatch"})

    prev_hashes = manifest.get("previous_input_sha256")
    if isinstance(prev_hashes, dict):
        for k, v in prev_hashes.items():
            if k in historical["file_fingerprints"] and historical["file_fingerprints"][k] != v:
                blockers.append({"code": "HISTORICAL_HASH_MISMATCH", "message": f"Manifest previous_input_sha256 mismatch on {k}"})

    seen_ids: Set[str] = set()
    dup_ids, dup_reqs, dup_states = [], [], []
    seen_req_sigs: Dict[str, str] = {}
    seen_state_sigs: Dict[str, str] = {}
    hist_id_hits, hist_scen_hits, hist_req_hits, hist_state_hits, hist_near_hits = [], [], [], [], []

    source_kinds: Counter[str] = Counter()
    actor_kinds: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    tool_counts_resolved: Counter[str] = Counter()
    tool_counts_human_reviewed: Counter[str] = Counter()
    observed_tools: Counter[str] = Counter()

    non_human_rows, invalid_source_rows, synthetic_rows = [], [], []
    unresolved_rows, prefilled_unreviewed, exec_verified_rows = [], [], []
    cap_violations, cand_violations, boundary_violations, privacy_warnings, multi_tool_warnings = [], [], [], [], []
    context_dep_suspected_count = 0
    standalone_review_req_count = 0
    case_rows_by_id: Dict[str, Dict[str, Any]] = {}
    pinned_tools = set(catalog_spec["tool_ids"])

    for idx, row in enumerate(rows):
        rid = str(row.get("id") or "").strip()
        if not rid:
            blockers.append({"code": "MISSING_ROW_ID", "message": f"Row {idx} missing id"})
            rid = f"__missing_{idx}__"
        if rid in seen_ids:
            dup_ids.append(rid)
        seen_ids.add(rid)
        case_rows_by_id[rid] = row

        sid = str(row.get("scenario_id") or row.get("session_group") or "").strip()
        if not sid:
            blockers.append({"code": "MISSING_SCENARIO_ID", "message": f"Row {rid} missing scenario_id"})
        if rid in historical["ids"]:
            hist_id_hits.append({"id": rid, "sources": historical["ids"][rid]})
        if sid and sid in historical["scenarios"]:
            hist_scen_hits.append({"id": rid, "scenario_id": sid, "sources": historical["scenarios"][sid]})

        req = row.get("request", "")
        ctx = row.get("context", [])
        if not isinstance(req, str) or not req.strip():
            boundary_violations.append({"id": rid, "issue": "empty_or_non_string_request"})
            req = ""
        texts = [("request", req)]
        if not isinstance(ctx, list):
            boundary_violations.append({"id": rid, "issue": "context_not_a_list"})
            ctx = []
        else:
            for c_idx, msg in enumerate(ctx):
                if not isinstance(msg, dict):
                    boundary_violations.append({"id": rid, "issue": f"context[{c_idx}]_not_dict"})
                    continue
                if set(msg.keys()) & POST_CALL_CONTEXT_KEYS:
                    boundary_violations.append({"id": rid, "issue": f"context[{c_idx}]_post_call_keys"})
                if set(msg.keys()) - {"role", "content"}:
                    boundary_violations.append({"id": rid, "issue": f"context[{c_idx}]_extra_keys"})
                if msg.get("role") not in ("user", "assistant", "tool") or not isinstance(msg.get("content"), str):
                    boundary_violations.append({"id": rid, "issue": f"context[{c_idx}]_invalid_role_or_content"})
                else:
                    texts.append((f"context[{c_idx}]", msg["content"]))

        for fname, txt in texts:
            for tid in catalog_spec["tool_ids"]:
                if tid in txt:
                    boundary_violations.append({"id": rid, "issue": f"tool_id_in_{fname}:{tid}"})
            for pname, pat in UNSANITIZED_PII_PATTERNS:
                if pat.search(txt):
                    privacy_warnings.append({"id": rid, "field": fname, "pattern": pname})

        r_sig, s_sig = normalize_text_signature(req), normalize_state_signature(req, ctx)
        if r_sig:
            if r_sig in seen_req_sigs:
                dup_reqs.append({"id": rid, "first_id": seen_req_sigs[r_sig]})
            else:
                seen_req_sigs[r_sig] = rid
            if r_sig in historical["req_signatures"]:
                hist_req_hits.append({"id": rid, "matches": [{k: v for k, v in m.items() if k != "request"} for m in historical["req_signatures"][r_sig][:3]]})
        if s_sig:
            if s_sig in seen_state_sigs:
                dup_states.append({"id": rid, "first_id": seen_state_sigs[s_sig]})
            else:
                seen_state_sigs[s_sig] = rid
            if s_sig in historical["state_signatures"]:
                hist_state_hits.append({"id": rid, "matches": [{k: v for k, v in m.items() if k != "request"} for m in historical["state_signatures"][s_sig][:3]]})

        rtoks = tokenize_for_similarity(req)
        if rtoks and r_sig not in historical["req_signatures"]:
            best_sim, best_meta = 0.0, None
            for h_item in historical["tokenized_requests"]:
                ulen = len(rtoks | h_item["tokens"])
                sim = len(rtoks & h_item["tokens"]) / ulen if ulen else 0.0
                if sim > best_sim:
                    best_sim, best_meta = sim, h_item["meta"]
            if best_sim >= near_duplicate_threshold and best_meta is not None:
                hist_near_hits.append({"id": rid, "similarity": round(best_sim, 4), "historical_match": {k: v for k, v in best_meta.items() if k != "request"}})

        cands = row.get("candidate_tools")
        if not isinstance(cands, list) or len(cands) != 10 or set(cands) != pinned_tools:
            cand_violations.append({"id": rid, "issue": "candidate_tools_mismatch"})
            cands = list(catalog_spec["tool_ids"])
        if row.get("tool_catalog_version") != catalog_spec["catalog_version"]:
            cand_violations.append({"id": rid, "issue": "row_catalog_version_mismatch"})

        src = row.get("source")
        src_kind = src.get("kind") if isinstance(src, dict) else None
        actor = "unspecified"
        if isinstance(src, dict):
            raw_act = src.get("actor_kind") or src.get("actor") or src.get("origin") or src.get("record_type")
            if isinstance(raw_act, str) and raw_act.strip():
                actor = raw_act.strip().lower()
                if "discord_user_message" in actor or actor in ("human", "user", "human_user"):
                    actor = "user"
            if src.get("automated") is True or src.get("is_cron") is True or src.get("is_worker") is True:
                actor = "automated"
            if src.get("context_dependency_suspected") is True:
                context_dep_suspected_count += 1
            if src.get("standalone_label_review_required") is True:
                standalone_review_req_count += 1
            prov = src.get("provenance") or src.get("session_ref") or src.get("source_alias") or src.get("alias")
            if src_kind == "real" and (not isinstance(prov, str) or not prov.strip()):
                invalid_source_rows.append({"id": rid, "issue": "missing_provenance_or_source_alias"})
        else:
            invalid_source_rows.append({"id": rid, "issue": "source_not_object"})

        if src_kind not in VALID_SOURCE_KINDS:
            invalid_source_rows.append({"id": rid, "issue": f"invalid_source_kind:{src_kind}"})
            src_kind = "invalid"
        source_kinds[src_kind] += 1
        if src_kind == "synthetic":
            synthetic_rows.append(rid)

        if any(pat.search(req) for pat in AUTOMATED_REQUEST_PATTERNS):
            actor = "automated_pattern_detected"
        actor_kinds[actor] += 1
        if actor in NON_HUMAN_ACTOR_KINDS or actor == "automated_pattern_detected":
            non_human_rows.append({"id": rid, "actor": actor})

        obs = [row["observed_choice"]] if isinstance(row.get("observed_choice"), dict) else (
            [c for c in row.get("observed_choices", []) if isinstance(c, dict)] if isinstance(row.get("observed_choices"), list) else []
        )
        for oc in obs:
            if isinstance(oc.get("tool"), str):
                observed_tools[oc["tool"]] += 1

        rev = row.get("review_status")
        if rev not in VALID_REVIEW_STATUSES:
            blockers.append({"code": "INVALID_REVIEW_STATUS", "message": f"Row {rid} invalid review_status '{rev}'"})
            rev = "invalid"
        review_statuses[rev] += 1
        if isinstance(src, dict) and src.get("human_origin_verified") is True and rev != "human_reviewed":
            invalid_source_rows.append({"id": rid, "issue": "unreviewed_row_claims_human_origin_verified"})

        ev = row.get("execution_verified")
        if not isinstance(ev, bool):
            blockers.append({"code": "INVALID_EXECUTION_VERIFIED", "message": f"Row {rid} non-bool execution_verified"})
        elif ev is True:
            exec_verified_rows.append(rid)

        dec = row.get("decision")
        acc = row.get("acceptable_choices")
        if not isinstance(acc, list):
            blockers.append({"code": "INVALID_ACCEPTABLE_CHOICES", "message": f"Row {rid} acceptable_choices not list"})
            acc = []

        if dec is None:
            decisions["null"] += 1
            unresolved_rows.append(rid)
            if acc:
                cap_violations.append({"id": rid, "issue": "null_decision_with_non_empty_choices"})
            if rev == "human_reviewed":
                blockers.append({"code": "NULL_DECISION_MARKED_HUMAN_REVIEWED", "message": f"Row {rid} null decision marked human_reviewed"})
        elif dec not in VALID_DECISIONS:
            decisions["invalid"] += 1
            blockers.append({"code": "INVALID_DECISION", "message": f"Row {rid} invalid decision '{dec}'"})
        else:
            decisions[dec] += 1
            if dec == "select_tools":
                if not acc:
                    unresolved_rows.append(rid)
                    cap_violations.append({"id": rid, "issue": "select_tools_with_empty_choices"})
                else:
                    rtools: Set[str] = set()
                    for ch in acc:
                        for err in _check_choice(ch, cands, catalog_spec):
                            cap_violations.append({"id": rid, "issue": err})
                        if isinstance(ch, dict) and isinstance(ch.get("tool"), str):
                            rtools.add(ch["tool"])
                    if len(rtools) == 1:
                        only_t = next(iter(rtools))
                        tool_counts_resolved[only_t] += 1
                        if src_kind == "real" and actor not in NON_HUMAN_ACTOR_KINDS and actor != "automated_pattern_detected" and rev == "human_reviewed":
                            tool_counts_human_reviewed[only_t] += 1
                    elif len(rtools) > 1 and rev not in ("excluded", "needs_review", "needs_human_review"):
                        multi_tool_warnings.append({"id": rid, "issue": f"multi_tool_choices:{sorted(rtools)}"})
            elif acc:
                cap_violations.append({"id": rid, "issue": f"{dec}_with_non_empty_choices"})
            if rev == "needs_human_review" and acc and obs:
                prefilled_unreviewed.append(rid)

    for cond, code, msg in (
        (dup_ids, "DUPLICATE_PACKET_IDS", f"{len(dup_ids)} duplicate row IDs"),
        (dup_reqs, "DUPLICATE_PACKET_REQUESTS", f"{len(dup_reqs)} duplicate normalized requests"),
        (hist_id_hits, "HISTORICAL_ID_LEAKAGE", f"{len(hist_id_hits)} historical ID collisions"),
        (hist_scen_hits, "HISTORICAL_SCENARIO_LEAKAGE", f"{len(hist_scen_hits)} historical scenario collisions"),
        (hist_req_hits or hist_state_hits, "HISTORICAL_EXACT_REQUEST_LEAKAGE", f"{len(hist_req_hits)} request / {len(hist_state_hits)} state historical collisions"),
        (invalid_source_rows, "INVALID_SOURCE_PROVENANCE", f"{len(invalid_source_rows)} invalid source rows"),
        (synthetic_rows, "SYNTHETIC_SOURCE_IN_REAL_BENCHMARK", f"{len(synthetic_rows)} synthetic rows in real packet"),
        (non_human_rows, "NON_HUMAN_ACTOR_SOURCE", f"{len(non_human_rows)} non-human actor rows"),
        (cand_violations, "CANDIDATE_SET_OR_CATALOG_VIOLATION", f"{len(cand_violations)} candidate/catalog violations"),
        (cap_violations, "CATALOG_CAPABILITY_VIOLATION", f"{len(cap_violations)} catalog capability violations"),
        (boundary_violations, "FEATURE_BOUNDARY_OR_POST_TOOL_LEAKAGE", f"{len(boundary_violations)} feature boundary violations"),
        (privacy_warnings, "UNSANITIZED_PII_OR_SECRET_DETECTED", f"{len(privacy_warnings)} unsanitized PII/nickname/path matches"),
    ):
        if cond:
            blockers.append({"code": code, "message": msg})

    if hist_near_hits:
        warnings.append({"code": "HISTORICAL_NEAR_DUPLICATE_REQUESTS", "message": f"{len(hist_near_hits)} near-duplicate historical requests"})
    if prefilled_unreviewed:
        warnings.append({"code": "UNREVIEWED_LABEL_PREFILLED", "message": f"{len(prefilled_unreviewed)} unreviewed prefilled rows"})

    m_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    if isinstance(m_counts.get("candidate_requests"), int) and m_counts["candidate_requests"] != len(rows):
        blockers.append({"code": "MANIFEST_CANDIDATE_COUNT_MISMATCH", "message": "candidate_requests count mismatch"})
    if isinstance(m_counts.get("by_decision"), dict) and dict(decisions) != m_counts["by_decision"]:
        blockers.append({"code": "MANIFEST_DECISION_COUNTS_MISMATCH", "message": "by_decision count mismatch"})
    if isinstance(m_counts.get("by_review_status"), dict) and dict(review_statuses) != m_counts["by_review_status"]:
        blockers.append({"code": "MANIFEST_REVIEW_COUNTS_MISMATCH", "message": "by_review_status count mismatch"})

    ai_rep = audit_ai_proposals(proposals_path, case_rows_by_id, catalog_spec)
    if ai_rep["violations"]:
        blockers.append({"code": "AI_PROPOSALS_CONTRACT_VIOLATION", "message": f"{len(ai_rep['violations'])} AI proposal violations"})

    per_tool_quota: Dict[str, Dict[str, int]] = {}
    rev_shortage_total, res_shortage_total, ai_shortage_total = 0, 0, 0
    for tid in catalog_spec["tool_ids"]:
        res_c = tool_counts_resolved.get(tid, 0)
        rev_c = tool_counts_human_reviewed.get(tid, 0)
        ai_c = ai_rep["eligible_single_tool_counts"].get(tid, 0)
        s_rev = max(0, TARGET_PER_TOOL - rev_c)
        s_res = max(0, TARGET_PER_TOOL - res_c)
        s_ai = max(0, TARGET_PER_TOOL - ai_c)
        rev_shortage_total += s_rev
        res_shortage_total += s_res
        ai_shortage_total += s_ai
        per_tool_quota[tid] = {
            "target": TARGET_PER_TOOL,
            "real_human_reviewed_count": rev_c,
            "resolved_label_count": res_c,
            "ai_proposed_eligible_count": ai_c,
            "observed_choice_count": observed_tools.get(tid, 0),
            "real_human_reviewed_shortage": s_rev,
            "resolved_label_shortage": s_res,
            "ai_proposed_shortage": s_ai,
        }

    total_rows = len(rows)
    all_resolved = total_rows > 0 and len(unresolved_rows) == 0
    all_human_reviewed = (
        total_rows == TARGET_TEN_TOOLS_TOTAL and all_resolved
        and review_statuses.get("human_reviewed", 0) == TARGET_TEN_TOOLS_TOTAL and rev_shortage_total == 0
    )

    if total_rows == 0:
        label_gate = "empty_packet"
    elif unresolved_rows:
        label_gate = "unresolved_gold_with_ai_proposals_needs_human_review" if (ai_rep["present"] and ai_rep["proposal_count"] > 0) else "pending_initial_labels_needs_human_review"
    elif all_human_reviewed:
        label_gate = "human_reviewed_complete"
    else:
        label_gate = "needs_human_review"

    if blockers or not all_resolved or packet_status != "frozen":
        eval_gate = "refused"
    elif all_human_reviewed:
        eval_gate = "ready_for_reviewed_benchmark"
    else:
        eval_gate = "provisional_exploratory_only_not_human_reviewed"

    return {
        "audit_version": "frozen-eval-audit-v1",
        "schema_version": schema_version,
        "packet_status": packet_status,
        "packet_is_immutable_frozen": packet_status == "frozen",
        "freeze_implies_human_verified": False,
        "label_review_gate": label_gate,
        "evaluation_gate": eval_gate,
        "provisional_ai_agreement_available": not blockers and ai_rep["present"] and ai_rep["agreement_summary"]["unanimous_select_tools_count"] > 0,
        "files": {
            "cases_path": str(cases_path), "cases_sha256": cases_sha,
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
            "proposals_path": ai_rep["path"], "proposals_sha256": ai_rep["sha256"],
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
            "total_rows": total_rows,
            "unique_ids": len(seen_ids),
            "unique_scenarios": len({r.get("scenario_id") for r in rows if r.get("scenario_id")}),
            "by_source_kind": dict(source_kinds),
            "by_actor_kind": dict(actor_kinds),
            "by_review_status": dict(review_statuses),
            "by_decision": dict(decisions),
            "unresolved_label_count": len(unresolved_rows),
            "execution_verified_true_count": len(exec_verified_rows),
            "context_dependency_suspected_count": context_dep_suspected_count,
            "standalone_label_review_required_count": standalone_review_req_count,
        },
        "quota_and_shortage": {
            "target_total": TARGET_TEN_TOOLS_TOTAL,
            "target_per_tool": TARGET_PER_TOOL,
            "real_human_reviewed_total": sum(tool_counts_human_reviewed.values()),
            "real_human_reviewed_shortage": rev_shortage_total,
            "resolved_label_total": sum(tool_counts_resolved.values()),
            "resolved_label_shortage": res_shortage_total,
            "ai_proposed_eligible_total": sum(ai_rep["eligible_single_tool_counts"].values()),
            "ai_proposed_shortage": ai_shortage_total,
            "per_tool": per_tool_quota,
        },
        "leakage": {
            "duplicate_packet_ids": dup_ids,
            "duplicate_packet_requests": dup_reqs,
            "duplicate_packet_states": dup_states,
            "historical_id_hits": hist_id_hits,
            "historical_scenario_hits": hist_scen_hits,
            "historical_exact_request_hits": hist_req_hits,
            "historical_exact_state_hits": hist_state_hits,
            "historical_near_duplicate_hits": hist_near_hits,
            "near_duplicate_threshold": near_duplicate_threshold,
            "zero_exact_leakage": not (dup_ids or dup_reqs or hist_id_hits or hist_scen_hits or hist_req_hits or hist_state_hits),
        },
        "source_authenticity": {
            "invalid_source_rows": invalid_source_rows,
            "synthetic_rows": synthetic_rows,
            "non_human_actor_rows": non_human_rows,
        },
        "label_and_catalog_audit": {
            "unresolved_label_rows": unresolved_rows,
            "observed_promoted_without_review": prefilled_unreviewed,
            "execution_verified_true_rows": exec_verified_rows,
            "candidate_set_violations": cand_violations,
            "catalog_capability_violations": cap_violations,
            "multi_tool_single_forced_warnings": multi_tool_warnings,
        },
        "feature_and_privacy_hygiene": {
            "boundary_violations": boundary_violations,
            "privacy_warnings": privacy_warnings,
        },
        "ai_proposals": ai_rep,
        "blockers": blockers,
        "warnings": warnings,
    }


def validate_safe_output_path(output_path: Path, protected_paths: Sequence[Optional[Path]]) -> Path:
    output_resolved = Path(output_path).resolve()
    validate_output_summary_path(output_path, DEFAULT_ROOT / "datasets")
    for p in protected_paths:
        if p is None or not Path(p).exists():
            continue
        p_path = Path(p)
        p_res = p_path.resolve()
        if p_res.is_dir():
            try:
                output_resolved.relative_to(p_res)
                raise ValueError(f"Refusing to write audit output inside protected directory: {p_path}")
            except ValueError as exc:
                if "Refusing to write audit output" in str(exc):
                    raise
        elif output_resolved == p_res or (output_path.exists() and os.path.samefile(output_path, p_path)):
            raise ValueError(f"Refusing to overwrite or alias input file: {p_path}")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, help="Directory with cases.jsonl and candidate-manifest.json/manifest.json")
    parser.add_argument("--cases", type=Path, help="Explicit path to cases.jsonl")
    parser.add_argument("--manifest", type=Path, help="Explicit path to manifest JSON")
    parser.add_argument("--proposals", type=Path, help="Explicit path to optional AI proposals JSON")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to pinned 10-tool catalog")
    parser.add_argument("--near-dup-threshold", type=float, default=0.55, help="Near-duplicate token Jaccard threshold")
    parser.add_argument("--output-json", type=Path, help="Write deterministic JSON audit report to path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report to stdout")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code when blockers are present")
    args = parser.parse_args(argv)

    try:
        report = run_frozen_eval_audit(
            packet_dir=args.packet_dir, cases_path=args.cases, manifest_path=args.manifest,
            proposals_path=args.proposals, catalog_path=args.catalog, near_duplicate_threshold=args.near_dup_threshold,
        )
        if args.output_json:
            target = validate_safe_output_path(args.output_json, [args.packet_dir, args.cases, args.manifest, args.proposals, args.catalog, *DEFAULT_HISTORICAL_PATHS])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        sys.stderr.write(f"Frozen-eval audit error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"packet_status={report.get('packet_status')} label_review_gate={report.get('label_review_gate')} evaluation_gate={report.get('evaluation_gate')} blockers={len(report.get('blockers', []))}")
    return 1 if (args.strict and report.get("blockers")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
