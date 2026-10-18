"""Independent audit tool for Jery Laya pilot 32-case engineering export.

Python 3.9+ standard library only.
Audits outputs/laya-pilot-20261017/export/{manifest.json, train.jsonl, validation.jsonl}
against datasets/mcp-routing-v2 and accepted contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DEFAULT_ROOT / "datasets/mcp-routing-v2"
DEFAULT_EXPORT = DEFAULT_ROOT / "outputs/laya-pilot-20261017/export"

REQUIRED_SOURCE_HASH_KEYS = {"train.jsonl", "validation.jsonl", "tools.json"}
REQUIRED_SPLIT_KEYS = {"train", "validation"}
CARD_WHITELIST_FIELDS = {"alias", "server", "name", "summary"}
MAX_SUMMARY_LENGTH = 120
ALLOWED_QUESTIONS = {"route_in_window"}


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_raw_dataset_rows(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load raw dataset rows indexed by source_id per split, and catalog."""
    dataset_dir = Path(dataset_dir)
    catalog_path = dataset_dir / "tools.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing tools.json at {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    raw_by_split: Dict[str, Dict[str, Any]] = {"train": {}, "validation": {}}
    for split in ("train", "validation"):
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing raw split file: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_by_split[split][row["id"]] = row

    return raw_by_split, catalog


def load_pilot_export(export_dir: Path) -> Dict[str, Any]:
    """Load pilot export manifest and split jsonl files."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    splits_data = {}
    for split in ("train", "validation"):
        split_file = export_dir / f"{split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        rows = []
        for line_no, line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupt JSON at {split_file}:{line_no}: {e}")
        splits_data[split] = {
            "path": split_file,
            "rows": rows,
            "sha256": compute_sha256(split_file),
        }

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": compute_sha256(manifest_path),
        "manifest_size": manifest_path.stat().st_size,
        "splits": splits_data,
    }


def audit_source_checksums(manifest: Dict[str, Any], dataset_dir: Path) -> Dict[str, Any]:
    """Verify that source_sha256 in manifest contains all required keys and matches exact raw files."""
    dataset_dir = Path(dataset_dir)
    source_hashes = manifest.get("source_sha256", {})
    results = {}
    mismatches = []

    # Check for missing required hash keys
    missing_keys = REQUIRED_SOURCE_HASH_KEYS - set(source_hashes.keys())
    if missing_keys:
        for k in sorted(missing_keys):
            mismatches.append(f"Missing required source hash key in manifest: {k}")

    for filename in sorted(source_hashes.keys()):
        expected_hash = source_hashes[filename]
        target_file = dataset_dir / filename
        if not target_file.exists():
            mismatches.append(f"Missing source file in dataset: {target_file}")
            continue
        actual_hash = compute_sha256(target_file)
        matches = actual_hash == expected_hash
        results[filename] = {
            "expected": expected_hash,
            "actual": actual_hash,
            "matches": matches,
        }
        if not matches:
            mismatches.append(f"{filename}: expected {expected_hash}, got {actual_hash}")

    return {
        "verified": len(mismatches) == 0,
        "details": results,
        "mismatches": mismatches,
    }


def audit_export_integrity(export_data: Dict[str, Any]) -> Dict[str, Any]:
    """Verify split checksums, row counts, and scenario isolation."""
    manifest = export_data["manifest"]
    splits = export_data["splits"]
    errors = []

    if manifest.get("export_version") != "decision-export-v1":
        errors.append(f"Unexpected export_version: {manifest.get('export_version')}")
    if manifest.get("purpose") != "engineering_smoke_only":
        errors.append(f"Unexpected purpose: {manifest.get('purpose')}")

    missing_split_keys = REQUIRED_SPLIT_KEYS - set(manifest.get("splits", {}).keys())
    if missing_split_keys:
        for k in sorted(missing_split_keys):
            errors.append(f"Missing split key in manifest.splits: {k}")

    split_stats = {}
    scenario_groups_by_split: Dict[str, Set[str]] = {}

    for split in ("train", "validation"):
        split_meta = manifest.get("splits", {}).get(split, {})
        actual_data = splits[split]
        rows = actual_data["rows"]

        if actual_data["sha256"] != split_meta.get("sha256"):
            errors.append(
                f"{split}.jsonl sha256 mismatch: manifest {split_meta.get('sha256')} != actual {actual_data['sha256']}"
            )

        if len(rows) != split_meta.get("rows"):
            errors.append(
                f"{split} row count mismatch: manifest {split_meta.get('rows')} != actual {len(rows)}"
            )

        scenarios = {r["metadata"]["scenario_id"] for r in rows}
        scenario_groups_by_split[split] = scenarios

        decisions = Counter(r["metadata"]["original_decision"] for r in rows)
        internal_labels = Counter(r["gold"]["route_in_window"]["label"] for r in rows)
        negative_windows = sum(1 for r in rows if r["metadata"].get("derived_negative_window"))

        split_stats[split] = {
            "rows": len(rows),
            "scenario_groups": len(scenarios),
            "decisions": dict(decisions),
            "internal_labels": dict(internal_labels),
            "negative_windows": negative_windows,
        }

    overlap = scenario_groups_by_split.get("train", set()) & scenario_groups_by_split.get("validation", set())
    if overlap:
        errors.append(f"Scenario group leakage between train and validation: {overlap}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": split_stats,
        "scenario_leakage": list(overlap),
    }


def audit_source_row_fidelity(
    rows: List[Dict[str, Any]], raw_rows_by_id: Dict[str, Any], split_name: str
) -> Dict[str, Any]:
    """Verify 100% fidelity of metadata and state against raw dataset source rows."""
    errors = []
    verified_rows = 0

    for idx, row in enumerate(rows):
        row_id = row.get("metadata", {}).get("source_id", f"row_{idx}")
        meta = row.get("metadata", {})
        source_id = meta.get("source_id")

        if not source_id or source_id not in raw_rows_by_id:
            errors.append(f"{split_name}:{row_id}: source_id '{source_id}' not found in raw {split_name}.jsonl")
            continue

        raw = raw_rows_by_id[source_id]

        # 1. State equality check
        state = row.get("state", {})
        if state.get("request") != raw.get("request"):
            errors.append(f"{split_name}:{row_id}: state.request differs from raw source request")
        if state.get("context") != raw.get("context"):
            errors.append(f"{split_name}:{row_id}: state.context differs from raw source context")

        # 2. Metadata equality check
        if meta.get("original_candidate_tools") != raw.get("candidate_tools"):
            errors.append(f"{split_name}:{row_id}: metadata.original_candidate_tools differs from raw candidate_tools")
        if meta.get("acceptable_choices") != raw.get("acceptable_choices"):
            errors.append(f"{split_name}:{row_id}: metadata.acceptable_choices differs from raw acceptable_choices")
        if meta.get("scenario_id") != raw.get("scenario_id"):
            errors.append(f"{split_name}:{row_id}: metadata.scenario_id differs from raw scenario_id")
        if meta.get("original_decision") != raw.get("decision"):
            errors.append(f"{split_name}:{row_id}: metadata.original_decision differs from raw decision")
        if meta.get("tool_catalog_version") != raw.get("tool_catalog_version"):
            errors.append(f"{split_name}:{row_id}: metadata.tool_catalog_version differs from raw tool_catalog_version")
        if meta.get("source") != raw.get("source"):
            errors.append(f"{split_name}:{row_id}: metadata.source differs from raw source")
        if meta.get("review_status") != raw.get("review_status"):
            errors.append(f"{split_name}:{row_id}: metadata.review_status differs from raw review_status")
        if meta.get("execution_verified") != raw.get("execution_verified"):
            errors.append(f"{split_name}:{row_id}: metadata.execution_verified differs from raw execution_verified")
        if meta.get("source_split") != raw.get("split"):
            errors.append(f"{split_name}:{row_id}: metadata.source_split differs from raw split")

        verified_rows += 1

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "verified_rows": verified_rows,
    }


def audit_window_semantics(rows: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    """Audit C vs W mapping, window_none semantics, alias uniqueness, and target rules."""
    errors = []
    total_rows = len(rows)
    non_select_counts = Counter()
    negative_window_count = 0
    positive_select_count = 0

    for idx, row in enumerate(rows):
        row_id = row.get("metadata", {}).get("source_id", f"row_{idx}")
        meta = row.get("metadata", {})
        c_tools = meta.get("original_candidate_tools", [])
        w_tools = meta.get("window_tools", [])
        aliases = meta.get("alias_to_tool", {})
        orig_dec = meta.get("original_decision")
        is_negative = meta.get("derived_negative_window", False)
        acceptable = {c["tool"] for c in meta.get("acceptable_choices", [])}

        gold = row.get("gold", {}).get("route_in_window", {})
        label = gold.get("label")
        probs = gold.get("probabilities", {})

        # 1. W subset of C
        if not (1 <= len(w_tools) <= 8):
            errors.append(f"{row_id} ({split_name}): window length {len(w_tools)} not in 1..8")
        if not set(w_tools).issubset(set(c_tools)):
            diff = set(w_tools) - set(c_tools)
            errors.append(f"{row_id} ({split_name}): window contains fabricated tools not in C: {diff}")

        # 2. Aliases match W and are unique
        expected_aliases = {f"T{i}": tool for i, tool in enumerate(w_tools)}
        if aliases != expected_aliases:
            errors.append(f"{row_id} ({split_name}): alias mapping mismatch: {aliases} != {expected_aliases}")
        if len(set(aliases.keys())) != len(w_tools):
            errors.append(f"{row_id} ({split_name}): duplicate aliases found")

        # 3. Questions criteria
        criteria = row.get("questions", {}).get("route_in_window", {}).get("criteria", {})
        expected_criteria_keys = set(aliases.keys()) | {"no_tool", "clarify", "window_none"}
        if set(criteria.keys()) != expected_criteria_keys:
            errors.append(
                f"{row_id} ({split_name}): criteria keys mismatch: {set(criteria.keys())} != {expected_criteria_keys}"
            )

        # 4. Target semantics
        if orig_dec == "select_tools":
            if is_negative:
                negative_window_count += 1
                if any(t in acceptable for t in w_tools):
                    errors.append(f"{row_id} ({split_name}): negative window contains acceptable tool")
                if label != "window_none":
                    errors.append(f"{row_id} ({split_name}): negative window target label must be window_none, got {label}")
                if probs != {"window_none": 1.0}:
                    errors.append(f"{row_id} ({split_name}): negative window probs must be {{'window_none': 1.0}}, got {probs}")
            else:
                positive_select_count += 1
                matching_aliases = [alias for alias, tool in aliases.items() if tool in acceptable]
                if not matching_aliases:
                    errors.append(f"{row_id} ({split_name}): positive select has no matching tool in W")
                if label not in matching_aliases:
                    errors.append(f"{row_id} ({split_name}): label {label} not in matching aliases {matching_aliases}")
                expected_prob = 1.0 / len(matching_aliases) if matching_aliases else 0.0
                for a in matching_aliases:
                    if abs(probs.get(a, 0.0) - expected_prob) > 1e-6:
                        errors.append(
                            f"{row_id} ({split_name}): probability for {a} is {probs.get(a)}, expected {expected_prob}"
                        )
                if set(probs.keys()) != set(matching_aliases):
                    errors.append(f"{row_id} ({split_name}): extra keys in positive select probs: {set(probs.keys())}")
        elif orig_dec == "unsupported":
            non_select_counts["unsupported"] += 1
            if label != "window_none":
                errors.append(f"{row_id} ({split_name}): unsupported original decision must map to window_none, got {label}")
            if probs != {"window_none": 1.0}:
                errors.append(f"{row_id} ({split_name}): unsupported probs must be {{'window_none': 1.0}}, got {probs}")
        elif orig_dec in ("no_tool", "clarify"):
            non_select_counts[orig_dec] += 1
            if label != orig_dec:
                errors.append(f"{row_id} ({split_name}): {orig_dec} label must be {orig_dec}, got {label}")
            if probs != {orig_dec: 1.0}:
                errors.append(f"{row_id} ({split_name}): {orig_dec} probs must be {{{orig_dec}: 1.0}}, got {probs}")
        else:
            errors.append(f"{row_id} ({split_name}): unknown original decision {orig_dec}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "positive_select": positive_select_count,
        "negative_windows": negative_window_count,
        "non_select": dict(non_select_counts),
        "total_rows": total_rows,
    }


def audit_sanitization_and_cards(
    rows: List[Dict[str, Any]], split_name: str, catalog: Dict[str, Any]
) -> Dict[str, Any]:
    """Audit card field whitelist, 120-char limit, duplicate cards, unexpected questions, and leakage."""
    errors = []
    tools_by_id = {t["id"]: t for t in catalog.get("tools", [])}
    flag_checks = {
        "synthetic": 0,
        "needs_human_review": 0,
        "execution_unverified": 0,
    }

    decision_leakage_keywords = {"select_tools", "no_tool", "clarify", "unsupported", "window_none"}

    for idx, row in enumerate(rows):
        row_id = row.get("metadata", {}).get("source_id", f"row_{idx}")
        state = row.get("state", {})
        meta = row.get("metadata", {})
        aliases = meta.get("alias_to_tool", {})

        # 1. Verification flags integrity
        if meta.get("source") == "synthetic":
            flag_checks["synthetic"] += 1
        else:
            errors.append(f"{row_id}: source is not synthetic ({meta.get('source')})")

        if meta.get("review_status") == "needs_human_review":
            flag_checks["needs_human_review"] += 1
        else:
            errors.append(f"{row_id}: review_status is not needs_human_review ({meta.get('review_status')})")

        if meta.get("execution_verified") is False:
            flag_checks["execution_unverified"] += 1
        else:
            errors.append(f"{row_id}: execution_verified is not False ({meta.get('execution_verified')})")

        # 2. State structure & leakage check across request and context
        expected_state_keys = {"request", "context", "candidates"}
        if set(state.keys()) != expected_state_keys:
            errors.append(f"{row_id}: state keys {set(state.keys())} != {expected_state_keys}")

        req_text = state.get("request", "")
        ctx_texts = []
        for ctx_item in state.get("context", []):
            if isinstance(ctx_item, dict):
                ctx_texts.append(json.dumps(ctx_item, ensure_ascii=False))
            elif isinstance(ctx_item, str):
                ctx_texts.append(ctx_item)
        full_text_to_check = req_text + " " + " ".join(ctx_texts)

        # Ensure no tool ID leaked in request or context
        for t_id in tools_by_id:
            if t_id in full_text_to_check:
                errors.append(f"{row_id}: full tool ID {t_id} leaked in request/context")

        # Ensure no explicit decision token leakage in request or context
        for dec_kw in decision_leakage_keywords:
            if re.search(r"\b" + re.escape(dec_kw) + r"\b", full_text_to_check):
                errors.append(f"{row_id}: decision keyword '{dec_kw}' leaked in request/context")

        # 3. Card fields whitelist, 120-char limit, duplicate cards
        candidates = state.get("candidates", [])
        if len(candidates) != len(aliases):
            errors.append(f"{row_id}: candidates length {len(candidates)} != aliases length {len(aliases)}")

        seen_card_aliases = set()
        for card in candidates:
            # Field whitelist
            extra_fields = set(card.keys()) - CARD_WHITELIST_FIELDS
            if extra_fields:
                errors.append(f"{row_id}: card contains unwhitelisted fields: {extra_fields}")

            alias = card.get("alias")
            if alias in seen_card_aliases:
                errors.append(f"{row_id}: duplicate card alias '{alias}' in state.candidates")
            seen_card_aliases.add(alias)

            if alias not in aliases:
                errors.append(f"{row_id}: card alias {alias} not in alias_to_tool")
                continue

            tool_id = aliases[alias]
            spec = tools_by_id.get(tool_id)
            if not spec:
                errors.append(f"{row_id}: tool_id {tool_id} not in catalog")
                continue

            if card.get("server") != spec.get("server"):
                errors.append(f"{row_id}: card server {card.get('server')} != spec {spec.get('server')}")
            if card.get("name") != spec.get("name"):
                errors.append(f"{row_id}: card name {card.get('name')} != spec {spec.get('name')}")

            # 120 character limit check
            summary = card.get("summary", "")
            if len(summary) > MAX_SUMMARY_LENGTH:
                errors.append(
                    f"{row_id}: card summary length {len(summary)} exceeds {MAX_SUMMARY_LENGTH} limit"
                )

            # Prefix of catalog description check
            raw_desc_norm = " ".join(spec.get("description", "").split())
            if not raw_desc_norm.startswith(summary):
                errors.append(f"{row_id}: card summary for {alias} is not a valid prefix of catalog description")

        # 4. Questions whitelist (reject unexpected questions like action_choice)
        questions = row.get("questions", {})
        unexpected_questions = set(questions.keys()) - ALLOWED_QUESTIONS
        if unexpected_questions:
            errors.append(f"{row_id}: unexpected question keys found: {unexpected_questions}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "flag_checks": flag_checks,
        "total_rows": len(rows),
    }


def run_full_pilot_audit(
    export_dir: Path = DEFAULT_EXPORT, dataset_dir: Path = DEFAULT_DATASET
) -> Dict[str, Any]:
    """Run end-to-end audit on pilot export against raw dataset."""
    export_dir = Path(export_dir)
    dataset_dir = Path(dataset_dir)

    raw_by_split, catalog = load_raw_dataset_rows(dataset_dir)
    export_data = load_pilot_export(export_dir)
    manifest = export_data["manifest"]

    source_audit = audit_source_checksums(manifest, dataset_dir)
    export_audit = audit_export_integrity(export_data)

    split_audits = {}
    for split in ("train", "validation"):
        rows = export_data["splits"][split]["rows"]
        raw_rows = raw_by_split[split]
        fidelity_audit = audit_source_row_fidelity(rows, raw_rows, split)
        sem_audit = audit_window_semantics(rows, split)
        san_audit = audit_sanitization_and_cards(rows, split, catalog)
        split_audits[split] = {
            "source_fidelity": fidelity_audit,
            "semantics": sem_audit,
            "sanitization_and_cards": san_audit,
        }

    all_valid = (
        source_audit["verified"]
        and export_audit["valid"]
        and all(
            sa["source_fidelity"]["valid"]
            and sa["semantics"]["valid"]
            and sa["sanitization_and_cards"]["valid"]
            for sa in split_audits.values()
        )
    )

    all_errors = list(source_audit["mismatches"]) + list(export_audit["errors"])
    for split, sa in split_audits.items():
        all_errors.extend(sa["source_fidelity"]["errors"])
        all_errors.extend(sa["semantics"]["errors"])
        all_errors.extend(sa["sanitization_and_cards"]["errors"])

    return {
        "audit_passed": all_valid,
        "export_dir": str(export_dir),
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": export_data["manifest_sha256"],
        "manifest_size": export_data["manifest_size"],
        "source_checksums": source_audit,
        "export_integrity": export_audit,
        "split_audits": split_audits,
        "error_count": len(all_errors),
        "all_errors": all_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Jery Laya pilot 32-case engineering export")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT, help="Path to export directory")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="Path to v2 dataset directory")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    result = run_full_pilot_audit(args.export_dir, args.dataset_dir)
    output_str = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_str + "\n", encoding="utf-8")
        print(f"Audit summary written to {args.output_json}")

    print(f"Audit status: {'PASS' if result['audit_passed'] else 'FAIL'}")
    print(f"Total errors: {result['error_count']}")
    if not result["audit_passed"]:
        for err in result["all_errors"][:10]:
            print(f" - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
