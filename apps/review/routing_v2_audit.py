"""Independent audit and summary tool for laya-routing-v2 reports, epoch records, and validation controls.

Python 3.9+ standard library only.
Audits laya-routing-v2 and legacy training reports, epoch histories (metrics.jsonl),
supervisor tracking states (state.json), and validation export controls without
modifying any input files or claiming raw synthetic labels are human-reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    if __package__:
        from .audit import validate_output_summary_path
    else:
        from audit import validate_output_summary_path
except ImportError:
    from apps.review.audit import validate_output_summary_path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_EXPORT = DEFAULT_ROOT / "outputs/laya-full-20261017/export/validation.jsonl"
DEFAULT_BALANCED_STATE = DEFAULT_ROOT / "outputs/laya-balanced-20261018/tracking/state.json"
DEFAULT_BALANCED_RUNS_DIR = DEFAULT_ROOT / "outputs/model/laya-balanced-20261018"
DEFAULT_OBJECTIVE_AUDIT = DEFAULT_ROOT / "outputs/laya-diagnostics-20261019/objective-audit.json"

QUESTION = "route_in_window"
NULLS = ("no_tool", "clarify", "window_none")
DECISIONS = ("select_tools", "no_tool", "clarify", "unsupported")
DELEGATE = "delegate"
WINDOW_SIZE = 8
EXPECTED_V2_POLICY = "laya-routing-v2"
DIAGNOSTIC_W_NONE_WEIGHTS = (1.0, 0.5, 0.25)


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _ratio(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def _is_finite_prob(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)) and 0.0 <= float(val) <= 1.0


def load_json_lines(path: Path) -> List[Dict[str, Any]]:
    """Read complete JSONL records without modifying the source file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def extract_split_diagnostics(split_block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract case-level and window-level diagnostics from a validation/before_validation block."""
    if not isinstance(split_block, dict):
        return None
    case_block = split_block.get("case", {}) if isinstance(split_block.get("case"), dict) else {}
    window_block = split_block.get("window", {}) if isinstance(split_block.get("window"), dict) else {}
    by_kind = window_block.get("by_gold_kind", {}) if isinstance(window_block.get("by_gold_kind"), dict) else {}
    per_dec = case_block.get("per_decision", {}) if isinstance(case_block.get("per_decision"), dict) else {}

    def _kind_acc(kind: str) -> Optional[float]:
        entry = by_kind.get(kind)
        if isinstance(entry, dict) and _is_finite_prob(entry.get("accuracy")):
            return float(entry["accuracy"])
        return None

    def _kind_counts(kind: str) -> Optional[Dict[str, Any]]:
        entry = by_kind.get(kind)
        if isinstance(entry, dict):
            return {
                "correct": entry.get("correct"),
                "windows": entry.get("windows"),
                "accuracy": entry.get("accuracy"),
            }
        return None

    def _dec_metrics(dec: str) -> Optional[Dict[str, Any]]:
        entry = per_dec.get(dec)
        if isinstance(entry, dict):
            return {
                "precision": entry.get("precision"),
                "recall": entry.get("recall"),
                "f1": entry.get("f1"),
                "support": entry.get("support"),
            }
        return None

    tool_win_acc = _kind_acc("tool")
    neg_win_acc = _kind_acc("window_none")
    overall_win_acc = window_block.get("any_of_accuracy")
    case_acc = split_block.get("case_accuracy", case_block.get("case_accuracy"))

    return {
        "case_accuracy": case_acc,
        "decision_accuracy": case_block.get("decision_accuracy"),
        "decision_macro_f1": split_block.get("decision_macro_f1", case_block.get("decision_macro_f1")),
        "tool_accuracy": case_block.get("tool_accuracy"),
        "coverage": split_block.get("coverage", case_block.get("coverage")),
        "unnecessary_call_rate": case_block.get("unnecessary_call_rate"),
        "out_of_candidate_selections": case_block.get("out_of_candidate_selections"),
        "delegated": case_block.get("delegated"),
        "overall_window_accuracy": overall_win_acc,
        "positive_tool_window_recall": tool_win_acc,
        "positive_tool_window_accuracy": tool_win_acc,
        "window_none_accuracy": neg_win_acc,
        "no_tool_window_accuracy": _kind_acc("no_tool"),
        "clarify_window_accuracy": _kind_acc("clarify"),
        "window_by_gold_kind": {
            k: _kind_counts(k) for k in ("tool", "window_none", "no_tool", "clarify") if _kind_counts(k) is not None
        },
        "per_decision": {
            d: _dec_metrics(d) for d in DECISIONS if _dec_metrics(d) is not None
        },
    }


def compute_validation_export_control(validation_jsonl_path: Path) -> Dict[str, Any]:
    """Independently compute validation window/case counts and the constant always-window_none control."""
    rows = load_json_lines(validation_jsonl_path)
    if not rows:
        raise ValueError(f"Empty validation export: {validation_jsonl_path}")

    cases_windows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gold_by_case: Dict[str, Dict[str, Any]] = {}
    window_label_counts: Counter[str] = Counter()
    window_none_hits = 0

    for row in rows:
        meta = row["metadata"]
        sid = meta["source_id"]
        cases_windows[sid].append(row)
        gold_by_case[sid] = {
            "decision": meta["original_decision"],
            "candidate_tools": meta["original_candidate_tools"],
            "acceptable_choices": meta["acceptable_choices"],
        }
        q_gold = row["gold"][QUESTION]
        label = q_gold["label"]
        bucket = label if label in NULLS else "tool"
        window_label_counts[bucket] += 1
        probs = q_gold.get("probabilities", {})
        if probs.get("window_none", 0.0) > 0.0:
            window_none_hits += 1

    # Under constant choice='window_none' for every window:
    # Single-window cases resolve 'window_none' -> decision='unsupported', tool=None.
    # Multi-window cases have no tool winners in first pass; plurality vote over all 'window_none' -> 'unsupported'.
    case_decision_counts = Counter(g["decision"] for g in gold_by_case.values())
    total_cases = len(gold_by_case)
    total_windows = len(rows)
    unsupported_cases = case_decision_counts.get("unsupported", 0)

    # Verify via exact case_metrics logic for constant 'unsupported' prediction
    correct_cases = unsupported_cases
    always_none_case_acc = _ratio(correct_cases, total_cases)
    always_none_win_acc = window_none_hits / total_windows

    return {
        "validation_export_path": str(validation_jsonl_path),
        "validation_export_sha256": compute_sha256(validation_jsonl_path),
        "windows": total_windows,
        "cases": total_cases,
        "select_tool_cases": case_decision_counts.get("select_tools", 0),
        "negative_windows": window_label_counts.get("window_none", 0),
        "positive_tool_windows": window_label_counts.get("tool", 0),
        "no_tool_windows": window_label_counts.get("no_tool", 0),
        "clarify_windows": window_label_counts.get("clarify", 0),
        "case_decisions": dict(case_decision_counts),
        "window_labels": dict(window_label_counts),
        "always_window_none_control": {
            "window_accuracy": always_none_win_acc,
            "window_accuracy_rounded": round(always_none_win_acc, 4),
            "case_accuracy": always_none_case_acc,
            "tool_accuracy": 0.0,
            "unnecessary_call_rate": 0.0,
            "coverage": 1.0,
            "predicted_decision_for_all_cases": "unsupported",
            "explanation": (
                f"Because {window_label_counts.get('window_none', 0)}/{total_windows} "
                f"({always_none_win_acc * 100:.2f}%) validation windows have gold label window_none, "
                f"always predicting window_none achieves {always_none_win_acc * 100:.2f}% window accuracy "
                f"while predicting unsupported for every case ({correct_cases}/{total_cases} = "
                f"{(always_none_case_acc or 0.0) * 100:.2f}% case accuracy, 0% tool accuracy)."
            ),
        },
        "review_status_note": "Synthetic exploratory dataset (mcp-routing-v2); raw labels are not human-reviewed.",
    }


def _check_model_descriptor(desc: Any, name: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(desc, dict):
        errors.append(f"Missing or non-dict {name} descriptor in v2 report")
        return None
    required = ("artifact", "artifact_sha256", "epoch", "validation_case_accuracy", "format")
    for key in required:
        if key not in desc:
            errors.append(f"{name} descriptor missing required field '{key}'")
    if "artifact" in desc and (not isinstance(desc["artifact"], str) or not desc["artifact"].strip()):
        errors.append(f"{name}.artifact must be a non-empty string")
    if "artifact_sha256" in desc and (not isinstance(desc["artifact_sha256"], str) or len(desc["artifact_sha256"]) != 64):
        errors.append(f"{name}.artifact_sha256 must be a 64-char hex sha256 string")
    if "epoch" in desc and (not isinstance(desc["epoch"], int) or isinstance(desc["epoch"], bool) or desc["epoch"] < 1):
        errors.append(f"{name}.epoch must be a positive trained integer epoch (>= 1), got {desc.get('epoch')!r}")
    if "validation_case_accuracy" in desc and not _is_finite_prob(desc["validation_case_accuracy"]):
        errors.append(f"{name}.validation_case_accuracy must be a finite probability in [0, 1]")
    if "format" in desc and (not isinstance(desc["format"], str) or not desc["format"].strip()):
        errors.append(f"{name}.format must be a non-empty string")
    return desc


def _infer_always_none_from_split_block(split_block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(split_block, dict):
        return None
    w = split_block.get("window")
    if not isinstance(w, dict):
        return None
    total_w = w.get("windows")
    none_w = (w.get("by_gold_kind", {}) or {}).get("window_none", {}).get("windows")
    if isinstance(total_w, int) and total_w > 0 and isinstance(none_w, int) and 0 <= none_w <= total_w:
        return none_w / total_w
    return None


def resolve_validation_export_for_run(
    run_dir: Path,
    root_dir: Path = DEFAULT_ROOT,
) -> Optional[Path]:
    """Resolve the matching validation.jsonl path declared by a run's report.json or config.json."""
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    config_path = run_dir / "config.json"

    export_dir_str: Optional[str] = None
    assets_root_str: Optional[str] = None

    if report_path.exists():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            export_dir_str = (rep.get("export") or {}).get("dir") or (rep.get("config") or {}).get("export_dir")
            assets_root_str = (rep.get("config") or {}).get("assets_root")
        except Exception:
            pass
    if not export_dir_str and config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            export_dir_str = cfg.get("export_dir")
            assets_root_str = cfg.get("assets_root")
        except Exception:
            pass

    if not export_dir_str:
        return None

    exp_path = Path(export_dir_str)
    candidates: List[Path] = []
    if exp_path.is_absolute():
        candidates.append(exp_path / "validation.jsonl")
    else:
        if assets_root_str:
            candidates.append(Path(assets_root_str) / exp_path / "validation.jsonl")
        candidates.append(Path(root_dir) / exp_path / "validation.jsonl")
        candidates.append(run_dir / exp_path / "validation.jsonl")

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def audit_single_run(
    run_dir: Path,
    report_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    always_none_control_win_acc: Optional[float] = None,
    validation_control: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Audit a single training run directory (or explicit report/metrics paths)."""
    run_dir = Path(run_dir)
    r_path = Path(report_path) if report_path else (run_dir / "report.json")
    m_path = Path(metrics_path) if metrics_path else (run_dir / "metrics.jsonl")
    before_val_path = run_dir / "before_validation.json"
    config_path = run_dir / "config.json"

    errors: List[str] = []
    warnings: List[str] = []
    flags: List[str] = []

    report: Optional[Dict[str, Any]] = None
    if r_path.exists():
        report = json.loads(r_path.read_text(encoding="utf-8"))

    epoch_records: List[Dict[str, Any]] = []
    if m_path.exists():
        epoch_records = load_json_lines(m_path)

    if report is None and not epoch_records:
        raise FileNotFoundError(f"Neither report.json nor metrics.jsonl found in {run_dir}")

    config: Dict[str, Any] = {}
    if report and isinstance(report.get("config"), dict):
        config = report["config"]
    elif config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    before_val: Optional[Dict[str, Any]] = None
    if report and isinstance(report.get("before_validation"), dict):
        before_val = report["before_validation"]
    elif before_val_path.exists():
        before_val = json.loads(before_val_path.read_text(encoding="utf-8"))

    selection = report.get("selection", {}) if (report and isinstance(report.get("selection"), dict)) else {}

    # Check validation export alignment if validation_control is provided
    if validation_control is not None:
        val_sha = validation_control.get("validation_export_sha256")
        val_windows = validation_control.get("windows")
        val_cases = validation_control.get("cases")

        if report is not None:
            rep_val_sha = ((report.get("export") or {}).get("sha256") or {}).get("validation")
            if rep_val_sha and val_sha and rep_val_sha != val_sha:
                errors.append(
                    f"Mismatched validation export sha256: report expects {rep_val_sha}, but provided validation.jsonl has {val_sha}"
                )
            rep_val_data = (report.get("data") or {}).get("validation")
            if isinstance(rep_val_data, dict):
                if isinstance(val_windows, int) and rep_val_data.get("windows") != val_windows:
                    errors.append(
                        f"Mismatched validation windows count: report.data.validation has {rep_val_data.get('windows')}, validation.jsonl has {val_windows}"
                    )
                if isinstance(val_cases, int) and rep_val_data.get("cases") != val_cases:
                    errors.append(
                        f"Mismatched validation cases count: report.data.validation has {rep_val_data.get('cases')}, validation.jsonl has {val_cases}"
                    )
        if before_val is not None:
            bv_win = (before_val.get("window") or {}).get("windows")
            bv_cases = (before_val.get("case") or {}).get("cases")
            if isinstance(val_windows, int) and isinstance(bv_win, int) and bv_win != val_windows:
                errors.append(
                    f"Mismatched before_validation windows ({bv_win}) vs provided validation.jsonl ({val_windows})"
                )
            if isinstance(val_cases, int) and isinstance(bv_cases, int) and bv_cases != val_cases:
                errors.append(
                    f"Mismatched before_validation cases ({bv_cases}) vs provided validation.jsonl ({val_cases})"
                )

    # Determine contract status: v2 vs legacy (never reinterpret legacy as v2)
    config_policy = config.get("training_policy_version")
    sel_policy = selection.get("policy_version")
    v2_indicators = (
        config_policy == EXPECTED_V2_POLICY
        or sel_policy == EXPECTED_V2_POLICY
        or "window_none_loss_weight" in config
        or (report is not None and ("best_trained" in report or "search_score" in selection or "best_trained_epoch" in selection))
    )
    contract_mode = "laya-routing-v2" if v2_indicators else "legacy"
    run_status = "completed" if report is not None else "incomplete_or_interrupted"

    run_id = (
        (report.get("run_id") if report else None)
        or config.get("run_id")
        or run_dir.name
    )

    # Validate epoch records sequence and extract history
    epoch_summaries: List[Dict[str, Any]] = []
    seen_epochs: List[int] = []
    for idx, rec in enumerate(epoch_records):
        ep = rec.get("epoch")
        if not isinstance(ep, int) or isinstance(ep, bool) or ep < 1:
            errors.append(f"metrics.jsonl row {idx + 1} has invalid trained epoch: {ep!r} (must be int >= 1)")
        else:
            seen_epochs.append(ep)
        val_block = rec.get("validation")
        if not isinstance(val_block, dict):
            errors.append(f"metrics.jsonl epoch {ep} missing validation dict")
            diag = None
        else:
            diag = extract_split_diagnostics(val_block)
            if diag is None or not _is_finite_prob(diag.get("case_accuracy")):
                errors.append(f"metrics.jsonl epoch {ep} has invalid validation.case_accuracy")
            if not isinstance(val_block.get("case"), dict) or not isinstance(val_block.get("window"), dict):
                errors.append(f"metrics.jsonl epoch {ep} validation must contain both 'case' and 'window' dicts")
        epoch_summaries.append({
            "epoch": ep,
            "train_loss": rec.get("train_loss"),
            "steps": rec.get("steps"),
            "validation": diag,
            "callback": rec.get("callback"),
        })

    if seen_epochs and seen_epochs != list(range(1, len(seen_epochs) + 1)):
        errors.append(f"metrics.jsonl epochs are not contiguous 1..N: {seen_epochs}")

    # Baseline epoch 0 summary
    baseline_diag = extract_split_diagnostics(before_val)
    baseline_score = baseline_diag["case_accuracy"] if baseline_diag else None

    # Best trained & last trained from epoch history
    history_best_ep: Optional[int] = None
    history_best_score: Optional[float] = None
    history_best_diag: Optional[Dict[str, Any]] = None
    history_last_ep: Optional[int] = None
    history_last_score: Optional[float] = None
    history_last_diag: Optional[Dict[str, Any]] = None

    for item in epoch_summaries:
        ep = item["epoch"]
        diag = item["validation"]
        if isinstance(ep, int) and diag and _is_finite_prob(diag.get("case_accuracy")):
            score = float(diag["case_accuracy"])
            if history_best_score is None or score > history_best_score:
                history_best_score = score
                history_best_ep = ep
                history_best_diag = diag
            history_last_ep = ep
            history_last_score = score
            history_last_diag = diag

    completed_epochs = selection.get("completed_epochs") if selection else (len(seen_epochs) if seen_epochs else None)
    if report is not None and seen_epochs and completed_epochs is not None:
        if completed_epochs != len(seen_epochs) or completed_epochs != seen_epochs[-1]:
            errors.append(
                f"selection.completed_epochs ({completed_epochs}) does not match metrics.jsonl length ({len(seen_epochs)})"
            )

    # Serving-selected model
    serving_epoch = selection.get("selected_epoch") if selection else (report.get("validation", {}).get("epoch") if report else None)
    serving_best_epoch = selection.get("best_epoch") if selection else None
    serving_best_score = selection.get("best_score") if selection else None
    serving_val_diag = extract_split_diagnostics(report.get("validation")) if report else None
    serving_test_diag = extract_split_diagnostics(report.get("test")) if report else None

    # Infer constant always-window_none control accuracy from validation export or run's own window gold counts
    effective_always_none_acc = always_none_control_win_acc
    if effective_always_none_acc is None and validation_control is not None:
        effective_always_none_acc = float(validation_control["always_window_none_control"]["window_accuracy"])
    if effective_always_none_acc is None:
        effective_always_none_acc = (
            _infer_always_none_from_split_block(before_val)
            or (
                _infer_always_none_from_split_block(report.get("validation"))
                if report
                else None
            )
            or (
                _infer_always_none_from_split_block(epoch_records[0].get("validation"))
                if epoch_records
                else None
            )
        )

    # General consistency checks on report (both legacy and v2)
    if report is not None:
        if before_val is None or not _is_finite_prob(baseline_score):
            errors.append("report.before_validation missing or has invalid case_accuracy")
        val_sec = report.get("validation")
        test_sec = report.get("test")
        if not isinstance(val_sec, dict):
            errors.append("report.validation missing")
        if not isinstance(test_sec, dict):
            errors.append("report.test missing")
        if isinstance(val_sec, dict) and isinstance(test_sec, dict) and serving_epoch is not None:
            if val_sec.get("epoch") != serving_epoch:
                errors.append(f"report.validation.epoch ({val_sec.get('epoch')}) != selected_epoch ({serving_epoch})")
            if test_sec.get("epoch") != serving_epoch:
                errors.append(f"report.test.epoch ({test_sec.get('epoch')}) != selected_epoch ({serving_epoch})")
        if selection:
            if serving_best_epoch != serving_epoch:
                errors.append(
                    f"selection.best_epoch ({serving_best_epoch}) != selection.selected_epoch ({serving_epoch})"
                )
            if isinstance(val_sec, dict) and _is_finite_prob(serving_best_score) and _is_finite_prob(val_sec.get("case_accuracy")):
                if abs(float(val_sec["case_accuracy"]) - float(serving_best_score)) > 1e-6:
                    errors.append(
                        f"report.validation.case_accuracy ({val_sec.get('case_accuracy')}) != selection.best_score ({serving_best_score})"
                    )
            if serving_epoch == 0 and baseline_score is not None and _is_finite_prob(serving_best_score):
                if abs(float(serving_best_score) - float(baseline_score)) > 1e-6:
                    errors.append(
                        f"Serving selected_epoch is 0 (baseline), but selection.best_score ({serving_best_score}) != before_validation.case_accuracy ({baseline_score})"
                    )
            elif isinstance(serving_epoch, int) and serving_epoch > 0 and seen_epochs:
                matched = [e for e in epoch_summaries if e["epoch"] == serving_epoch]
                if not matched:
                    errors.append(f"Serving selected_epoch {serving_epoch} not found in metrics.jsonl")
                elif matched[0]["validation"] and _is_finite_prob(serving_best_score):
                    ep_acc = float(matched[0]["validation"]["case_accuracy"])
                    if abs(ep_acc - float(serving_best_score)) > 1e-6:
                        errors.append(
                            f"Serving selected_epoch {serving_epoch} score in selection ({serving_best_score}) != metrics.jsonl score ({ep_acc})"
                        )

            # Verify serving best_score is at least as high as baseline (when include_baseline is true) and trained epochs
            include_baseline = config.get("validation_callback", {}).get("include_baseline", True)
            if include_baseline and baseline_score is not None and _is_finite_prob(serving_best_score):
                if float(serving_best_score) + 1e-6 < float(baseline_score):
                    errors.append(
                        f"selection.best_score ({serving_best_score}) is lower than baseline epoch 0 ({baseline_score})"
                    )
            if history_best_score is not None and _is_finite_prob(serving_best_score):
                if float(serving_best_score) + 1e-6 < float(history_best_score):
                    errors.append(
                        f"selection.best_score ({serving_best_score}) is lower than best trained validation score ({history_best_score})"
                    )

    # Strict v2 contract validation
    w_none_weight = config.get("window_none_loss_weight")
    if contract_mode == "laya-routing-v2":
        if config_policy != EXPECTED_V2_POLICY:
            errors.append(
                f"v2 config.training_policy_version must be {EXPECTED_V2_POLICY!r}, got {config_policy!r}"
            )
        if (
            not isinstance(w_none_weight, (int, float))
            or isinstance(w_none_weight, bool)
            or not math.isfinite(float(w_none_weight))
            or not (0.0 < float(w_none_weight) <= 1.0)
        ):
            errors.append(
                f"v2 config.window_none_loss_weight must be in (0, 1], got {w_none_weight!r}"
            )
        cb_cfg = config.get("validation_callback")
        if not isinstance(cb_cfg, dict):
            errors.append("v2 config.validation_callback dict is required")
        else:
            if cb_cfg.get("min_epochs") != 10:
                errors.append(f"v2 config.validation_callback.min_epochs must be 10, got {cb_cfg.get('min_epochs')!r}")
            if cb_cfg.get("patience") != 5:
                errors.append(f"v2 config.validation_callback.patience must be 5, got {cb_cfg.get('patience')!r}")
            if cb_cfg.get("min_delta") != 0.004:
                errors.append(f"v2 config.validation_callback.min_delta must be 0.004, got {cb_cfg.get('min_delta')!r}")

        if report is not None:
            if sel_policy != EXPECTED_V2_POLICY:
                errors.append(
                    f"v2 report.selection.policy_version must be {EXPECTED_V2_POLICY!r}, got {sel_policy!r}"
                )
            for req_sel_key in (
                "policy_version",
                "best_trained_epoch",
                "best_trained_score",
                "search_score",
                "trained_bad_epochs",
                "min_epochs",
                "best_epoch",
                "best_score",
                "selected_epoch",
                "completed_epochs",
            ):
                if req_sel_key not in selection:
                    errors.append(f"v2 report.selection missing required field '{req_sel_key}'")

            bt_epoch = selection.get("best_trained_epoch")
            bt_score = selection.get("best_trained_score")
            s_score = selection.get("search_score")
            t_bad = selection.get("trained_bad_epochs")
            sel_min_ep = selection.get("min_epochs")

            if sel_min_ep != 10:
                errors.append(f"v2 selection.min_epochs must be 10, got {sel_min_ep!r}")
            if not isinstance(t_bad, int) or isinstance(t_bad, bool) or t_bad < 0:
                errors.append(f"v2 selection.trained_bad_epochs must be a non-negative integer, got {t_bad!r}")
            if not isinstance(bt_epoch, int) or isinstance(bt_epoch, bool) or bt_epoch < 1:
                errors.append(f"v2 selection.best_trained_epoch must be an integer >= 1 (never baseline 0), got {bt_epoch!r}")
            if isinstance(completed_epochs, int) and isinstance(bt_epoch, int) and bt_epoch > completed_epochs:
                errors.append(f"v2 selection.best_trained_epoch ({bt_epoch}) > completed_epochs ({completed_epochs})")
            if not _is_finite_prob(bt_score):
                errors.append(f"v2 selection.best_trained_score must be a finite probability in [0, 1], got {bt_score!r}")
            if not _is_finite_prob(s_score):
                errors.append(f"v2 selection.search_score must be a finite probability in [0, 1], got {s_score!r}")
            if _is_finite_prob(bt_score) and _is_finite_prob(s_score):
                if abs(float(s_score) - float(bt_score)) > 1e-6:
                    errors.append(
                        f"v2 selection.search_score ({s_score}) != selection.best_trained_score ({bt_score})"
                    )

            # Compare against metrics.jsonl trained history
            if history_best_score is not None and _is_finite_prob(bt_score):
                if abs(float(bt_score) - float(history_best_score)) > 1e-6:
                    errors.append(
                        f"v2 selection.best_trained_score ({bt_score}) != max trained validation accuracy in metrics.jsonl ({history_best_score})"
                    )
            if isinstance(bt_epoch, int) and seen_epochs:
                ep_match = [e for e in epoch_summaries if e["epoch"] == bt_epoch]
                if not ep_match:
                    errors.append(f"v2 selection.best_trained_epoch ({bt_epoch}) not found in metrics.jsonl")
                elif ep_match[0]["validation"] and _is_finite_prob(bt_score):
                    ep_score = float(ep_match[0]["validation"]["case_accuracy"])
                    if abs(ep_score - float(bt_score)) > 1e-6:
                        errors.append(
                            f"v2 selection.best_trained_epoch ({bt_epoch}) has score {ep_score} in metrics.jsonl, but best_trained_score is {bt_score}"
                        )

            # Check early stopping min_epochs adherence in v2
            if selection.get("stopped") is True and selection.get("reason") == "early_stopping":
                if isinstance(completed_epochs, int) and isinstance(sel_min_ep, int) and completed_epochs < sel_min_ep:
                    errors.append(
                        f"v2 run stopped early at epoch {completed_epochs} before min_epochs ({sel_min_ep})"
                    )

            # Check report.best_trained and report.last_trained descriptors
            bt_desc = _check_model_descriptor(report.get("best_trained"), "report.best_trained", errors)
            lt_desc = _check_model_descriptor(report.get("last_trained"), "report.last_trained", errors)

            if bt_desc:
                if bt_desc.get("epoch") != bt_epoch:
                    errors.append(
                        f"report.best_trained.epoch ({bt_desc.get('epoch')}) != selection.best_trained_epoch ({bt_epoch})"
                    )
                if _is_finite_prob(bt_desc.get("validation_case_accuracy")) and _is_finite_prob(bt_score):
                    if abs(float(bt_desc["validation_case_accuracy"]) - float(bt_score)) > 1e-6:
                        errors.append(
                            f"report.best_trained.validation_case_accuracy ({bt_desc.get('validation_case_accuracy')}) != selection.best_trained_score ({bt_score})"
                        )
            if lt_desc:
                if lt_desc.get("epoch") != completed_epochs:
                    errors.append(
                        f"report.last_trained.epoch ({lt_desc.get('epoch')}) != selection.completed_epochs ({completed_epochs})"
                    )
                if history_last_score is not None and _is_finite_prob(lt_desc.get("validation_case_accuracy")):
                    if abs(float(lt_desc["validation_case_accuracy"]) - float(history_last_score)) > 1e-6:
                        errors.append(
                            f"report.last_trained.validation_case_accuracy ({lt_desc.get('validation_case_accuracy')}) != last epoch validation accuracy ({history_last_score})"
                        )

    # Diagnostic pattern detection (both legacy and v2)
    baseline_retained = (serving_epoch == 0)
    if baseline_retained:
        flags.append("BASELINE_RETAINED_FOR_SERVING")

    # Detect premature baseline-based early stop pattern (e.g., stopping at epoch 5 < 10 while recovering)
    cb_min_epochs = config.get("validation_callback", {}).get("min_epochs")
    effective_min_epochs = cb_min_epochs if isinstance(cb_min_epochs, int) else 10
    trained_scores = [
        float(e["validation"]["case_accuracy"])
        for e in epoch_summaries
        if e.get("validation") and _is_finite_prob(e["validation"].get("case_accuracy"))
    ]
    is_recovering = len(trained_scores) >= 2 and trained_scores[-1] > trained_scores[0]
    monotonic_recovery = len(trained_scores) >= 3 and all(
        b >= a for a, b in zip(trained_scores, trained_scores[1:])
    ) and trained_scores[-1] > trained_scores[0]

    premature_baseline_stop = False
    if (
        selection.get("stopped") is True
        and serving_best_epoch == 0
        and isinstance(completed_epochs, int)
        and completed_epochs < effective_min_epochs
    ):
        premature_baseline_stop = True
        flags.append("PREMATURE_BASELINE_EARLY_STOP")
        if is_recovering:
            flags.append("STOPPED_DURING_TRAINED_RECOVERY")

    # Detect misleading aggregate window accuracy (high window_none accuracy + collapsed positive tool recall)
    misleading_window_epochs: List[int] = []
    for item in epoch_summaries:
        diag = item["validation"]
        if not diag:
            continue
        win_acc = diag.get("overall_window_accuracy")
        tool_rec = diag.get("positive_tool_window_recall")
        case_acc = diag.get("case_accuracy")
        if (
            _is_finite_prob(win_acc)
            and _is_finite_prob(tool_rec)
            and _is_finite_prob(case_acc)
            and float(win_acc) >= 0.65
            and float(tool_rec) <= 0.15
            and (baseline_score is None or float(case_acc) < float(baseline_score))
        ):
            misleading_window_epochs.append(item["epoch"])
    if misleading_window_epochs:
        flags.append("MISLEADING_HIGH_WINDOW_ACCURACY_REJECTION_COLLAPSE")

    # Detect always-selecting-tools hazard (high unnecessary_call_rate and zero no_tool/clarify recall)
    check_diag = history_best_diag or serving_val_diag
    if check_diag:
        ucr = check_diag.get("unnecessary_call_rate")
        no_tool_rec = (check_diag.get("per_decision", {}).get("no_tool") or {}).get("recall")
        clarify_rec = (check_diag.get("per_decision", {}).get("clarify") or {}).get("recall")
        if _is_finite_prob(ucr) and float(ucr) >= 0.5:
            flags.append("HIGH_UNNECESSARY_CALL_RATE")
        if no_tool_rec == 0.0 and clarify_rec == 0.0:
            flags.append("ZERO_NULL_DECISION_RECALL")

    # Determine validation ranking score (search_score in v2, best_trained validation score in legacy, never test!)
    v2_search_score = selection.get("search_score") if contract_mode == "laya-routing-v2" else None
    ranking_validation_score = (
        float(v2_search_score)
        if _is_finite_prob(v2_search_score)
        else history_best_score
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "contract_mode": contract_mode,
        "run_status": run_status,
        "audit_passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "diagnostic_flags": flags,
        "policy": {
            "training_policy_version": config_policy or sel_policy or "legacy_unspecified",
            "window_none_loss_weight": w_none_weight,
            "is_standard_diagnostic_w_none": (
                w_none_weight in DIAGNOSTIC_W_NONE_WEIGHTS if w_none_weight is not None else False
            ),
            "head_learning_rate": config.get("head_learning_rate"),
            "weight_decay": config.get("weight_decay"),
            "label_smoothing": config.get("label_smoothing"),
            "seed": config.get("seed"),
            "validation_callback": config.get("validation_callback"),
        },
        "epochs_overview": {
            "baseline_epoch": 0,
            "best_trained_epoch": selection.get("best_trained_epoch", history_best_ep),
            "serving_selected_epoch": serving_epoch,
            "last_trained_epoch": (report.get("last_trained", {}) or {}).get("epoch", history_last_ep) if report else history_last_ep,
            "completed_epochs": completed_epochs,
            "baseline_retained_for_serving": baseline_retained,
            "premature_baseline_early_stop": premature_baseline_stop,
            "recovering_when_stopped": is_recovering and premature_baseline_stop,
            "monotonic_trained_recovery": monotonic_recovery,
        },
        "scores_summary": {
            "baseline_epoch0_validation_case_accuracy": baseline_score,
            "best_trained_validation_case_accuracy": selection.get("best_trained_score", history_best_score),
            "search_score": v2_search_score,
            "ranking_validation_score": ranking_validation_score,
            "serving_selected_validation_case_accuracy": (
                serving_val_diag["case_accuracy"] if serving_val_diag else serving_best_score
            ),
            "last_trained_validation_case_accuracy": (
                (report.get("last_trained", {}) or {}).get("validation_case_accuracy", history_last_score)
                if report
                else history_last_score
            ),
            "serving_test_case_accuracy_observational_only": (
                serving_test_diag["case_accuracy"] if serving_test_diag else None
            ),
            "always_window_none_control_window_accuracy": (
                round(effective_always_none_acc, 4) if effective_always_none_acc is not None else None
            ),
        },
        "models": {
            "baseline_epoch0": baseline_diag,
            "best_trained_validation": {
                "epoch": selection.get("best_trained_epoch", history_best_ep),
                "descriptor": report.get("best_trained") if report else None,
                "metrics": history_best_diag,
            },
            "serving_selected": {
                "epoch": serving_epoch,
                "baseline_retained": baseline_retained,
                "restore_descriptor": report.get("restore") if report else None,
                "validation_metrics": serving_val_diag,
                "test_metrics_observational_only": serving_test_diag,
            },
            "last_trained": {
                "epoch": (report.get("last_trained", {}) or {}).get("epoch", history_last_ep) if report else history_last_ep,
                "descriptor": report.get("last_trained") if report else None,
                "metrics": history_last_diag,
            },
        },
        "epoch_trajectory": [
            {
                "epoch": e["epoch"],
                "train_loss": e["train_loss"],
                "case_accuracy": e["validation"]["case_accuracy"] if e["validation"] else None,
                "positive_tool_window_recall": e["validation"]["positive_tool_window_recall"] if e["validation"] else None,
                "window_none_accuracy": e["validation"]["window_none_accuracy"] if e["validation"] else None,
                "overall_window_accuracy": e["validation"]["overall_window_accuracy"] if e["validation"] else None,
                "unnecessary_call_rate": e["validation"]["unnecessary_call_rate"] if e["validation"] else None,
                "no_tool_recall": (
                    (e["validation"].get("per_decision", {}).get("no_tool") or {}).get("recall")
                    if e["validation"]
                    else None
                ),
                "clarify_recall": (
                    (e["validation"].get("per_decision", {}).get("clarify") or {}).get("recall")
                    if e["validation"]
                    else None
                ),
            }
            for e in epoch_summaries
        ],
        "misleading_window_accuracy_epochs": misleading_window_epochs,
    }


def rank_runs_by_validation(run_audits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank configurations strictly by validation/search_score (best trained validation accuracy), NEVER test.

    Ties break by lower head_learning_rate, then run_id.
    """
    sortable: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for audit in run_audits:
        score = audit["scores_summary"].get("ranking_validation_score")
        val_score = float(score) if _is_finite_prob(score) else -1.0
        lr = audit["policy"].get("head_learning_rate")
        lr_val = float(lr) if isinstance(lr, (int, float)) and math.isfinite(float(lr)) else float("inf")
        sortable.append((-val_score, lr_val, str(audit["run_id"]), audit))

    sortable.sort(key=lambda item: (item[0], item[1], item[2]))
    ranked: List[Dict[str, Any]] = []
    for rank_idx, (_, _, _, audit) in enumerate(sortable, 1):
        ranked.append({
            "rank": rank_idx,
            "run_id": audit["run_id"],
            "contract_mode": audit["contract_mode"],
            "run_status": audit["run_status"],
            "window_none_loss_weight": audit["policy"].get("window_none_loss_weight"),
            "head_learning_rate": audit["policy"].get("head_learning_rate"),
            "ranking_metric": "search_score (best_trained_validation_case_accuracy)",
            "ranking_validation_score": audit["scores_summary"].get("ranking_validation_score"),
            "serving_selected_validation_score": audit["scores_summary"].get("serving_selected_validation_case_accuracy"),
            "serving_selected_epoch": audit["epochs_overview"].get("serving_selected_epoch"),
            "best_trained_epoch": audit["epochs_overview"].get("best_trained_epoch"),
            "completed_epochs": audit["epochs_overview"].get("completed_epochs"),
            "baseline_retained_for_serving": audit["epochs_overview"].get("baseline_retained_for_serving"),
            "test_used_for_ranking": False,
        })
    return ranked


def audit_tracking_state(
    state_path: Path,
    run_audits_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Audit Hyperopt/supervisor tracking state.json for floored/constant search objectives vs serving scores."""
    state_path = Path(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = state.get("plan", {}) if isinstance(state.get("plan"), dict) else {}
    children = state.get("children", []) if isinstance(state.get("children"), list) else []

    policy_version = plan.get("training_policy_version", "legacy_unspecified")
    is_v2_plan = (policy_version == EXPECTED_V2_POLICY)

    errors: List[str] = []
    flags: List[str] = []
    finished_children: List[Dict[str, Any]] = []
    search_objectives: List[float] = []
    serving_validations: List[float] = []
    trained_best_scores: List[float] = []

    for child in children:
        local_id = child.get("local_id")
        status = child.get("status")
        selected_epoch = child.get("selected_epoch")
        final_val = child.get("final_validation")
        search_val = child.get("search_validation")

        run_audit = (run_audits_by_id or {}).get(local_id) if local_id else None
        best_trained = (
            run_audit["scores_summary"]["best_trained_validation_case_accuracy"]
            if run_audit
            else None
        )

        if status == "finished":
            if is_v2_plan:
                if not _is_finite_prob(search_val):
                    errors.append(f"v2 finished child {local_id} missing valid search_validation in state.json")
                elif best_trained is not None and abs(float(search_val) - float(best_trained)) > 1e-6:
                    errors.append(
                        f"v2 child {local_id} search_validation ({search_val}) != run best_trained_score ({best_trained})"
                    )
                used_val = float(search_val) if _is_finite_prob(search_val) else None
            else:
                used_val = float(search_val) if _is_finite_prob(search_val) else (
                    float(final_val) if _is_finite_prob(final_val) else None
                )

            obj = round(1.0 - used_val, 6) if used_val is not None else None
            if obj is not None:
                search_objectives.append(round(obj, 4))
            if _is_finite_prob(final_val):
                serving_validations.append(float(final_val))
            if _is_finite_prob(best_trained):
                trained_best_scores.append(float(best_trained))

            finished_children.append({
                "index": child.get("index"),
                "local_id": local_id,
                "status": status,
                "head_lr": child.get("head_lr"),
                "selected_epoch": selected_epoch,
                "completed_epochs": child.get("completed_epochs"),
                "serving_final_validation": final_val,
                "search_validation": search_val,
                "best_trained_validation": best_trained,
                "effective_hyperopt_loss": round(1.0 - used_val, 4) if used_val is not None else None,
            })

    # Detect constant/floored Hyperopt search objectives separately from valid identical serving models
    all_serving_baseline = bool(finished_children) and all(c.get("selected_epoch") == 0 for c in finished_children)
    equal_objectives = len(search_objectives) >= 2 and len(set(search_objectives)) == 1
    distinct_trained_scores = len(set(round(s, 4) for s in trained_best_scores)) > 1

    floored_search_objective_detected = False
    if equal_objectives:
        flags.append("EQUAL_HYPEROPT_SEARCH_OBJECTIVES")
        if all_serving_baseline and distinct_trained_scores:
            floored_search_objective_detected = True
            flags.append("BASELINE_FLOORED_HYPEROPT_OBJECTIVE_CONCEALS_TRAINED_DIFFERENCES")
            if is_v2_plan:
                errors.append(
                    "v2 Hyperopt search objectives are baseline-floored despite distinct best_trained_scores"
                )

    return {
        "state_path": str(state_path),
        "state_sha256": compute_sha256(state_path),
        "parent_run_id": state.get("parent_run_id"),
        "training_policy_version": policy_version,
        "contract_mode": "laya-routing-v2" if is_v2_plan else "legacy",
        "audit_passed": len(errors) == 0,
        "errors": errors,
        "diagnostic_flags": flags,
        "finished_trials_count": len(finished_children),
        "total_trials_count": len(children),
        "search_objectives_completed": search_objectives,
        "equal_search_objectives": equal_objectives,
        "all_serving_models_retained_baseline": all_serving_baseline,
        "valid_identical_serving_models": all_serving_baseline and len(set(serving_validations)) <= 1,
        "floored_search_objective_conceals_trained_variance": floored_search_objective_detected,
        "finished_children": finished_children,
    }


def audit_experiment_suite(
    runs_dir: Optional[Path] = None,
    run_dirs: Optional[Sequence[Path]] = None,
    state_path: Optional[Path] = None,
    validation_export_path: Optional[Path] = None,
    objective_audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a comprehensive read-only audit over specified runs, tracking state, and validation export."""
    discovered_dirs: List[Path] = []
    if run_dirs:
        for p in run_dirs:
            d = Path(p)
            if not d.exists():
                raise FileNotFoundError(f"Specified run directory does not exist: {d}")
            if d not in discovered_dirs:
                discovered_dirs.append(d)
    if runs_dir:
        r_dir = Path(runs_dir)
        if not r_dir.exists():
            raise FileNotFoundError(f"Specified runs directory does not exist: {r_dir}")
        for child in sorted(r_dir.iterdir()):
            if child.is_dir() and ((child / "report.json").exists() or (child / "metrics.jsonl").exists()):
                if child not in discovered_dirs:
                    discovered_dirs.append(child)

    if not discovered_dirs:
        raise ValueError("Zero training runs discovered to audit (refusing to report PASS on empty run set)")

    val_control: Optional[Dict[str, Any]] = None
    always_none_win_acc: Optional[float] = None
    if validation_export_path is not None:
        v_path = Path(validation_export_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Specified validation export does not exist: {v_path}")
        val_control = compute_validation_export_control(v_path)
        always_none_win_acc = float(val_control["always_window_none_control"]["window_accuracy"])

    run_audits: List[Dict[str, Any]] = []
    run_audits_by_id: Dict[str, Dict[str, Any]] = {}
    for d in discovered_dirs:
        ra = audit_single_run(
            d,
            always_none_control_win_acc=always_none_win_acc,
            validation_control=val_control,
        )
        run_audits.append(ra)
        run_audits_by_id[ra["run_id"]] = ra
        run_audits_by_id[d.name] = ra

    state_audit: Optional[Dict[str, Any]] = None
    if state_path is not None:
        s_path = Path(state_path)
        if not s_path.exists():
            raise FileNotFoundError(f"Specified state.json does not exist: {s_path}")
        state_audit = audit_tracking_state(s_path, run_audits_by_id=run_audits_by_id)

    objective_audit_cross_check: Optional[Dict[str, Any]] = None
    if objective_audit_path is not None:
        o_path = Path(objective_audit_path)
        if not o_path.exists():
            raise FileNotFoundError(f"Specified objective-audit.json does not exist: {o_path}")
        obj_doc = json.loads(o_path.read_text(encoding="utf-8"))
        mismatches: List[str] = []
        if val_control:
            v_doc = obj_doc.get("validation", {})
            for k in ("windows", "cases", "select_tool_cases", "negative_windows"):
                if v_doc.get(k) != val_control.get(k):
                    mismatches.append(f"validation.{k}: objective-audit={v_doc.get(k)} vs computed={val_control.get(k)}")
            ctrl_doc = obj_doc.get("always_window_none_control", {})
            if abs(float(ctrl_doc.get("window_accuracy", 0.0)) - val_control["always_window_none_control"]["window_accuracy"]) > 1e-9:
                mismatches.append("always_window_none_control.window_accuracy mismatch")
            if abs(float(ctrl_doc.get("case_accuracy", 0.0)) - val_control["always_window_none_control"]["case_accuracy"]) > 1e-6:
                mismatches.append("always_window_none_control.case_accuracy mismatch")
        objective_audit_cross_check = {
            "objective_audit_path": str(o_path),
            "objective_audit_sha256": compute_sha256(o_path),
            "matches_independent_computation": len(mismatches) == 0,
            "mismatches": mismatches,
        }

    ranking = rank_runs_by_validation(run_audits)
    all_runs_passed = bool(run_audits) and all(r["audit_passed"] for r in run_audits)
    state_passed = state_audit["audit_passed"] if state_audit else True
    cross_check_passed = (
        objective_audit_cross_check["matches_independent_computation"]
        if objective_audit_cross_check
        else True
    )

    return {
        "audit_tool": "apps/review/routing_v2_audit.py",
        "audit_passed": all_runs_passed and state_passed and cross_check_passed,
        "ranking_policy": "validation_search_score_only_never_test",
        "review_status_note": "Raw synthetic labels (mcp-routing-v2) are unreviewed exploratory targets.",
        "validation_export_control": val_control,
        "tracking_state_audit": state_audit,
        "objective_audit_cross_check": objective_audit_cross_check,
        "validation_ranking": ranking,
        "runs": run_audits,
    }


def format_console_summary(suite: Dict[str, Any]) -> str:
    """Format a concise human-readable summary of the audit suite."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("  JERY LAYA-ROUTING-V2 & EPOCH TRAJECTORY INDEPENDENT AUDIT")
    lines.append("=" * 80)
    lines.append(f"  Overall Audit Verdict : {'PASS' if suite['audit_passed'] else 'FAIL'}")
    lines.append(f"  Ranking Policy        : {suite['ranking_policy']}")
    lines.append(f"  Label Status          : {suite['review_status_note']}")

    val_ctrl = suite.get("validation_export_control")
    if val_ctrl:
        ctrl = val_ctrl["always_window_none_control"]
        lines.append("\n[1] Validation Export & Constant Always-window_none Control")
        lines.append(
            f"  Cases: {val_ctrl['cases']} (select_tools={val_ctrl['select_tool_cases']}) | "
            f"Windows: {val_ctrl['windows']} (window_none={val_ctrl['negative_windows']}, "
            f"tool={val_ctrl['positive_tool_windows']}, no_tool={val_ctrl['no_tool_windows']}, "
            f"clarify={val_ctrl['clarify_windows']})"
        )
        lines.append(
            f"  Always-window_none Control -> Window Acc: {ctrl['window_accuracy'] * 100:.2f}% "
            f"| Case Acc: {ctrl['case_accuracy'] * 100:.2f}% | Tool Acc: 0.00%"
        )

    st = suite.get("tracking_state_audit")
    if st:
        lines.append("\n[2] Hyperopt / Supervisor Tracking State Audit")
        lines.append(
            f"  Policy Version: {st['training_policy_version']} ({st['contract_mode']}) | "
            f"Finished Trials: {st['finished_trials_count']}/{st['total_trials_count']}"
        )
        lines.append(f"  Completed Search Objectives : {st['search_objectives_completed']}")
        lines.append(f"  Equal/Floored Objectives    : {st['floored_search_objective_conceals_trained_variance']}")
        lines.append(f"  Diagnostic Flags            : {st['diagnostic_flags']}")

    lines.append("\n[3] Run-Level Epoch & Contract Audit")
    for r in suite.get("runs", []):
        eo = r["epochs_overview"]
        sc = r["scores_summary"]
        lines.append(
            f"  - {r['run_id']} [{r['contract_mode']} | {r['run_status']}] "
            f"Verdict={'PASS' if r['audit_passed'] else 'FAIL'}"
        )
        lines.append(
            f"    Epochs : baseline=0 | best_trained={eo['best_trained_epoch']} | "
            f"serving_selected={eo['serving_selected_epoch']} | last_trained={eo['last_trained_epoch']} | "
            f"completed={eo['completed_epochs']} | baseline_retained={eo['baseline_retained_for_serving']}"
        )
        lines.append(
            f"    ValAcc : baseline={sc['baseline_epoch0_validation_case_accuracy']} | "
            f"best_trained={sc['best_trained_validation_case_accuracy']} | "
            f"search_score={sc['search_score']} | serving={sc['serving_selected_validation_case_accuracy']} | "
            f"last_trained={sc['last_trained_validation_case_accuracy']}"
        )
        if r["epoch_trajectory"]:
            traj = ", ".join(
                f"ep{e['epoch']}:case={e['case_accuracy']}/toolWin={round(e['positive_tool_window_recall'], 4) if e['positive_tool_window_recall'] is not None else None}/win={round(e['overall_window_accuracy'], 4) if e['overall_window_accuracy'] is not None else None}"
                for e in r["epoch_trajectory"]
            )
            lines.append(f"    History: {traj}")
        if r["diagnostic_flags"]:
            lines.append(f"    Flags  : {r['diagnostic_flags']}")
        if r["errors"]:
            lines.append(f"    Errors : {r['errors']}")

    lines.append("\n[4] Validation-Only Ranking (Test Never Used)")
    for row in suite.get("validation_ranking", []):
        lines.append(
            f"  #{row['rank']} {row['run_id']} | val_rank_score={row['ranking_validation_score']} | "
            f"best_trained_ep={row['best_trained_epoch']} | serving_ep={row['serving_selected_epoch']} "
            f"(serving_val={row['serving_selected_validation_score']}) | lr={row['head_learning_rate']} | "
            f"w_none={row['window_none_loss_weight']}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def _protect_output_path(output_path: Path, protected_paths: Sequence[Optional[Path]]) -> None:
    """Ensure output_path never overwrites or resides inside any input directory or aliases an input file."""
    out_resolved = output_path.resolve()
    for p in protected_paths:
        if p is None:
            continue
        p_path = Path(p)
        if not p_path.exists():
            continue
        if p_path.is_dir():
            validate_output_summary_path(output_path, p_path)
            for sub in p_path.iterdir():
                if sub.is_dir():
                    validate_output_summary_path(output_path, sub)
        else:
            validate_output_summary_path(output_path, p_path.parent)
            if out_resolved == p_path.resolve():
                raise ValueError(f"Refusing to overwrite input file: {output_path}")
            if output_path.exists():
                try:
                    if os.path.samefile(output_path, p_path):
                        raise ValueError(f"Output path aliases existing input file: {output_path}")
                except OSError:
                    pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository/assets root for default relative path resolution (defaults to repo root)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="Directory containing child run folders")
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", help="Specific run directory to audit (repeatable)")
    parser.add_argument("--state-json", type=Path, default=None, help="Path to supervisor state.json")
    parser.add_argument("--validation-export", type=Path, default=None, help="Path to validation.jsonl export")
    parser.add_argument("--objective-audit", type=Path, default=None, help="Path to objective-audit.json")
    parser.add_argument("--output-json", type=Path, help="Write full audit JSON report to path")
    parser.add_argument("--json", action="store_true", help="Print JSON output to stdout instead of console table")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout output")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()

    # Isolation rule: when --run-dir or --runs-dir is explicitly supplied, never silently attach
    # legacy default runs/state/objective evidence or a mismatched default validation export.
    explicit_mode = bool(args.run_dirs or args.runs_dir or args.state_json or args.validation_export or args.objective_audit)

    runs_dir = args.runs_dir
    run_dirs = args.run_dirs
    state_json = args.state_json
    val_export = args.validation_export
    obj_audit = args.objective_audit

    if not explicit_mode:
        default_runs = root / "outputs/model/laya-balanced-20261018"
        default_state = root / "outputs/laya-balanced-20261018/tracking/state.json"
        default_val = root / "outputs/laya-full-20261017/export/validation.jsonl"
        default_obj = root / "outputs/laya-diagnostics-20261019/objective-audit.json"
        if default_runs.exists():
            runs_dir = default_runs
        if default_state.exists():
            state_json = default_state
        if default_val.exists():
            val_export = default_val
        if default_obj.exists():
            obj_audit = default_obj
    elif val_export is None and run_dirs:
        # In explicit --run-dir mode, auto-select validation.jsonl ONLY if every specified run resolves
        # to the exact same existing export path declared in its own report/config.
        resolved = {resolve_validation_export_for_run(Path(d), root_dir=root) for d in run_dirs}
        if len(resolved) == 1:
            only = next(iter(resolved))
            if only is not None and only.exists():
                val_export = only

    protected: List[Optional[Path]] = [
        runs_dir,
        state_json,
        val_export,
        obj_audit,
        *(run_dirs or []),
    ]

    if args.output_json:
        try:
            _protect_output_path(args.output_json, protected)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        suite = audit_experiment_suite(
            runs_dir=runs_dir,
            run_dirs=run_dirs,
            state_path=state_json,
            validation_export_path=val_export,
            objective_audit_path=obj_audit,
        )
    except Exception as exc:
        sys.stderr.write(f"Audit execution error: {exc}\n")
        return 1

    output_str = json.dumps(suite, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_str + "\n", encoding="utf-8")

    if not args.quiet:
        if args.json:
            print(output_str)
        else:
            print(format_console_summary(suite))

    return 0 if suite["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
