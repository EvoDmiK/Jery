"""Unit tests for apps/review/routing_v2_audit.py.

Verifies:
- Rejection of inconsistent selected/best-trained epochs or search_score in laya-routing-v2 reports.
- Preservation of all input files (byte-for-byte immutability and output path protection against aliases/symlinks/hardlinks).
- Tolerance of legacy reports with absent v2 fields and explicit 'legacy' status (never reinterpreted as v2).
- Ranking strictly by validation/search_score (never test metrics).
- Constant negative-window (always-window_none) control calculation and misleading aggregate window accuracy flagging.
- Detection of equal/floored Hyperopt search objectives vs valid identical serving models and premature baseline-based early stops.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.review.routing_v2_audit import (
    audit_experiment_suite,
    audit_single_run,
    audit_tracking_state,
    compute_validation_export_control,
    rank_runs_by_validation,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_validation_block(
    case_acc: float,
    tool_win_acc: float,
    none_win_acc: float,
    overall_win_acc: float,
    unnecessary_call_rate: float = 0.125,
    no_tool_recall: float = 0.25,
    clarify_recall: float = 0.25,
    epoch: int | None = None,
) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "case_accuracy": case_acc,
        "decision_macro_f1": 0.25,
        "coverage": 0.95,
        "case": {
            "cases": 248,
            "case_accuracy": case_acc,
            "decision_accuracy": 0.5,
            "decision_macro_f1": 0.25,
            "tool_accuracy": case_acc,
            "unnecessary_call_rate": unnecessary_call_rate,
            "out_of_candidate_selections": 0,
            "delegated": 12,
            "coverage": 0.95,
            "per_decision": {
                "select_tools": {"precision": 0.95, "recall": case_acc, "f1": 0.5, "support": 232},
                "no_tool": {"precision": 0.5, "recall": no_tool_recall, "f1": 0.33, "support": 4},
                "clarify": {"precision": 0.5, "recall": clarify_recall, "f1": 0.33, "support": 4},
                "unsupported": {"precision": 0.1, "recall": 0.5, "f1": 0.16, "support": 8},
            },
        },
        "window": {
            "windows": 1592,
            "any_of_accuracy": overall_win_acc,
            "soft_ce": 1.1,
            "by_gold_kind": {
                "tool": {"correct": int(round(tool_win_acc * 235)), "windows": 235, "accuracy": tool_win_acc},
                "window_none": {"correct": int(round(none_win_acc * 1305)), "windows": 1305, "accuracy": none_win_acc},
                "no_tool": {"correct": 6, "windows": 26, "accuracy": 6 / 26},
                "clarify": {"correct": 6, "windows": 26, "accuracy": 6 / 26},
            },
        },
    }
    if epoch is not None:
        block["epoch"] = epoch
    return block


def create_v2_run_fixture(
    run_dir: Path,
    run_id: str = "trial1-w0.5",
    w_none_weight: float = 0.5,
    head_lr: float = 4.87975e-5,
    epoch_case_accs: List[float] | None = None,
    baseline_case_acc: float = 0.1613,
    test_case_acc: float = 0.1504,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Create a valid laya-routing-v2 run directory with report.json and metrics.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if epoch_case_accs is None:
        epoch_case_accs = [0.0484, 0.0605, 0.0887, 0.1129, 0.1290, 0.1452, 0.1532, 0.1573, 0.1492, 0.1411]

    metrics_rows: List[Dict[str, Any]] = []
    best_trained_ep = 1
    best_trained_score = -1.0
    for idx, acc in enumerate(epoch_case_accs, 1):
        if acc > best_trained_score:
            best_trained_score = acc
            best_trained_ep = idx
        metrics_rows.append({
            "epoch": idx,
            "train_loss": round(2.0 - idx * 0.08, 4),
            "steps": idx * 2133,
            "train_seconds": idx * 250.0,
            "validation": _make_validation_block(
                case_acc=acc,
                tool_win_acc=min(1.0, acc + 0.05),
                none_win_acc=0.85,
                overall_win_acc=0.74,
            ),
            "time": f"2026-10-19T05:{idx:02d}:00+00:00",
        })

    completed_epochs = len(epoch_case_accs)
    serving_ep = 0 if baseline_case_acc >= best_trained_score else best_trained_ep
    serving_score = baseline_case_acc if serving_ep == 0 else best_trained_score
    trained_bad_epochs = completed_epochs - best_trained_ep

    report: Dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "training_policy_version": "laya-routing-v2",
            "window_none_loss_weight": w_none_weight,
            "head_learning_rate": head_lr,
            "weight_decay": 0.01,
            "label_smoothing": 0.0,
            "seed": 3407,
            "epochs": 10,
            "validation_callback": {
                "monitor": "validation.case_accuracy",
                "min_epochs": 10,
                "patience": 5,
                "min_delta": 0.004,
                "include_baseline": True,
            },
        },
        "before_validation": _make_validation_block(
            case_acc=baseline_case_acc,
            tool_win_acc=0.2681,
            none_win_acc=0.0398,
            overall_win_acc=0.0879,
            unnecessary_call_rate=0.625,
        ),
        "selection": {
            "policy_version": "laya-routing-v2",
            "monitor": "validation.case_accuracy",
            "mode": "max",
            "min_epochs": 10,
            "patience": 5,
            "min_delta": 0.004,
            "best_epoch": serving_ep,
            "best_score": serving_score,
            "selected_epoch": serving_ep,
            "best_trained_epoch": best_trained_ep,
            "best_trained_score": best_trained_score,
            "search_score": best_trained_score,
            "bad_epochs": completed_epochs if serving_ep == 0 else trained_bad_epochs,
            "trained_bad_epochs": trained_bad_epochs,
            "stopped": False,
            "completed_epochs": completed_epochs,
            "max_epochs": 10,
            "reason": "max_epochs",
        },
        "best_trained": {
            "artifact": f"models/laya-routing-v2/{run_id}/best-trained/trainable.safetensors",
            "artifact_sha256": "a" * 64,
            "epoch": best_trained_ep,
            "validation_case_accuracy": best_trained_score,
            "format": "unsloth-torch-laya-routing-v1",
        },
        "last_trained": {
            "artifact": f"models/laya-routing-v2/{run_id}/last-trained/trainable.safetensors",
            "artifact_sha256": "b" * 64,
            "epoch": completed_epochs,
            "validation_case_accuracy": epoch_case_accs[-1],
            "format": "unsloth-torch-laya-routing-v1",
        },
        "restore": {
            "artifact": f"models/laya-routing-v2/{run_id}/final/trainable.safetensors",
            "artifact_sha256": "c" * 64,
            "artifact_bytes": 59086748,
            "format": "unsloth-torch-laya-routing-v1",
            "same_validation_case_predictions": True,
            "validation_max_abs_probability_diff": 0.0,
            "encoder_unchanged": True,
        },
        "validation": _make_validation_block(
            case_acc=serving_score,
            tool_win_acc=0.2681 if serving_ep == 0 else min(1.0, serving_score + 0.05),
            none_win_acc=0.0398 if serving_ep == 0 else 0.85,
            overall_win_acc=0.0879 if serving_ep == 0 else 0.74,
            unnecessary_call_rate=0.625 if serving_ep == 0 else 0.125,
            epoch=serving_ep,
        ),
        "test": _make_validation_block(
            case_acc=test_case_acc,
            tool_win_acc=0.2672,
            none_win_acc=0.0302,
            overall_win_acc=0.0672,
            unnecessary_call_rate=0.5625,
            epoch=serving_ep,
        ),
    }

    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in metrics_rows),
        encoding="utf-8",
    )
    return report, metrics_rows


class TestRoutingV2Audit(unittest.TestCase):
    def test_valid_v2_report_passes_and_distinguishes_all_four_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "trial1-w0.5"
            create_v2_run_fixture(
                run_dir,
                epoch_case_accs=[0.0484, 0.0605, 0.0887, 0.1129, 0.1290, 0.1452, 0.1532, 0.1573, 0.1492, 0.1411],
                baseline_case_acc=0.1613,
            )
            res = audit_single_run(run_dir)
            self.assertTrue(res["audit_passed"], msg=f"Unexpected errors: {res['errors']}")
            self.assertEqual(res["contract_mode"], "laya-routing-v2")
            eo = res["epochs_overview"]
            self.assertEqual(eo["baseline_epoch"], 0)
            self.assertEqual(eo["best_trained_epoch"], 8)
            self.assertEqual(eo["serving_selected_epoch"], 0)
            self.assertEqual(eo["last_trained_epoch"], 10)
            self.assertEqual(eo["completed_epochs"], 10)
            self.assertTrue(eo["baseline_retained_for_serving"])
            self.assertIn("BASELINE_RETAINED_FOR_SERVING", res["diagnostic_flags"])

            sc = res["scores_summary"]
            self.assertAlmostEqual(sc["baseline_epoch0_validation_case_accuracy"], 0.1613)
            self.assertAlmostEqual(sc["best_trained_validation_case_accuracy"], 0.1573)
            self.assertAlmostEqual(sc["search_score"], 0.1573)
            self.assertAlmostEqual(sc["serving_selected_validation_case_accuracy"], 0.1613)
            self.assertAlmostEqual(sc["last_trained_validation_case_accuracy"], 0.1411)

    def test_rejects_inconsistent_search_score_or_best_trained_or_selected_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)

            # Case 1: search_score != best_trained_score (e.g., clipped to serving baseline 0.1613)
            r1 = base_dir / "bad_search_score"
            report, _ = create_v2_run_fixture(r1)
            report["selection"]["search_score"] = 0.1613  # Clipped to baseline instead of 0.1573!
            (r1 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res1 = audit_single_run(r1)
            self.assertFalse(res1["audit_passed"])
            self.assertTrue(any("search_score" in e for e in res1["errors"]))

            # Case 2: best_trained_epoch is 0 (baseline instead of trained epoch >= 1)
            r2 = base_dir / "bad_best_trained_ep0"
            report, _ = create_v2_run_fixture(r2)
            report["selection"]["best_trained_epoch"] = 0
            report["best_trained"]["epoch"] = 0
            (r2 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res2 = audit_single_run(r2)
            self.assertFalse(res2["audit_passed"])
            self.assertTrue(any("best_trained_epoch" in e for e in res2["errors"]))

            # Case 3: best_trained_score does not match max in metrics.jsonl
            r3 = base_dir / "bad_best_trained_mismatch_history"
            report, _ = create_v2_run_fixture(r3)
            report["selection"]["best_trained_score"] = 0.1400
            report["selection"]["search_score"] = 0.1400
            report["best_trained"]["validation_case_accuracy"] = 0.1400
            (r3 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res3 = audit_single_run(r3)
            self.assertFalse(res3["audit_passed"])
            self.assertTrue(any("metrics.jsonl" in e for e in res3["errors"]))

            # Case 4: report.best_trained descriptor disagrees with selection.best_trained_epoch
            r4 = base_dir / "bad_best_trained_descriptor"
            report, _ = create_v2_run_fixture(r4)
            report["best_trained"]["epoch"] = 5
            (r4 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res4 = audit_single_run(r4)
            self.assertFalse(res4["audit_passed"])
            self.assertTrue(any("report.best_trained.epoch" in e for e in res4["errors"]))

            # Case 5: report.last_trained descriptor disagrees with completed_epochs
            r5 = base_dir / "bad_last_trained_descriptor"
            report, _ = create_v2_run_fixture(r5)
            report["last_trained"]["epoch"] = 8
            (r5 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res5 = audit_single_run(r5)
            self.assertFalse(res5["audit_passed"])
            self.assertTrue(any("report.last_trained.epoch" in e for e in res5["errors"]))

            # Case 6: validation.epoch != test.epoch or != selected_epoch
            r6 = base_dir / "bad_selected_epoch_mismatch"
            report, _ = create_v2_run_fixture(r6)
            report["test"]["epoch"] = 8  # selected_epoch is 0, but test evaluated epoch 8
            (r6 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res6 = audit_single_run(r6)
            self.assertFalse(res6["audit_passed"])
            self.assertTrue(any("report.test.epoch" in e for e in res6["errors"]))

            # Case 7: invalid window_none_loss_weight outside (0, 1]
            r7 = base_dir / "bad_w_none_weight"
            report, _ = create_v2_run_fixture(r7, w_none_weight=0.0)
            (r7 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res7 = audit_single_run(r7)
            self.assertFalse(res7["audit_passed"])
            self.assertTrue(any("window_none_loss_weight" in e for e in res7["errors"]))

            # Case 8: premature early stopping before min_epochs=10 in v2
            r8 = base_dir / "bad_v2_early_stop"
            report, _ = create_v2_run_fixture(r8, epoch_case_accs=[0.03, 0.08, 0.09, 0.11, 0.15])
            report["selection"]["stopped"] = True
            report["selection"]["reason"] = "early_stopping"
            (r8 / "report.json").write_text(json.dumps(report), encoding="utf-8")
            res8 = audit_single_run(r8)
            self.assertFalse(res8["audit_passed"])
            self.assertTrue(any("before min_epochs" in e for e in res8["errors"]))

    def test_tolerates_legacy_reports_with_explicit_legacy_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "legacy-trial4"
            report, _ = create_v2_run_fixture(
                run_dir,
                epoch_case_accs=[0.0323, 0.0806, 0.0887, 0.1169, 0.1573],
                baseline_case_acc=0.1613,
            )
            # Strip all v2 fields to match legacy report structure
            del report["config"]["training_policy_version"]
            del report["config"]["window_none_loss_weight"]
            del report["config"]["validation_callback"]["min_epochs"]
            for k in ("policy_version", "best_trained_epoch", "best_trained_score", "search_score", "trained_bad_epochs", "min_epochs"):
                report["selection"].pop(k, None)
            report["selection"]["stopped"] = True
            report["selection"]["reason"] = "early_stopping"
            del report["best_trained"]
            del report["last_trained"]
            (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

            res = audit_single_run(run_dir)
            self.assertTrue(res["audit_passed"], msg=f"Legacy run should pass with legacy status: {res['errors']}")
            self.assertEqual(res["contract_mode"], "legacy")
            self.assertEqual(res["epochs_overview"]["best_trained_epoch"], 5)
            self.assertAlmostEqual(res["scores_summary"]["best_trained_validation_case_accuracy"], 0.1573)
            self.assertIsNone(res["scores_summary"]["search_score"])
            self.assertIn("PREMATURE_BASELINE_EARLY_STOP", res["diagnostic_flags"])
            self.assertIn("STOPPED_DURING_TRAINED_RECOVERY", res["diagnostic_flags"])

    def test_ranks_strictly_by_validation_search_score_never_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Run A has LOWER validation search_score (0.1400) but HIGHER test_case_acc (0.9900)
            dir_a = base / "run_a_high_test_low_val"
            create_v2_run_fixture(
                dir_a,
                run_id="run_a",
                w_none_weight=1.0,
                epoch_case_accs=[0.10, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                test_case_acc=0.9900,
            )
            # Run B has HIGHER validation search_score (0.1800) but LOWER test_case_acc (0.0100)
            dir_b = base / "run_b_high_val_low_test"
            create_v2_run_fixture(
                dir_b,
                run_id="run_b",
                w_none_weight=0.25,
                epoch_case_accs=[0.10, 0.12, 0.15, 0.18, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17],
                test_case_acc=0.0100,
            )

            audit_a = audit_single_run(dir_a)
            audit_b = audit_single_run(dir_b)
            ranked = rank_runs_by_validation([audit_a, audit_b])

            self.assertEqual(ranked[0]["run_id"], "run_b")
            self.assertEqual(ranked[1]["run_id"], "run_a")
            self.assertAlmostEqual(ranked[0]["ranking_validation_score"], 0.1800)
            self.assertAlmostEqual(ranked[1]["ranking_validation_score"], 0.1400)
            self.assertFalse(ranked[0]["test_used_for_ranking"])
            self.assertFalse(ranked[1]["test_used_for_ranking"])

    def test_constant_negative_window_control_and_input_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "runs" / "trial1"
            create_v2_run_fixture(run_dir)

            # Create a mini validation.jsonl export with 2 cases (1 select_tools across 2 windows, 1 unsupported single window)
            val_export = base / "export" / "validation.jsonl"
            val_export.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "metadata": {
                        "source_id": "c1",
                        "window_index": 0,
                        "window_count": 2,
                        "export_id": "c1_w0",
                        "window_tools": ["mcp__s__t1"],
                        "alias_to_tool": {"T0": "mcp__s__t1"},
                        "original_candidate_tools": ["mcp__s__t1", "mcp__s__t2"],
                        "original_decision": "select_tools",
                        "acceptable_choices": [{"tool": "mcp__s__t1"}],
                    },
                    "gold": {"route_in_window": {"label": "T0", "probabilities": {"T0": 1.0, "window_none": 0.0}}},
                },
                {
                    "metadata": {
                        "source_id": "c1",
                        "window_index": 1,
                        "window_count": 2,
                        "export_id": "c1_w1",
                        "window_tools": ["mcp__s__t2"],
                        "alias_to_tool": {"T0": "mcp__s__t2"},
                        "original_candidate_tools": ["mcp__s__t1", "mcp__s__t2"],
                        "original_decision": "select_tools",
                        "acceptable_choices": [{"tool": "mcp__s__t1"}],
                    },
                    "gold": {"route_in_window": {"label": "window_none", "probabilities": {"T0": 0.0, "window_none": 1.0}}},
                },
                {
                    "metadata": {
                        "source_id": "c2",
                        "window_index": 0,
                        "window_count": 1,
                        "export_id": "c2_w0",
                        "window_tools": ["mcp__s__t3"],
                        "alias_to_tool": {"T0": "mcp__s__t3"},
                        "original_candidate_tools": ["mcp__s__t3"],
                        "original_decision": "unsupported",
                        "acceptable_choices": [],
                    },
                    "gold": {"route_in_window": {"label": "window_none", "probabilities": {"T0": 0.0, "window_none": 1.0}}},
                },
            ]
            val_export.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            ctrl = compute_validation_export_control(val_export)
            self.assertEqual(ctrl["windows"], 3)
            self.assertEqual(ctrl["cases"], 2)
            self.assertEqual(ctrl["negative_windows"], 2)
            self.assertAlmostEqual(ctrl["always_window_none_control"]["window_accuracy"], 2 / 3)
            self.assertAlmostEqual(ctrl["always_window_none_control"]["case_accuracy"], 0.5)

            # Verify CLI refuses to overwrite input files or symlinks/hardlinks
            tracked_files = [run_dir / "report.json", run_dir / "metrics.jsonl", val_export]
            before_bytes = {p: p.read_bytes() for p in tracked_files}

            symlink_target = base / "symlink_out.json"
            hardlink_target = base / "hardlink_out.json"
            symlink_target.symlink_to(val_export)
            os.link(run_dir / "report.json", hardlink_target)

            for bad_out in (val_export, run_dir / "report.json", symlink_target, hardlink_target):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "apps/review/routing_v2_audit.py"),
                        "--runs-dir",
                        str(base / "runs"),
                        "--validation-export",
                        str(val_export),
                        "--output-json",
                        str(bad_out),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0, msg=f"Should reject output path {bad_out}")
                self.assertEqual({p: p.read_bytes() for p in tracked_files}, before_bytes)

    def test_detects_floored_hyperopt_objectives_vs_valid_serving_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            r1 = base / "trial1"
            r2 = base / "trial2"
            create_v2_run_fixture(r1, run_id="trial1", epoch_case_accs=[0.04, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
            create_v2_run_fixture(r2, run_id="trial2", epoch_case_accs=[0.04, 0.08, 0.11, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
            audits = {"trial1": audit_single_run(r1), "trial2": audit_single_run(r2)}

            # Legacy state where both trials had final_validation=0.1613 (baseline-floored)
            legacy_state_path = base / "legacy_state.json"
            legacy_state = {
                "parent_run_id": "parent1",
                "plan": {"training_policy_version": "laya-balanced-v1"},
                "children": [
                    {"index": 1, "local_id": "trial1", "status": "finished", "selected_epoch": 0, "final_validation": 0.1613},
                    {"index": 2, "local_id": "trial2", "status": "finished", "selected_epoch": 0, "final_validation": 0.1613},
                ],
            }
            legacy_state_path.write_text(json.dumps(legacy_state), encoding="utf-8")
            st_legacy = audit_tracking_state(legacy_state_path, run_audits_by_id=audits)
            self.assertTrue(st_legacy["audit_passed"])
            self.assertTrue(st_legacy["floored_search_objective_conceals_trained_variance"])
            self.assertTrue(st_legacy["valid_identical_serving_models"])
            self.assertIn("BASELINE_FLOORED_HYPEROPT_OBJECTIVE_CONCEALS_TRAINED_DIFFERENCES", st_legacy["diagnostic_flags"])

            # V2 state where search_validation was mistakenly floored to 0.1613 -> MUST fail v2 audit!
            v2_bad_state_path = base / "v2_bad_state.json"
            v2_bad_state = {
                "parent_run_id": "parent2",
                "plan": {"training_policy_version": "laya-routing-v2"},
                "children": [
                    {"index": 1, "local_id": "trial1", "status": "finished", "selected_epoch": 0, "final_validation": 0.1613, "search_validation": 0.1613},
                    {"index": 2, "local_id": "trial2", "status": "finished", "selected_epoch": 0, "final_validation": 0.1613, "search_validation": 0.1613},
                ],
            }
            v2_bad_state_path.write_text(json.dumps(v2_bad_state), encoding="utf-8")
            st_v2_bad = audit_tracking_state(v2_bad_state_path, run_audits_by_id=audits)
            self.assertFalse(st_v2_bad["audit_passed"])

    def test_live_integration_assets_if_available(self) -> None:
        """Verify read-only integration assets at /Volumes/Dove-Nest-SSD/projects/Jery/outputs if present."""
        val_path = Path("/Volumes/Dove-Nest-SSD/projects/Jery/outputs/laya-full-20261017/export/validation.jsonl")
        state_path = Path("/Volumes/Dove-Nest-SSD/projects/Jery/outputs/laya-balanced-20261018/tracking/state.json")
        runs_dir = Path("/Volumes/Dove-Nest-SSD/projects/Jery/outputs/model/laya-balanced-20261018")
        obj_path = Path("/Volumes/Dove-Nest-SSD/projects/Jery/outputs/laya-diagnostics-20261019/objective-audit.json")
        if not (val_path.exists() and state_path.exists() and runs_dir.exists() and obj_path.exists()):
            self.skipTest("Live integration assets not present on this machine")

        before_hashes = {
            p: _sha256_bytes(p.read_bytes())
            for p in [val_path, state_path, obj_path, *runs_dir.glob("*/*.json"), *runs_dir.glob("*/*.jsonl")]
        }
        suite = audit_experiment_suite(
            runs_dir=runs_dir,
            state_path=state_path,
            validation_export_path=val_path,
            objective_audit_path=obj_path,
        )
        after_hashes = {p: _sha256_bytes(p.read_bytes()) for p in before_hashes}
        self.assertEqual(before_hashes, after_hashes, "Read-only integration assets must not be modified")
        self.assertTrue(suite["audit_passed"])

        vc = suite["validation_export_control"]
        self.assertEqual(vc["cases"], 248)
        self.assertEqual(vc["select_tool_cases"], 232)
        self.assertEqual(vc["windows"], 1592)
        self.assertEqual(vc["negative_windows"], 1305)
        self.assertAlmostEqual(vc["always_window_none_control"]["window_accuracy"], 1305 / 1592, places=6)
        self.assertAlmostEqual(vc["always_window_none_control"]["case_accuracy"], 0.0323, places=4)

        runs_by_id = {r["run_id"]: r for r in suite["runs"]}
        t1 = runs_by_id["trial1-lr4.87975e-05-9325d1ca"]
        self.assertAlmostEqual(t1["models"]["baseline_epoch0"]["positive_tool_window_recall"], 63 / 235, places=4)
        self.assertAlmostEqual(t1["epoch_trajectory"][0]["positive_tool_window_recall"], 4 / 235, places=4)
        self.assertAlmostEqual(t1["epoch_trajectory"][0]["overall_window_accuracy"], 1257 / 1592, places=4)
        self.assertAlmostEqual(t1["epoch_trajectory"][0]["case_accuracy"], 0.0484, places=4)

        t4 = runs_by_id["trial4-lr0.000665576-9325d1ca"]
        self.assertEqual(
            [e["case_accuracy"] for e in t4["epoch_trajectory"]],
            [0.0323, 0.0806, 0.0887, 0.1169, 0.1573],
        )
        self.assertTrue(t4["epochs_overview"]["premature_baseline_early_stop"])
        self.assertTrue(t4["epochs_overview"]["monotonic_trained_recovery"])


if __name__ == "__main__":
    unittest.main()
