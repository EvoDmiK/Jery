"""Unit tests for full export audit tool (dev-agy).

Uses isolated temporary fixtures to test verification of full export integrity,
multi-window candidate partition coverage, source row fidelity across train/val/test,
window semantics, card whitelisting, 120-char exact normalized prefix, duplicate detection,
safe CLI output paths without file/symlink overwriting, and comprehensive corruption detection.
Does not depend on ignored root outputs or hardcoded paths for unit test passes.
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
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.review.full_audit import (
    compute_sha256,
    run_full_audit,
)


def create_mock_dataset_and_export(base_dir: Path) -> Tuple[Path, Path]:
    """Create a miniature, self-contained valid 3-split dataset and multi-window export for testing."""
    dataset_dir = base_dir / "dataset"
    export_dir = base_dir / "export"
    dataset_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)

    # 1. tools.json (10 tools to support 10-candidate partitioning)
    tools = []
    for i in range(10):
        tools.append({
            "id": f"mcp__test__tool_{i}",
            "server": "test",
            "name": f"tool_{i}",
            "description": f"Tool {i} description for full audit testing with sufficient length.",
            "input_schema": {"type": "object", "properties": {}},
        })
    catalog = {
        "catalog_version": "test-catalog-v2",
        "tools": tools,
    }
    catalog_path = dataset_dir / "tools.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    # 2. Raw dataset rows: train (multi-window select_tools + unsupported), val (single window clarify), test (single window no_tool)
    train_tools_10 = [f"mcp__test__tool_{i}" for i in range(10)]
    raw_train = [
        {
            "id": "train_case_1",
            "scenario_id": "scenario_train_1",
            "split": "train",
            "request": "문서 목록을 확인해줘.",
            "context": [{"role": "user", "text": "선행 대화 맥락"}],
            "candidate_tools": train_tools_10,
            "decision": "select_tools",
            "acceptable_choices": [{"tool": "mcp__test__tool_0"}],
            "tool_catalog_version": "test-catalog-v2",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
        {
            "id": "train_case_2",
            "scenario_id": "scenario_train_2",
            "split": "train",
            "request": "도메인 외 항공편 예약해줘.",
            "context": [],
            "candidate_tools": ["mcp__test__tool_1", "mcp__test__tool_2"],
            "decision": "unsupported",
            "acceptable_choices": [],
            "tool_catalog_version": "test-catalog-v2",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    ]

    raw_val = [
        {
            "id": "val_case_1",
            "scenario_id": "scenario_val_1",
            "split": "validation",
            "request": "파일을 다른 이름으로 변경해줘.",
            "context": [],
            "candidate_tools": ["mcp__test__tool_3"],
            "decision": "clarify",
            "acceptable_choices": [],
            "tool_catalog_version": "test-catalog-v2",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        }
    ]

    raw_test = [
        {
            "id": "test_case_1",
            "scenario_id": "scenario_test_1",
            "split": "test",
            "request": "안녕 반가워.",
            "context": [],
            "candidate_tools": ["mcp__test__tool_4"],
            "decision": "no_tool",
            "acceptable_choices": [],
            "tool_catalog_version": "test-catalog-v2",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        }
    ]

    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_train), encoding="utf-8"
    )
    (dataset_dir / "validation.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_val), encoding="utf-8"
    )
    (dataset_dir / "test.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_test), encoding="utf-8"
    )

    # 3. Export rows matching raw dataset
    # train_case_1: partitioned into 2 windows
    w0_tools = train_tools_10[:8]
    w0_cards = [
        {"alias": f"T{i}", "server": "test", "name": f"tool_{i}", "summary": f"Tool {i} description for full audit testing with sufficient length."}
        for i in range(8)
    ]
    w0_criteria = {f"T{i}": f"도구 T{i}" for i in range(8)}
    w0_criteria.update({"no_tool": "도구 없이 답변", "clarify": "필수 대상 확인 필요", "window_none": "이 창에 적합한 도구 없음"})

    train_case_1_w0 = {
        "state": {
            "request": "문서 목록을 확인해줘.",
            "context": [{"role": "user", "text": "선행 대화 맥락"}],
            "candidates": w0_cards,
        },
        "questions": {
            "route_in_window": {
                "type": "choice",
                "instructions": "Route instructions.",
                "criteria": w0_criteria,
            }
        },
        "gold": {
            "route_in_window": {
                "label": "T0",
                "probabilities": {"T0": 1.0},
            }
        },
        "metadata": {
            "export_version": "decision-export-v1",
            "export_scope": "exhaustive_original_candidates",
            "export_id": "train_case_1:window:0",
            "source_id": "train_case_1",
            "scenario_id": "scenario_train_1",
            "source_split": "train",
            "tool_catalog_version": "test-catalog-v2",
            "original_candidate_tools": train_tools_10,
            "window_tools": w0_tools,
            "alias_to_tool": {f"T{i}": t for i, t in enumerate(w0_tools)},
            "original_decision": "select_tools",
            "acceptable_choices": [{"tool": "mcp__test__tool_0"}],
            "derived_negative_window": False,
            "window_index": 0,
            "window_count": 2,
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    }

    # Window 1: tools 8..9 (does not contain acceptable tool_0 -> label window_none)
    w1_tools = train_tools_10[8:]
    w1_cards = [
        {"alias": f"T{i}", "server": "test", "name": f"tool_{i+8}", "summary": f"Tool {i+8} description for full audit testing with sufficient length."}
        for i in range(2)
    ]
    w1_criteria = {f"T{i}": f"도구 T{i}" for i in range(2)}
    w1_criteria.update({"no_tool": "도구 없이 답변", "clarify": "필수 대상 확인 필요", "window_none": "이 창에 적합한 도구 없음"})

    train_case_1_w1 = {
        "state": {
            "request": "문서 목록을 확인해줘.",
            "context": [{"role": "user", "text": "선행 대화 맥락"}],
            "candidates": w1_cards,
        },
        "questions": {
            "route_in_window": {
                "type": "choice",
                "instructions": "Route instructions.",
                "criteria": w1_criteria,
            }
        },
        "gold": {
            "route_in_window": {
                "label": "window_none",
                "probabilities": {"window_none": 1.0},
            }
        },
        "metadata": {
            "export_version": "decision-export-v1",
            "export_scope": "exhaustive_original_candidates",
            "export_id": "train_case_1:window:1",
            "source_id": "train_case_1",
            "scenario_id": "scenario_train_1",
            "source_split": "train",
            "tool_catalog_version": "test-catalog-v2",
            "original_candidate_tools": train_tools_10,
            "window_tools": w1_tools,
            "alias_to_tool": {f"T{i}": t for i, t in enumerate(w1_tools)},
            "original_decision": "select_tools",
            "acceptable_choices": [{"tool": "mcp__test__tool_0"}],
            "derived_negative_window": False,
            "window_index": 1,
            "window_count": 2,
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    }

    # train_case_2: unsupported (1 window of 2 tools, label window_none)
    train_case_2_w0 = {
        "state": {
            "request": "도메인 외 항공편 예약해줘.",
            "context": [],
            "candidates": [
                {"alias": "T0", "server": "test", "name": "tool_1", "summary": "Tool 1 description for full audit testing with sufficient length."},
                {"alias": "T1", "server": "test", "name": "tool_2", "summary": "Tool 2 description for full audit testing with sufficient length."},
            ],
        },
        "questions": {
            "route_in_window": {
                "type": "choice",
                "instructions": "Route instructions.",
                "criteria": {"T0": "도구 T0", "T1": "도구 T1", "no_tool": "도구 없음", "clarify": "확인 필요", "window_none": "도구 없음"},
            }
        },
        "gold": {
            "route_in_window": {
                "label": "window_none",
                "probabilities": {"window_none": 1.0},
            }
        },
        "metadata": {
            "export_version": "decision-export-v1",
            "export_scope": "exhaustive_original_candidates",
            "export_id": "train_case_2:window:0",
            "source_id": "train_case_2",
            "scenario_id": "scenario_train_2",
            "source_split": "train",
            "tool_catalog_version": "test-catalog-v2",
            "original_candidate_tools": ["mcp__test__tool_1", "mcp__test__tool_2"],
            "window_tools": ["mcp__test__tool_1", "mcp__test__tool_2"],
            "alias_to_tool": {"T0": "mcp__test__tool_1", "T1": "mcp__test__tool_2"},
            "original_decision": "unsupported",
            "acceptable_choices": [],
            "derived_negative_window": False,
            "window_index": 0,
            "window_count": 1,
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    }

    # val_case_1: clarify (1 window of 1 tool, label clarify)
    val_case_1_w0 = {
        "state": {
            "request": "파일을 다른 이름으로 변경해줘.",
            "context": [],
            "candidates": [
                {"alias": "T0", "server": "test", "name": "tool_3", "summary": "Tool 3 description for full audit testing with sufficient length."}
            ],
        },
        "questions": {
            "route_in_window": {
                "type": "choice",
                "instructions": "Route instructions.",
                "criteria": {"T0": "도구 T0", "no_tool": "도구 없음", "clarify": "확인 필요", "window_none": "도구 없음"},
            }
        },
        "gold": {
            "route_in_window": {
                "label": "clarify",
                "probabilities": {"clarify": 1.0},
            }
        },
        "metadata": {
            "export_version": "decision-export-v1",
            "export_scope": "exhaustive_original_candidates",
            "export_id": "val_case_1:window:0",
            "source_id": "val_case_1",
            "scenario_id": "scenario_val_1",
            "source_split": "validation",
            "tool_catalog_version": "test-catalog-v2",
            "original_candidate_tools": ["mcp__test__tool_3"],
            "window_tools": ["mcp__test__tool_3"],
            "alias_to_tool": {"T0": "mcp__test__tool_3"},
            "original_decision": "clarify",
            "acceptable_choices": [],
            "derived_negative_window": False,
            "window_index": 0,
            "window_count": 1,
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    }

    # test_case_1: no_tool (1 window of 1 tool, label no_tool)
    test_case_1_w0 = {
        "state": {
            "request": "안녕 반가워.",
            "context": [],
            "candidates": [
                {"alias": "T0", "server": "test", "name": "tool_4", "summary": "Tool 4 description for full audit testing with sufficient length."}
            ],
        },
        "questions": {
            "route_in_window": {
                "type": "choice",
                "instructions": "Route instructions.",
                "criteria": {"T0": "도구 T0", "no_tool": "도구 없음", "clarify": "확인 필요", "window_none": "도구 없음"},
            }
        },
        "gold": {
            "route_in_window": {
                "label": "no_tool",
                "probabilities": {"no_tool": 1.0},
            }
        },
        "metadata": {
            "export_version": "decision-export-v1",
            "export_scope": "exhaustive_original_candidates",
            "export_id": "test_case_1:window:0",
            "source_id": "test_case_1",
            "scenario_id": "scenario_test_1",
            "source_split": "test",
            "tool_catalog_version": "test-catalog-v2",
            "original_candidate_tools": ["mcp__test__tool_4"],
            "window_tools": ["mcp__test__tool_4"],
            "alias_to_tool": {"T0": "mcp__test__tool_4"},
            "original_decision": "no_tool",
            "acceptable_choices": [],
            "derived_negative_window": False,
            "window_index": 0,
            "window_count": 1,
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    }

    train_rows = [train_case_1_w0, train_case_1_w1, train_case_2_w0]
    val_rows = [val_case_1_w0]
    test_rows = [test_case_1_w0]

    train_content = "".join(json.dumps(r) + "\n" for r in train_rows)
    val_content = "".join(json.dumps(r) + "\n" for r in val_rows)
    test_content = "".join(json.dumps(r) + "\n" for r in test_rows)

    (export_dir / "train.jsonl").write_text(train_content, encoding="utf-8")
    (export_dir / "validation.jsonl").write_text(val_content, encoding="utf-8")
    (export_dir / "test.jsonl").write_text(test_content, encoding="utf-8")

    manifest = {
        "export_version": "decision-export-v1",
        "purpose": "full_synthetic_exploratory",
        "export_scope": "exhaustive_original_candidates",
        "seed": 3407,
        "dataset_version": "mcp-routing-v2",
        "catalog_version": "test-catalog-v2",
        "source_sha256": {
            "train.jsonl": compute_sha256(dataset_dir / "train.jsonl"),
            "validation.jsonl": compute_sha256(dataset_dir / "validation.jsonl"),
            "test.jsonl": compute_sha256(dataset_dir / "test.jsonl"),
            "tools.json": compute_sha256(dataset_dir / "tools.json"),
        },
        "splits": {
            "train": {
                "source_rows": 2,
                "rows": 3,
                "scenario_groups": 2,
                "sha256": compute_sha256(export_dir / "train.jsonl"),
                "original_decisions": {"select_tools": 2, "unsupported": 1},
                "window_labels": {"T0": 1, "window_none": 2},
                "candidate_coverage": 1.0,
            },
            "validation": {
                "source_rows": 1,
                "rows": 1,
                "scenario_groups": 1,
                "sha256": compute_sha256(export_dir / "validation.jsonl"),
                "original_decisions": {"clarify": 1},
                "window_labels": {"clarify": 1},
                "candidate_coverage": 1.0,
            },
            "test": {
                "source_rows": 1,
                "rows": 1,
                "scenario_groups": 1,
                "sha256": compute_sha256(export_dir / "test.jsonl"),
                "original_decisions": {"no_tool": 1},
                "window_labels": {"no_tool": 1},
                "candidate_coverage": 1.0,
            },
        },
        "max_window_tools": 8,
        "state_card_summary_characters": 120,
        "action_prediction": False,
        "limitations": ["Test fixture only"],
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return dataset_dir, export_dir


class TestFullAudit(unittest.TestCase):
    """Test suite for independent full export audit tool."""

    def test_clean_temp_fixture_passes(self):
        """Verify that an independent temporary fixture passes full audit completely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            result = run_full_audit(exp_dir, ds_dir)

            self.assertTrue(result["audit_passed"])
            self.assertEqual(result["error_count"], 0)
            self.assertEqual(result["all_errors"], [])
            self.assertTrue(result["source_checksums"]["verified"])
            self.assertTrue(result["export_integrity"]["valid"])

            # Verify candidate coverage stats
            train_cov = result["split_audits"]["train"]["candidate_coverage"]
            self.assertEqual(train_cov["total_source_cases"], 2)
            self.assertEqual(train_cov["full_coverage_cases"], 2)
            self.assertEqual(train_cov["candidate_coverage_ratio"], 1.0)
            self.assertEqual(train_cov["window_count_distribution"], {2: 1, 1: 1})

    def test_missing_source_hash_key_detected(self):
        """Ensure missing test.jsonl or tools.json in source_sha256 is detected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            del manifest["source_sha256"]["test.jsonl"]
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("Missing required source hash key" in e and "test.jsonl" in e for e in result["all_errors"]))

    def test_split_sha256_mismatch_detected(self):
        """Ensure tampered split file sha256 mismatch is detected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            with open(exp_dir / "train.jsonl", "a", encoding="utf-8") as f:
                f.write("\n")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("train.jsonl sha256 mismatch" in e for e in result["all_errors"]))

    def test_scenario_leakage_between_train_and_test_detected(self):
        """Ensure scenario group leakage between train and test splits is caught."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            test_lines = (exp_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(test_lines[0])
            row["metadata"]["scenario_id"] = "scenario_train_1"
            (exp_dir / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["test"]["sha256"] = compute_sha256(exp_dir / "test.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("Scenario group leakage between train and test" in e for e in result["all_errors"]))

    def test_source_fidelity_tampering_detected(self):
        """Ensure modified state.request not matching raw dataset is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            row["state"]["request"] = "변조된 요청 텍스트입니다."
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("state.request differs from raw source request" in e for e in result["all_errors"]))

    def test_missing_partition_window_detected(self):
        """Ensure that if a source row's window is omitted, candidate partition audit fails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            train_lines = [l for l in train_lines if "train_case_1:window:1" not in l]
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["rows"] = len(train_lines)
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("actual window count (1) != declared window_count (2)" in e or "candidates missing from windows" in e for e in result["all_errors"]))

    def test_fabricated_candidate_tool_in_window_detected(self):
        """Ensure fabricated tool not in original candidates C is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            row["metadata"]["window_tools"][0] = "mcp__fake__tool_x"
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("fabricated tools in windows" in e or "fabricated tools not in C" in e for e in result["all_errors"]))

    def test_window_none_label_violation_detected(self):
        """Ensure window lacking acceptable tool must have label window_none."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[1])  # window 1 (absent gold)
            row["gold"]["route_in_window"]["label"] = "T0"  # Invalid! Gold is not in window 1
            train_lines[1] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("absent gold in window must have label window_none" in e for e in result["all_errors"]))

    def test_card_summary_length_limit_detected(self):
        """Ensure card summary exceeding 120 characters is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            row["state"]["candidates"][0]["summary"] = "A" * 121
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("exceeds 120 limit" in e for e in result["all_errors"]))

    def test_card_missing_required_key_detected(self):
        """Ensure candidate cards missing any of the 4 required keys (e.g. missing summary) are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            # Delete summary from card 0
            del row["state"]["candidates"][0]["summary"]
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("card keys" in e and "!= expected" in e for e in result["all_errors"]))

    def test_card_empty_or_mismatched_summary_detected(self):
        """Ensure empty card summary or unnormalized summary not matching catalog description is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            # Set summary to empty string
            row["state"]["candidates"][0]["summary"] = ""
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("!= expected normalized description prefix" in e for e in result["all_errors"]))

    def test_tool_id_leakage_detected(self):
        """Ensure full tool ID leaked into request or context is caught."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            raw_train = [json.loads(l) for l in (ds_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            raw_train[0]["request"] = "mcp__test__tool_0 호출해주세요."
            (ds_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_train), encoding="utf-8")

            train_lines = (exp_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(train_lines[0])
            row["state"]["request"] = "mcp__test__tool_0 호출해주세요."
            train_lines[0] = json.dumps(row)
            (exp_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["source_sha256"]["train.jsonl"] = compute_sha256(ds_dir / "train.jsonl")
            manifest["splits"]["train"]["sha256"] = compute_sha256(exp_dir / "train.jsonl")
            (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = run_full_audit(exp_dir, ds_dir)
            self.assertFalse(result["audit_passed"])
            self.assertTrue(any("leaked in request/context" in e for e in result["all_errors"]))

    def test_cli_invocation_on_temp_fixture(self):
        """Verify CLI execution via subprocess on temporary fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            out_json = Path(tmp_dir) / "audit_result.json"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "apps/review/full_audit.py"),
                "--export-dir",
                str(exp_dir),
                "--dataset-dir",
                str(ds_dir),
                "--output-json",
                str(out_json),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Audit status: PASS", proc.stdout)
            self.assertIn("Total errors: 0", proc.stdout)
            self.assertTrue(out_json.exists())

            saved_data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(saved_data["audit_passed"])

    def test_cli_refuses_to_overwrite_dataset_or_export_files(self):
        """Ensure CLI refuses --output-json pointing to dataset files, export files, or symlinks to them."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir, exp_dir = create_mock_dataset_and_export(Path(tmp_dir))
            script_path = str(PROJECT_ROOT / "apps/review/full_audit.py")

            # 1. Target inside dataset directory (e.g. ds_dir / "train.jsonl")
            target_ds = ds_dir / "train.jsonl"
            before_bytes_ds = target_ds.read_bytes()
            proc_ds = subprocess.run(
                [sys.executable, script_path, "--export-dir", str(exp_dir), "--dataset-dir", str(ds_dir), "--output-json", str(target_ds)],
                capture_output=True, text=True
            )
            self.assertNotEqual(proc_ds.returncode, 0)
            self.assertIn("Refusing", proc_ds.stderr)
            self.assertEqual(target_ds.read_bytes(), before_bytes_ds)

            # 2. Target inside export directory (e.g. exp_dir / "train.jsonl")
            target_exp = exp_dir / "train.jsonl"
            before_bytes_exp = target_exp.read_bytes()
            proc_exp = subprocess.run(
                [sys.executable, script_path, "--export-dir", str(exp_dir), "--dataset-dir", str(ds_dir), "--output-json", str(target_exp)],
                capture_output=True, text=True
            )
            self.assertNotEqual(proc_exp.returncode, 0)
            self.assertIn("Refusing", proc_exp.stderr)
            self.assertEqual(target_exp.read_bytes(), before_bytes_exp)

            # 3. Symlink pointing to export file
            symlink_target = Path(tmp_dir) / "symlink_train.jsonl"
            os.symlink(target_exp, symlink_target)
            proc_sym = subprocess.run(
                [sys.executable, script_path, "--export-dir", str(exp_dir), "--dataset-dir", str(ds_dir), "--output-json", str(symlink_target)],
                capture_output=True, text=True
            )
            self.assertNotEqual(proc_sym.returncode, 0)
            self.assertEqual(target_exp.read_bytes(), before_bytes_exp)

    def test_live_root_full_export_if_available(self):
        """Verify live full export if present in workspace without hardcoded absolute fallback."""
        target_export = PROJECT_ROOT / "outputs/laya-full-20261017/export"
        target_dataset = PROJECT_ROOT / "datasets/mcp-routing-v2"

        if not (target_export / "manifest.json").exists() or not (target_dataset / "tools.json").exists():
            self.skipTest("Live full export not found in workspace; skipping live check cleanly.")

        result = run_full_audit(target_export, target_dataset)
        self.assertTrue(result["audit_passed"], f"Live full export audit failed: {result['all_errors'][:5]}")
        self.assertEqual(result["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
