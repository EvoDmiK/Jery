"""Unit tests for pilot export audit tool (dev-agy).

Uses isolated temporary fixtures to test verification of export integrity,
source row fidelity, card whitelisting, 120-char limit, duplicate detection,
and corruption detection (tampered source_id, tampered context, missing checksum keys).
Does not depend on ignored root outputs or hardcoded Mac paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.review.pilot_audit import (
    compute_sha256,
    run_full_pilot_audit,
)


def create_mock_dataset_and_export(base_dir: Path) -> Tuple[Path, Path]:
    """Create a miniature, self-contained valid dataset and export for testing."""
    dataset_dir = base_dir / "dataset"
    export_dir = base_dir / "export"
    dataset_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)

    # 1. tools.json
    catalog = {
        "catalog_version": "test-catalog-v1",
        "tools": [
            {
                "id": "mcp__test__tool_a",
                "server": "test",
                "name": "tool_a",
                "description": "Tool A description for unit testing with sufficient length.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "id": "mcp__test__tool_b",
                "server": "test",
                "name": "tool_b",
                "description": "Tool B description for unit testing with sufficient length.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "id": "mcp__test__tool_c",
                "server": "test",
                "name": "tool_c",
                "description": "Tool C description for unit testing with sufficient length.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }
    (dataset_dir / "tools.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    # 2. Raw dataset rows
    raw_train = [
        {
            "id": "train_row_1",
            "scenario_id": "scenario_train_1",
            "split": "train",
            "request": "도구 A를 실행해줘.",
            "context": [{"role": "user", "text": "선행 대화 맥락"}],
            "candidate_tools": ["mcp__test__tool_a", "mcp__test__tool_b"],
            "decision": "select_tools",
            "acceptable_choices": [{"tool": "mcp__test__tool_a"}],
            "tool_catalog_version": "test-catalog-v1",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
        {
            "id": "train_row_2",
            "scenario_id": "scenario_train_2",
            "split": "train",
            "request": "도메인 외 비행기 표 예약해줘.",
            "context": [],
            "candidate_tools": ["mcp__test__tool_b"],
            "decision": "unsupported",
            "acceptable_choices": [],
            "tool_catalog_version": "test-catalog-v1",
            "source": "synthetic",
            "review_status": "needs_human_review",
            "execution_verified": False,
        },
    ]

    raw_val = [
        {
            "id": "val_row_1",
            "scenario_id": "scenario_val_1",
            "split": "validation",
            "request": "그 문서를 다른 이름으로 변경해줘.",
            "context": [],
            "candidate_tools": ["mcp__test__tool_c"],
            "decision": "clarify",
            "acceptable_choices": [],
            "tool_catalog_version": "test-catalog-v1",
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

    # 3. Export rows matching raw dataset exactly
    export_train = [
        {
            "state": {
                "request": "도구 A를 실행해줘.",
                "context": [{"role": "user", "text": "선행 대화 맥락"}],
                "candidates": [
                    {
                        "alias": "T0",
                        "server": "test",
                        "name": "tool_a",
                        "summary": "Tool A description for unit testing with sufficient length.",
                    },
                    {
                        "alias": "T1",
                        "server": "test",
                        "name": "tool_b",
                        "summary": "Tool B description for unit testing with sufficient length.",
                    },
                ],
            },
            "questions": {
                "route_in_window": {
                    "type": "choice",
                    "instructions": "Route instructions.",
                    "criteria": {
                        "T0": "도구 T0",
                        "T1": "도구 T1",
                        "no_tool": "도구 없음",
                        "clarify": "확인 필요",
                        "window_none": "창 내 없음",
                    },
                }
            },
            "gold": {"route_in_window": {"label": "T0", "probabilities": {"T0": 1.0}}},
            "metadata": {
                "export_version": "decision-export-v1",
                "source_id": "train_row_1",
                "scenario_id": "scenario_train_1",
                "source_split": "train",
                "tool_catalog_version": "test-catalog-v1",
                "original_candidate_tools": ["mcp__test__tool_a", "mcp__test__tool_b"],
                "window_tools": ["mcp__test__tool_a", "mcp__test__tool_b"],
                "alias_to_tool": {"T0": "mcp__test__tool_a", "T1": "mcp__test__tool_b"},
                "original_decision": "select_tools",
                "acceptable_choices": [{"tool": "mcp__test__tool_a"}],
                "derived_negative_window": False,
                "source": "synthetic",
                "review_status": "needs_human_review",
                "execution_verified": False,
            },
        },
        {
            "state": {
                "request": "도메인 외 비행기 표 예약해줘.",
                "context": [],
                "candidates": [
                    {
                        "alias": "T0",
                        "server": "test",
                        "name": "tool_b",
                        "summary": "Tool B description for unit testing with sufficient length.",
                    }
                ],
            },
            "questions": {
                "route_in_window": {
                    "type": "choice",
                    "instructions": "Route instructions.",
                    "criteria": {
                        "T0": "도구 T0",
                        "no_tool": "도구 없음",
                        "clarify": "확인 필요",
                        "window_none": "창 내 없음",
                    },
                }
            },
            "gold": {"route_in_window": {"label": "window_none", "probabilities": {"window_none": 1.0}}},
            "metadata": {
                "export_version": "decision-export-v1",
                "source_id": "train_row_2",
                "scenario_id": "scenario_train_2",
                "source_split": "train",
                "tool_catalog_version": "test-catalog-v1",
                "original_candidate_tools": ["mcp__test__tool_b"],
                "window_tools": ["mcp__test__tool_b"],
                "alias_to_tool": {"T0": "mcp__test__tool_b"},
                "original_decision": "unsupported",
                "acceptable_choices": [],
                "derived_negative_window": False,
                "source": "synthetic",
                "review_status": "needs_human_review",
                "execution_verified": False,
            },
        },
    ]

    export_val = [
        {
            "state": {
                "request": "그 문서를 다른 이름으로 변경해줘.",
                "context": [],
                "candidates": [
                    {
                        "alias": "T0",
                        "server": "test",
                        "name": "tool_c",
                        "summary": "Tool C description for unit testing with sufficient length.",
                    }
                ],
            },
            "questions": {
                "route_in_window": {
                    "type": "choice",
                    "instructions": "Route instructions.",
                    "criteria": {
                        "T0": "도구 T0",
                        "no_tool": "도구 없음",
                        "clarify": "확인 필요",
                        "window_none": "창 내 없음",
                    },
                }
            },
            "gold": {"route_in_window": {"label": "clarify", "probabilities": {"clarify": 1.0}}},
            "metadata": {
                "export_version": "decision-export-v1",
                "source_id": "val_row_1",
                "scenario_id": "scenario_val_1",
                "source_split": "validation",
                "tool_catalog_version": "test-catalog-v1",
                "original_candidate_tools": ["mcp__test__tool_c"],
                "window_tools": ["mcp__test__tool_c"],
                "alias_to_tool": {"T0": "mcp__test__tool_c"},
                "original_decision": "clarify",
                "acceptable_choices": [],
                "derived_negative_window": False,
                "source": "synthetic",
                "review_status": "needs_human_review",
                "execution_verified": False,
            },
        }
    ]

    train_content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in export_train)
    val_content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in export_val)
    (export_dir / "train.jsonl").write_text(train_content, encoding="utf-8")
    (export_dir / "validation.jsonl").write_text(val_content, encoding="utf-8")

    manifest = {
        "export_version": "decision-export-v1",
        "purpose": "engineering_smoke_only",
        "seed": 3407,
        "catalog_version": "test-catalog-v1",
        "source_sha256": {
            "train.jsonl": compute_sha256(dataset_dir / "train.jsonl"),
            "validation.jsonl": compute_sha256(dataset_dir / "validation.jsonl"),
            "tools.json": compute_sha256(dataset_dir / "tools.json"),
        },
        "splits": {
            "train": {
                "rows": len(export_train),
                "scenario_groups": 2,
                "original_decisions": {"select_tools": 1, "unsupported": 1},
                "internal_labels": {"T0": 1, "window_none": 1},
                "negative_windows": 0,
                "sha256": compute_sha256(export_dir / "train.jsonl"),
            },
            "validation": {
                "rows": len(export_val),
                "scenario_groups": 1,
                "original_decisions": {"clarify": 1},
                "internal_labels": {"clarify": 1},
                "negative_windows": 0,
                "sha256": compute_sha256(export_dir / "validation.jsonl"),
            },
        },
        "state_card_summary_characters": 120,
        "max_window_tools": 8,
        "action_prediction": False,
        "limitations": ["Mock limitation"],
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return dataset_dir, export_dir


class TestPilotAudit(unittest.TestCase):
    def test_clean_temp_fixture_passes(self) -> None:
        """Verify that an independent temporary fixture passes audit completely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertTrue(res["audit_passed"], f"Audit failed: {res.get('all_errors')}")
            self.assertEqual(res["error_count"], 0)

    def test_missing_source_hash_key_detected(self) -> None:
        """Ensure missing tools.json or split key in source_sha256 is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
            del manifest["source_sha256"]["tools.json"]
            (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("Missing required source hash key" in e for e in res["all_errors"]))

    def test_source_id_tampering_detected(self) -> None:
        """Ensure tampered source_id not found in raw dataset is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["metadata"]["source_id"] = "nonexistent_source_id_999"
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("source_id 'nonexistent_source_id_999' not found" in e for e in res["all_errors"]))

    def test_context_tampering_detected(self) -> None:
        """Ensure modified state.context not matching raw dataset is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["state"]["context"] = [{"role": "assistant", "text": "Injected malicious context"}]
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("state.context differs from raw source" in e for e in res["all_errors"]))

    def test_request_tampering_detected(self) -> None:
        """Ensure modified state.request not matching raw dataset is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["state"]["request"] = "Completely altered request text."
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("state.request differs from raw source" in e for e in res["all_errors"]))

    def test_card_summary_length_limit_detected(self) -> None:
        """Ensure card summary exceeding 120 characters is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["state"]["candidates"][0]["summary"] = "A" * 121
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("exceeds 120 limit" in e for e in res["all_errors"]))

    def test_card_unwhitelisted_field_detected(self) -> None:
        """Ensure candidate cards with unwhitelisted extra fields are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["state"]["candidates"][0]["unapproved_field"] = "malicious_payload"
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("unwhitelisted fields" in e for e in res["all_errors"]))

    def test_duplicate_card_alias_detected(self) -> None:
        """Ensure duplicate card aliases in candidates are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["state"]["candidates"][1]["alias"] = "T0"  # Force duplicate T0
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("duplicate card alias" in e for e in res["all_errors"]))

    def test_unexpected_question_detected(self) -> None:
        """Ensure unexpected questions (e.g. action_choice) are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            rows = [json.loads(l) for l in (export_dir / "train.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["questions"]["action_choice"] = {"type": "choice", "criteria": {}}
            (export_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["train"]["sha256"] = compute_sha256(export_dir / "train.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("unexpected question keys" in e for e in res["all_errors"]))

    def test_scenario_leakage_detected(self) -> None:
        """Ensure scenario group overlap between train and validation is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            val_rows = [json.loads(l) for l in (export_dir / "validation.jsonl").read_text().splitlines() if l.strip()]
            # Force validation row to use a train scenario ID
            val_rows[0]["metadata"]["scenario_id"] = "scenario_train_1"
            (export_dir / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val_rows))
            manifest = json.loads((export_dir / "manifest.json").read_text())
            manifest["splits"]["validation"]["sha256"] = compute_sha256(export_dir / "validation.jsonl")
            (export_dir / "manifest.json").write_text(json.dumps(manifest))

            res = run_full_pilot_audit(export_dir, dataset_dir)
            self.assertFalse(res["audit_passed"])
            self.assertTrue(any("Scenario group leakage" in e for e in res["all_errors"]))

    def test_cli_invocation_on_temp_fixture(self) -> None:
        """Verify CLI execution via subprocess on temporary fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir, export_dir = create_mock_dataset_and_export(Path(tmpdir))
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "apps/review/pilot_audit.py"),
                "--export-dir",
                str(export_dir),
                "--dataset-dir",
                str(dataset_dir),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("Audit status: PASS", result.stdout)
            self.assertIn("Total errors: 0", result.stdout)

    def test_live_root_export_if_available(self) -> None:
        """Verify live root pilot export if present on environment, otherwise skip cleanly."""
        live_export = Path("/Volumes/Dove-Nest-SSD/projects/Jery/outputs/laya-pilot-20261017/export")
        live_dataset = Path("/Volumes/Dove-Nest-SSD/projects/Jery/datasets/mcp-routing-v2")
        if not (live_export.exists() and live_dataset.exists()):
            self.skipTest("Live root export or dataset not available in this checkout environment.")
        res = run_full_pilot_audit(live_export, live_dataset)
        self.assertTrue(res["audit_passed"], f"Live audit failed: {res.get('all_errors')}")
        self.assertEqual(res["error_count"], 0)



if __name__ == "__main__":
    unittest.main()
