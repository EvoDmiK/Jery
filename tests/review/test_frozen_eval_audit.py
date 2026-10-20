"""Unit tests for apps/review/frozen_eval_audit.py using explicit synthetic fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from apps.review.frozen_eval_audit import (
    DEFAULT_CATALOG,
    DEFAULT_HISTORICAL_PATHS,
    compute_sha256,
    load_catalog_spec,
    run_frozen_eval_audit,
    validate_safe_output_path,
)


class TestFrozenEvalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog_spec = load_catalog_spec(DEFAULT_CATALOG)
        cls.ten_tools = cls.catalog_spec["tool_ids"]
        cls.catalog_version = cls.catalog_spec["catalog_version"]
        cls.catalog_sha = cls.catalog_spec["sha256"]
        cls.raw_hashes_before = {
            str(p): compute_sha256(p) for p in [DEFAULT_CATALOG, *DEFAULT_HISTORICAL_PATHS]
        }

    def tearDown(self) -> None:
        # Ensure raw historical datasets and catalogs are never modified by any test
        for p_str, expected_sha in self.raw_hashes_before.items():
            self.assertEqual(compute_sha256(Path(p_str)), expected_sha, f"Raw file mutated: {p_str}")

    def _make_row(
        self,
        idx: int,
        tool: str,
        *,
        action: str | None = None,
        decision: str | None = "select_tools",
        review_status: str = "human_reviewed",
        source_kind: str = "real",
        actor: str = "human_user",
        request_text: str | None = None,
        context: list | None = None,
    ) -> dict:
        if action is None and self.catalog_spec["action_enums"].get(tool):
            action = sorted(self.catalog_spec["action_enums"][tool])[0]
        choices = []
        if decision == "select_tools":
            choice = {"tool": tool}
            if action is not None:
                choice["action"] = action
            choices = [choice]
        req = request_text or f"실제 사용자 고유 테스트 요청 번호 {idx} ({tool.split('__')[-1]} 관련 독립 질의 {idx * 97})"
        return {
            "id": f"real_eval_case_{idx:03d}",
            "scenario_id": f"real_session_group_{idx:03d}",
            "request": req,
            "context": context if context is not None else [],
            "candidate_tools": list(self.ten_tools),
            "tool_catalog_version": self.catalog_version,
            "decision": decision,
            "acceptable_choices": choices,
            "observed_choice": {"tool": tool, **({"action": action} if action else {})},
            "source": {
                "kind": source_kind,
                "actor": actor,
                "provenance": f"session_hmac_{idx:03d}",
            },
            "review_status": review_status,
            "execution_verified": False,
        }

    def _write_packet(
        self,
        tmpdir: Path,
        rows: list[dict],
        *,
        status: str = "candidate",
        manifest_overrides: dict | None = None,
        proposals: dict | None = None,
    ) -> Path:
        cases_path = tmpdir / "cases.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        cases_path.write_text(text, encoding="utf-8")
        cases_sha = compute_sha256(cases_path)

        by_dec: dict[str, int] = {}
        by_rev: dict[str, int] = {}
        for r in rows:
            d_key = "null" if r.get("decision") is None else str(r.get("decision"))
            by_dec[d_key] = by_dec.get(d_key, 0) + 1
            rv = str(r.get("review_status"))
            by_rev[rv] = by_rev.get(rv, 0) + 1

        manifest = {
            "schema_version": "frozen-eval-v1",
            "status": status,
            "catalog_version": self.catalog_version,
            "candidate_order_seed": 3407,
            "source_snapshot": "hermes-state-snapshot-20261020",
            "leak_audit_fingerprint": "synthetic-fixture-fp",
            "sha256": {
                "cases.jsonl": cases_sha,
                "tools.json": self.catalog_sha,
            },
            "counts": {
                "by_decision": by_dec,
                "by_review_status": by_rev,
            },
        }
        if manifest_overrides:
            manifest.update(manifest_overrides)
        (tmpdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if proposals is not None:
            (tmpdir / "proposals.json").write_text(
                json.dumps(proposals, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return tmpdir

    def test_missing_prerequisites_reported_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = run_frozen_eval_audit(packet_dir=Path(td) / "nonexistent")
            self.assertEqual(res["packet_status"], "missing_prerequisites")
            self.assertEqual(res["evaluation_gate"], "refused")
            self.assertTrue(len(res["missing_prerequisites"]) >= 2)
            self.assertEqual(res["blockers"][0]["code"], "MISSING_PREREQUISITES")

    def test_candidate_packet_with_null_labels_and_ai_proposals_stays_needs_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rows = [
                self._make_row(
                    i,
                    self.ten_tools[i % 10],
                    decision=None,
                    review_status="needs_human_review",
                )
                for i in range(10)
            ]
            proposals = {
                "review_status": "needs_human_review",
                "execution_verified": False,
                "proposals": {
                    r["id"]: [
                        {
                            "proposer": "claude",
                            "confidence": "high",
                            "disposition": "select_tools",
                            "rationale": "Matches catalog capability directly.",
                            "decision": "select_tools",
                            "acceptable_choices": [r["observed_choice"]],
                        },
                        {
                            "proposer": "antigravity",
                            "confidence": "high",
                            "disposition": "select_tools",
                            "rationale": "Independent catalog verification agrees.",
                            "decision": "select_tools",
                            "acceptable_choices": [r["observed_choice"]],
                        },
                    ]
                    for r in rows
                },
            }
            self._write_packet(tmpdir, rows, status="candidate", proposals=proposals)
            res = run_frozen_eval_audit(packet_dir=tmpdir)

            self.assertEqual(res["packet_status"], "candidate")
            self.assertFalse(res["packet_is_immutable_frozen"])
            self.assertFalse(res["freeze_implies_human_verified"])
            self.assertEqual(res["label_review_gate"], "unresolved_gold_with_ai_proposals_needs_human_review")
            # Evaluation must be refused when gold labels are unresolved (null) or status != frozen
            self.assertEqual(res["evaluation_gate"], "refused")
            self.assertTrue(res["provisional_ai_agreement_available"])
            self.assertEqual(res["blockers"], [])
            self.assertEqual(res["counts"]["unresolved_label_count"], 10)
            self.assertEqual(res["quota_and_shortage"]["real_human_reviewed_shortage"], 100)

    def test_frozen_status_distinguished_from_label_review_and_complete_100_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            # First test: frozen file set, but labels still needs_human_review -> provisional only, NOT human_reviewed_complete
            unreviewed_rows = [
                self._make_row(i, self.ten_tools[i // 10], review_status="needs_human_review")
                for i in range(100)
            ]
            self._write_packet(tmpdir, unreviewed_rows, status="frozen")
            res_unrev = run_frozen_eval_audit(packet_dir=tmpdir)
            self.assertEqual(res_unrev["packet_status"], "frozen")
            self.assertTrue(res_unrev["packet_is_immutable_frozen"])
            self.assertFalse(res_unrev["freeze_implies_human_verified"])
            self.assertEqual(res_unrev["label_review_gate"], "needs_human_review")
            self.assertEqual(res_unrev["evaluation_gate"], "provisional_exploratory_only_not_human_reviewed")
            self.assertEqual(res_unrev["quota_and_shortage"]["real_human_reviewed_shortage"], 100)
            self.assertEqual(res_unrev["quota_and_shortage"]["resolved_label_shortage"], 0)

        with tempfile.TemporaryDirectory() as td2:
            tmpdir2 = Path(td2)
            # Second test: frozen file set AND all 100 (10 per tool) human_reviewed real human_user requests
            reviewed_rows = [
                self._make_row(i, self.ten_tools[i // 10], review_status="human_reviewed")
                for i in range(100)
            ]
            self._write_packet(tmpdir2, reviewed_rows, status="frozen")
            res_rev = run_frozen_eval_audit(packet_dir=tmpdir2)
            self.assertEqual(res_rev["blockers"], [])
            self.assertEqual(res_rev["packet_status"], "frozen")
            self.assertEqual(res_rev["label_review_gate"], "human_reviewed_complete")
            self.assertEqual(res_rev["evaluation_gate"], "ready_for_reviewed_benchmark")
            self.assertEqual(res_rev["quota_and_shortage"]["real_human_reviewed_total"], 100)
            self.assertEqual(res_rev["quota_and_shortage"]["real_human_reviewed_shortage"], 0)

    def test_detects_historical_leakage_across_v1_v2_and_contrast_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rows = [
                # 1. Exact request from hermes-ten-tools-contrast-source-v1
                self._make_row(
                    0,
                    "mcp__obsidian_vault__vault",
                    action="rename",
                    request_text="Obsidian의 Comet.md 파일명을 Orbit.md로 바꿔줘.",
                ),
                # 2. Exact ID and scenario_id from mcp-routing-v2
                {
                    **self._make_row(1, "mcp__obsidian_vault__bases", action="list"),
                    "id": "obsidian_vault.bases.1.base.subset.1",
                    "scenario_id": "obsidian_vault.bases.1",
                },
                # 3. Near-duplicate of contrast-source request
                self._make_row(
                    2,
                    "mcp__obsidian_vault__vault",
                    action="move",
                    request_text="볼트의 Inbox/Tide.md 노트를 Archive/Tide.md로 지금 옮겨줘!",
                ),
            ]
            self._write_packet(tmpdir, rows, status="frozen")
            res = run_frozen_eval_audit(packet_dir=tmpdir)
            codes = {b["code"] for b in res["blockers"]}
            self.assertIn("HISTORICAL_ID_LEAKAGE", codes)
            self.assertIn("HISTORICAL_SCENARIO_LEAKAGE", codes)
            self.assertIn("HISTORICAL_EXACT_REQUEST_LEAKAGE", codes)
            self.assertFalse(res["leakage"]["zero_exact_leakage"])
            self.assertGreaterEqual(len(res["leakage"]["historical_near_duplicate_hits"]), 1)
            self.assertEqual(res["evaluation_gate"], "refused")

    def test_rejects_synthetic_and_non_human_cron_worker_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rows = [
                self._make_row(0, self.ten_tools[0], source_kind="synthetic"),
                self._make_row(1, self.ten_tools[1], actor="cron"),
                self._make_row(
                    2,
                    self.ten_tools[2],
                    actor="human_user",
                    request_text="You are working inside Orca, a multi-agent IDE. Send worker_done.",
                ),
            ]
            self._write_packet(tmpdir, rows, status="frozen")
            res = run_frozen_eval_audit(packet_dir=tmpdir)
            codes = {b["code"] for b in res["blockers"]}
            self.assertIn("SYNTHETIC_SOURCE_IN_REAL_BENCHMARK", codes)
            self.assertIn("NON_HUMAN_ACTOR_SOURCE", codes)
            self.assertEqual(len(res["source_authenticity"]["synthetic_rows"]), 1)
            self.assertEqual(len(res["source_authenticity"]["non_human_actor_rows"]), 2)
            self.assertEqual(res["evaluation_gate"], "refused")

    def test_rejects_catalog_capability_feature_boundary_and_ai_proposal_violations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            # Row 0: Swallow tool has no action in schema, but action="buy" supplied
            r0 = self._make_row(0, "mcp__swallow_trader__swallow_order")
            r0["acceptable_choices"] = [{"tool": "mcp__swallow_trader__swallow_order", "action": "buy"}]
            # Row 1: Post-call result field & tool ID leaked into context/request + unsanitized email + speaker nickname
            r1 = self._make_row(
                1,
                "mcp__obsidian_vault__vault",
                action="read",
                request_text="[SpeakerNick] mcp__obsidian_vault__vault 도구로 alice@example.com 노트를 읽어줘",
                context=[{"role": "tool", "content": "ok", "tool_call_id": "call_123"}],
            )
            proposals = {
                "proposals": {
                    r0["id"]: {
                        "proposer": "claude",
                        "confidence": "high",
                        "disposition": "select_tools",
                        "rationale": "Tested",
                        "human_reviewed": True,
                        "model_prediction": "T0",
                    }
                }
            }
            self._write_packet(tmpdir, [r0, r1], status="candidate", proposals=proposals)
            res = run_frozen_eval_audit(packet_dir=tmpdir)
            codes = {b["code"] for b in res["blockers"]}
            self.assertIn("CATALOG_CAPABILITY_VIOLATION", codes)
            self.assertIn("FEATURE_BOUNDARY_OR_POST_TOOL_LEAKAGE", codes)
            self.assertIn("UNSANITIZED_PII_OR_SECRET_DETECTED", codes)
            self.assertIn("AI_PROPOSALS_CONTRACT_VIOLATION", codes)
            patterns = {w["pattern"] for w in res["feature_and_privacy_hygiene"]["privacy_warnings"]}
            self.assertIn("bracket_speaker_nickname", patterns)
            self.assertIn("raw_email", patterns)

    def test_cli_deterministic_output_and_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rows = [self._make_row(i, self.ten_tools[i], decision=None, review_status="needs_human_review") for i in range(10)]
            self._write_packet(tmpdir, rows, status="candidate")

            # Refuse writing inside packet directory or datasets directory
            with self.assertRaises(ValueError):
                validate_safe_output_path(tmpdir / "out.json", [tmpdir])
            with self.assertRaises(ValueError):
                validate_safe_output_path(DEFAULT_CATALOG, [DEFAULT_CATALOG])

            out_json = tmpdir.parent / f"audit_out_{tmpdir.name}.json"
            try:
                cmd = [
                    sys.executable,
                    "apps/review/frozen_eval_audit.py",
                    "--packet-dir",
                    str(tmpdir),
                    "--output-json",
                    str(out_json),
                    "--json",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                stdout_payload = json.loads(proc.stdout)
                file_payload = json.loads(out_json.read_text(encoding="utf-8"))
                self.assertEqual(stdout_payload, file_payload)
                self.assertEqual(file_payload["packet_status"], "candidate")
            finally:
                if out_json.exists():
                    out_json.unlink()

    def test_candidate_manifest_v1_schema_and_real_alias_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            row = self._make_row(0, self.ten_tools[0], decision=None, review_status="needs_human_review")
            row["source"] = {
                "kind": "real",
                "record_type": "standalone_root_discord_user_message",
                "source_alias": "hermes-profile-1",
                "sanitized": True,
            }
            cases_path = tmpdir / "cases.jsonl"
            cases_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            c_manifest = {
                "schema_version": "frozen-eval-candidates-v1",
                "status": "candidate",
                "catalog_version": self.catalog_version,
                "catalog_sha256": self.catalog_sha,
                "cases_sha256": compute_sha256(cases_path),
                "counts": {"candidate_requests": 1},
            }
            (tmpdir / "candidate-manifest.json").write_text(json.dumps(c_manifest) + "\n", encoding="utf-8")
            res = run_frozen_eval_audit(packet_dir=tmpdir)
            self.assertEqual(res["blockers"], [])
            self.assertEqual(res["schema_version"], "frozen-eval-candidates-v1")
            self.assertEqual(res["counts"]["by_actor_kind"], {"human_user": 1})


if __name__ == "__main__":
    unittest.main()
