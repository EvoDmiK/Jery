"""Unit tests for independent dataset audit tool (dev-agy).

Tests audit execution, raw data immutability, roundtrip summary serialization,
verification flag separation, unseen class detection, and CLI invocations.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.review.audit import (
    DEFAULT_DATASET,
    compute_sha256,
    format_console_report,
    load_dataset,
    run_full_audit,
)


class TestReviewAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_dir = DEFAULT_DATASET
        self.manifest = json.loads((self.dataset_dir / 'manifest.json').read_text(encoding='utf-8'))

    def test_raw_data_unmodified_before_and_after_audit(self) -> None:
        """Ensure audit operations are strictly read-only and preserve byte-exact integrity."""
        hashes_before = {
            f: compute_sha256(self.dataset_dir / f) for f in self.manifest['sha256']
        }
        self.assertEqual(hashes_before, self.manifest['sha256'])

        # Run audit
        res = run_full_audit(self.dataset_dir)
        self.assertEqual(res['status'], 'AUDIT_COMPLETE')

        # Re-verify hashes after audit
        hashes_after = {
            f: compute_sha256(self.dataset_dir / f) for f in self.manifest['sha256']
        }
        self.assertEqual(hashes_before, hashes_after)

    def test_separate_verification_flags(self) -> None:
        """Verify synthetic review status and execution verification are tracked separately."""
        res = run_full_audit(self.dataset_dir)
        flags = res['verification_flags']

        self.assertTrue(flags['manifest_synthetic'])
        self.assertFalse(flags['manifest_human_reviewed'])
        self.assertTrue(flags['all_synthetic'])
        self.assertTrue(flags['all_needs_human_review'])
        self.assertTrue(flags['all_execution_unverified'])

        # Distribution checks
        self.assertEqual(flags['review_status_distribution'], {'needs_human_review': 1162})
        self.assertEqual(flags['execution_verified_distribution'], {False: 1162})
        self.assertEqual(flags['source_distribution'], {'synthetic': 1162})

    def test_decision_distribution_and_baseline(self) -> None:
        """Verify decision counts and naive majority accuracy baseline calculation."""
        res = run_full_audit(self.dataset_dir)
        dec = res['decisions_and_imbalance']

        self.assertEqual(dec['overall']['total'], 1162)
        self.assertEqual(
            dec['overall']['counts'],
            {'select_tools': 1062, 'unsupported': 60, 'no_tool': 20, 'clarify': 20},
        )
        self.assertEqual(dec['overall']['majority_class'], 'select_tools')
        self.assertAlmostEqual(dec['overall']['majority_baseline_accuracy'], 91.39, places=2)

    def test_unseen_classes_and_action_hazards(self) -> None:
        """Verify detection of missing evaluation tools and zero-shot action classes."""
        res = run_full_audit(self.dataset_dir)
        tc = res['tool_coverage']
        act = res['action_coverage']

        # Tool coverage
        self.assertEqual(tc['total_registered_tools'], 96)
        self.assertEqual(tc['train_covered'], 96)
        self.assertEqual(tc['missing_in_validation_count'], 11)
        self.assertEqual(tc['missing_in_test_count'], 12)
        self.assertEqual(tc['missing_in_both_eval_splits_count'], 10)

        expected_missing_both = [
            'mcp__grafana__get_alert_group',
            'mcp__grafana__get_current_oncall_users',
            'mcp__grafana__get_dashboard_summary',
            'mcp__grafana__get_oncall_shift',
            'mcp__grafana__get_snapshot',
            'mcp__grafana__list_provisioning_repositories',
            'mcp__obsidian_vault__list_resources',
            'mcp__obsidian_vault__workflow',
            'mcp__swallow_trader__swallow_reset_portfolio',
            'mcp__swallow_trader__swallow_trigger_pipeline',
        ]
        self.assertEqual(sorted(tc['missing_in_both_eval_splits']), sorted(expected_missing_both))

        # Zero-shot action classes
        val_unseen = act['validation_unseen_actions']
        test_unseen = act['test_unseen_actions']
        self.assertIn('patch', val_unseen.get('mcp__obsidian_vault__edit', []))
        self.assertIn('patch', test_unseen.get('mcp__obsidian_vault__edit', []))
        self.assertIn('view', val_unseen.get('mcp__obsidian_vault__bases', []))
        self.assertIn('export', test_unseen.get('mcp__obsidian_vault__bases', []))

    def test_candidate_shortcuts_and_server_shift(self) -> None:
        """Verify candidate length shortcut heuristic and server distribution shift."""
        res = run_full_audit(self.dataset_dir)
        ch = res['candidate_heuristics']
        srv = res['server_distribution']

        # Candidate length 4 shortcut
        self.assertTrue(ch['length_4_is_always_non_select'])
        self.assertTrue(ch['missing_target_only_in_swallow'])
        self.assertEqual(ch['missing_target_scenarios_count'], 20)

        # Server shift
        grafana_shift = srv['grafana_shift']
        self.assertGreaterEqual(grafana_shift['val_train_ratio'], 1.8)

    def test_audit_summary_roundtrip(self) -> None:
        """Verify summary export to JSON and re-reading integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / 'audit_summary.json'
            res = run_full_audit(self.dataset_dir)
            out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

            # Re-read
            loaded = json.loads(out_file.read_text(encoding='utf-8'))
            self.assertEqual(loaded['status'], 'AUDIT_COMPLETE')
            self.assertEqual(loaded['dataset_version'], 'mcp-routing-v2')
            self.assertEqual(len(loaded['blockers']), len(res['blockers']))
            self.assertEqual(len(loaded['warnings']), len(res['warnings']))

    def test_cli_execution(self) -> None:
        """Verify CLI execution via python subprocess."""
        cmd = [sys.executable, str(PROJECT_ROOT / 'apps/review/audit.py'), '--json']
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI failed: {proc.stderr}")
        data = json.loads(proc.stdout)
        self.assertEqual(data['status'], 'AUDIT_COMPLETE')

        # Test text report formatting
        text_report = format_console_report(data)
        self.assertIn('JERY MCP ROUTING DATASET AUDIT REPORT', text_report)
        self.assertIn('CRITICAL BLOCKERS', text_report)


if __name__ == '__main__':
    unittest.main()
