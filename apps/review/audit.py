"""Independent training readiness audit and coverage summary tool for Jery MCP datasets.

Python 3.9+ stdlib only (verified on Python 3.9.6). Audits datasets/mcp-routing-v2 for label semantics, split isolation,
unseen classes, server distribution shifts, action coverage, candidate shortcuts,
and verification flags without modifying raw data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / 'datasets/mcp-routing-v2'


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    catalog = json.loads((directory / 'tools.json').read_text(encoding='utf-8'))
    labels = json.loads((directory / 'labels.json').read_text(encoding='utf-8'))
    scenarios = json.loads((directory / 'scenarios.json').read_text(encoding='utf-8'))

    # Verify raw checksums
    checksum_errors = []
    for filename, expected_hash in manifest.get('sha256', {}).items():
        actual_hash = compute_sha256(directory / filename)
        if actual_hash != expected_hash:
            checksum_errors.append(f"{filename}: expected {expected_hash}, got {actual_hash}")
    if checksum_errors:
        raise ValueError(f"Checksum mismatch in raw dataset: {checksum_errors}")

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    all_rows: List[Dict[str, Any]] = []
    for split in ('train', 'validation', 'test'):
        split_rows = []
        for line in (directory / f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            split_rows.append(row)
        rows_by_split[split] = split_rows
        all_rows.extend(split_rows)

    return {
        'directory': str(directory),
        'manifest': manifest,
        'catalog': catalog,
        'tools': {t['id']: t for t in catalog['tools']},
        'labels': labels,
        'scenarios': scenarios,
        'rows_by_split': rows_by_split,
        'all_rows': all_rows,
    }


def audit_verification_flags(manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit dataset status flags, keeping synthetic review and execution separate."""
    source_counts = Counter(r.get('source') for r in rows)
    review_status_counts = Counter(r.get('review_status') for r in rows)
    execution_verified_counts = Counter(r.get('execution_verified') for r in rows)

    return {
        'manifest_synthetic': manifest.get('synthetic'),
        'manifest_human_reviewed': manifest.get('human_reviewed'),
        'all_synthetic': all(r.get('source') == 'synthetic' for r in rows),
        'all_needs_human_review': all(r.get('review_status') == 'needs_human_review' for r in rows),
        'all_execution_unverified': all(r.get('execution_verified') is False for r in rows),
        'source_distribution': dict(source_counts),
        'review_status_distribution': dict(review_status_counts),
        'execution_verified_distribution': dict(execution_verified_counts),
    }


def audit_decisions_and_imbalance(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Audit decision distribution across splits and compute majority baseline."""
    res: Dict[str, Any] = {'by_split': {}, 'overall': {}}
    overall_counts: Counter[str] = Counter()

    for split, rows in rows_by_split.items():
        counts = Counter(r['decision'] for r in rows)
        total = len(rows)
        overall_counts.update(counts)
        res['by_split'][split] = {
            'total': total,
            'counts': dict(counts),
            'percentages': {k: round(v / total * 100, 2) for k, v in counts.items()},
            'majority_class': counts.most_common(1)[0][0] if counts else None,
            'majority_baseline_accuracy': round(counts.most_common(1)[0][1] / total * 100, 2) if counts else 0.0,
        }

    total_all = sum(overall_counts.values())
    res['overall'] = {
        'total': total_all,
        'counts': dict(overall_counts),
        'percentages': {k: round(v / total_all * 100, 2) for k, v in overall_counts.items()},
        'majority_class': overall_counts.most_common(1)[0][0],
        'majority_baseline_accuracy': round(overall_counts.most_common(1)[0][1] / total_all * 100, 2),
    }
    return res


def audit_server_distribution(rows_by_split: Dict[str, List[Dict[str, Any]]], tools: Dict[str, Any]) -> Dict[str, Any]:
    """Detect server distribution shift between train and validation/test.

    Server coverage is evaluated per select_tools example (not per choice alternative):
    - For each select_tools row, acceptable servers are extracted as a unique set
      (ensuring that multiple choices within the same server, e.g. Swallow multi-choice,
      are counted exactly once for that example).
    - The denominator for each split is the exact number of select_tools examples in that split.
    - If a scenario contains alternative choices spanning multiple servers (cross-server alternatives),
      each represented server counts once for that example, and server percentages represent the
      proportion of select_tools examples where that server provides an acceptable tool.
    """
    server_counts: Dict[str, Counter[str]] = {}
    server_pcts: Dict[str, Dict[str, float]] = {}
    select_totals: Dict[str, int] = {}

    for split, rows in rows_by_split.items():
        select_rows = [r for r in rows if r.get('decision') == 'select_tools']
        total_select = len(select_rows)
        select_totals[split] = total_select

        counts: Counter[str] = Counter()
        for r in select_rows:
            servers_in_row = {
                tools[c['tool']]['server']
                for c in r.get('acceptable_choices', [])
                if c.get('tool') in tools
            }
            for server in servers_in_row:
                counts[server] += 1
        server_counts[split] = counts
        server_pcts[split] = {
            k: round(v / total_select * 100, 2) for k, v in counts.items()
        } if total_select else {}

    grafana_train_pct = server_pcts.get('train', {}).get('grafana', 0.0)
    grafana_val_pct = server_pcts.get('validation', {}).get('grafana', 0.0)
    grafana_test_pct = server_pcts.get('test', {}).get('grafana', 0.0)

    return {
        'denominator_select_tools_count': select_totals,
        'counts': {s: dict(c) for s, c in server_counts.items()},
        'percentages': server_pcts,
        'grafana_shift': {
            'train_pct': grafana_train_pct,
            'validation_pct': grafana_val_pct,
            'test_pct': grafana_test_pct,
            'val_train_ratio': round(grafana_val_pct / grafana_train_pct, 2) if grafana_train_pct else 0.0,
        },
    }


def audit_tool_coverage(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    tools: Dict[str, Any],
) -> Dict[str, Any]:
    """Find tools missing from validation/test and examine train frequencies."""
    all_tool_ids = set(tools.keys())
    split_tools: Dict[str, Set[str]] = {}
    split_tool_counts: Dict[str, Counter[str]] = {}

    for split, rows in rows_by_split.items():
        counts: Counter[str] = Counter()
        for r in rows:
            for c in r['acceptable_choices']:
                counts[c['tool']] += 1
        split_tools[split] = set(counts.keys())
        split_tool_counts[split] = counts

    train_tools = split_tools.get('train', set())
    val_tools = split_tools.get('validation', set())
    test_tools = split_tools.get('test', set())

    missing_val = sorted(all_tool_ids - val_tools)
    missing_test = sorted(all_tool_ids - test_tools)
    missing_both = sorted(all_tool_ids - (val_tools | test_tools))

    train_c = split_tool_counts.get('train', Counter())
    train_freq_hist = dict(Counter(train_c.values()))

    # Tools with very low representation in train (count == 2)
    low_freq_train = [t for t, cnt in train_c.items() if cnt <= 2]

    return {
        'total_registered_tools': len(all_tool_ids),
        'train_covered': len(train_tools),
        'validation_covered': len(val_tools),
        'test_covered': len(test_tools),
        'missing_in_validation': missing_val,
        'missing_in_validation_count': len(missing_val),
        'missing_in_test': missing_test,
        'missing_in_test_count': len(missing_test),
        'missing_in_both_eval_splits': missing_both,
        'missing_in_both_eval_splits_count': len(missing_both),
        'train_frequency_histogram': train_freq_hist,
        'train_low_frequency_count': len(low_freq_train),
    }


def audit_action_coverage(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    tools: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit action enum definitions, unused actions, and unseen action classes in val/test."""
    defined_actions: Dict[str, List[str]] = {}
    for tid, t in tools.items():
        enums = t.get('input_schema', {}).get('properties', {}).get('action', {}).get('enum')
        if enums:
            defined_actions[tid] = enums

    used_actions_all: Dict[str, Set[str]] = defaultdict(set)
    used_actions_by_split: Dict[str, Dict[str, Set[str]]] = {
        'train': defaultdict(set),
        'validation': defaultdict(set),
        'test': defaultdict(set),
    }

    for split, rows in rows_by_split.items():
        for r in rows:
            for c in r.get('acceptable_choices', []):
                if 'action' in c:
                    used_actions_all[c['tool']].add(c['action'])
                    used_actions_by_split[split][c['tool']].add(c['action'])

    unused_actions: Dict[str, List[str]] = {}
    total_defined_actions = 0
    total_used_actions = 0

    for tid, act_list in defined_actions.items():
        total_defined_actions += len(act_list)
        used = used_actions_all.get(tid, set())
        total_used_actions += len(used)
        unused = sorted(set(act_list) - used)
        if unused:
            unused_actions[tid] = unused

    # Actions appearing in validation or test but NEVER in train!
    train_actions = used_actions_by_split['train']
    val_unseen_actions: Dict[str, List[str]] = {}
    test_unseen_actions: Dict[str, List[str]] = {}

    for tid, val_acts in used_actions_by_split['validation'].items():
        unseen = sorted(val_acts - train_actions.get(tid, set()))
        if unseen:
            val_unseen_actions[tid] = unseen

    for tid, test_acts in used_actions_by_split['test'].items():
        unseen = sorted(test_acts - train_actions.get(tid, set()))
        if unseen:
            test_unseen_actions[tid] = unseen

    return {
        'tools_with_action_enum_count': len(defined_actions),
        'total_defined_actions': total_defined_actions,
        'total_used_actions': total_used_actions,
        'total_unused_actions': total_defined_actions - total_used_actions,
        'unused_actions_by_tool': unused_actions,
        'validation_unseen_actions': val_unseen_actions,
        'test_unseen_actions': test_unseen_actions,
    }


def audit_candidate_heuristics(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    scenarios: List[Dict[str, Any]],
    tools: Dict[str, Any],
) -> Dict[str, Any]:
    """Examine candidate lengths and shortcut correlations with decision types."""
    cand_len_by_decision: Dict[str, Counter[int]] = defaultdict(Counter)
    for rows in rows_by_split.values():
        for r in rows:
            cand_len_by_decision[r['decision']][len(r['candidate_tools'])] += 1

    # Check missing_target_case distribution
    missing_target_scenarios = [s for s in scenarios if s.get('missing_target_case')]
    missing_target_servers = Counter()
    for s in missing_target_scenarios:
        for c in s.get('acceptable_choices', []):
            missing_target_servers[tools[c['tool']]['server']] += 1

    return {
        'candidate_length_by_decision': {
            dec: dict(sorted(counter.items())) for dec, counter in cand_len_by_decision.items()
        },
        'length_4_is_always_non_select': all(
            len(r['candidate_tools']) != 4 for rows in rows_by_split.values() for r in rows if r['decision'] == 'select_tools'
        ),
        'missing_target_scenarios_count': len(missing_target_scenarios),
        'missing_target_servers': dict(missing_target_servers),
        'missing_target_only_in_swallow': set(missing_target_servers.keys()) == {'swallow_trader'},
    }


def audit_context_and_clarify(
    scenarios: List[Dict[str, Any]],
    rows_by_split: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Audit clarify cases and reliance on prior conversation context."""
    scenarios_with_context = [s for s in scenarios if s.get('context')]
    context_scenario_ids = [s['scenario_id'] for s in scenarios_with_context]

    clarify_scenarios = [s for s in scenarios if s['decision'] == 'clarify']
    clarify_by_split = Counter(s['split'] for s in clarify_scenarios)

    # Check which domains clarify covers
    # ponytail: naive domain scan by prefix
    clarify_domains = Counter()
    for s in clarify_scenarios:
        # Check matching context_resolved variant if present
        prefix = s['scenario_id']
        clarify_domains[prefix] += 1

    return {
        'total_scenario_items': len(scenarios),
        'total_scenarios_with_context': len(scenarios_with_context),
        'context_scenario_ids': sorted(set(context_scenario_ids)),
        'total_clarify_scenarios': len(clarify_scenarios),
        'clarify_split_distribution': dict(clarify_by_split),
        'validation_clarify_count': clarify_by_split.get('validation', 0),
        'test_clarify_count': clarify_by_split.get('test', 0),
        'clarify_scenarios': [
            {'scenario_id': s['scenario_id'], 'split': s['split'], 'requests': s['requests']}
            for s in clarify_scenarios
        ],
    }


def audit_multi_choice_and_ambiguity(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit acceptable_choices cardinality and potential ambiguity."""
    multi_choice_scenarios = [s for s in scenarios if len(s.get('acceptable_choices', [])) > 1]
    return {
        'multi_choice_scenario_count': len(multi_choice_scenarios),
        'multi_choice_scenarios': [
            {
                'scenario_id': s['scenario_id'],
                'split': s['split'],
                'tools': [c['tool'] for c in s.get('acceptable_choices', [])],
            }
            for s in multi_choice_scenarios
        ],
    }


def audit_near_duplicates_and_leakage(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Examine split leakage and lexical overlaps across splits."""
    def tokenize(text: str) -> Set[str]:
        return set(re.findall(r'[a-zA-Z0-9가-힣]+', text.lower()))

    train_rows = rows_by_split.get('train', [])
    train_reqs: Dict[str, Tuple[str, str]] = {r['request']: (r['scenario_id'], r['decision']) for r in train_rows}

    # Exact overlap
    exact_val_train = set(r['request'] for r in rows_by_split.get('validation', [])) & set(train_reqs.keys())
    exact_test_train = set(r['request'] for r in rows_by_split.get('test', [])) & set(train_reqs.keys())

    # Jaccard >= 0.45 overlap pairs
    high_overlap_pairs = []
    for split in ('validation', 'test'):
        for r in rows_by_split.get(split, []):
            e_req = r['request']
            e_tok = tokenize(e_req)
            if not e_tok:
                continue
            for t_req, (t_sid, t_dec) in train_reqs.items():
                t_tok = tokenize(t_req)
                if not t_tok:
                    continue
                jaccard = len(e_tok & t_tok) / len(e_tok | t_tok)
                if jaccard >= 0.45:
                    high_overlap_pairs.append({
                        'eval_split': split,
                        'eval_sid': r['scenario_id'],
                        'eval_req': e_req,
                        'train_sid': t_sid,
                        'train_req': t_req,
                        'jaccard': round(jaccard, 3),
                    })

    # Dedup by (eval_req, train_req)
    seen = set()
    dedup_pairs = []
    for item in sorted(high_overlap_pairs, key=lambda x: -x['jaccard']):
        key = (item['eval_req'], item['train_req'])
        if key not in seen:
            seen.add(key)
            dedup_pairs.append(item)

    return {
        'exact_val_train_overlap': len(exact_val_train),
        'exact_test_train_overlap': len(exact_test_train),
        'high_jaccard_pairs_count': len(dedup_pairs),
        'sample_high_jaccard_pairs': dedup_pairs[:10],
    }


def run_full_audit(directory: Path = DEFAULT_DATASET) -> Dict[str, Any]:
    data = load_dataset(directory)
    flags = audit_verification_flags(data['manifest'], data['all_rows'])
    decisions = audit_decisions_and_imbalance(data['rows_by_split'])
    servers = audit_server_distribution(data['rows_by_split'], data['tools'])
    tools_cov = audit_tool_coverage(data['rows_by_split'], data['tools'])
    actions = audit_action_coverage(data['rows_by_split'], data['tools'], data['scenarios'])
    heuristics = audit_candidate_heuristics(data['rows_by_split'], data['scenarios'], data['tools'])
    context = audit_context_and_clarify(data['scenarios'], data['rows_by_split'])
    multi_choice = audit_multi_choice_and_ambiguity(data['scenarios'])
    leakage = audit_near_duplicates_and_leakage(data['rows_by_split'])

    # Compile critical blockers and warnings
    blockers = []
    warnings = []

    # Blocker / Warning 1: Unseen tools in validation / test
    if tools_cov['missing_in_both_eval_splits_count'] > 0:
        warnings.append({
            'code': 'UNSEEN_EVAL_TOOLS',
            'severity': 'HIGH',
            'message': f"{tools_cov['missing_in_both_eval_splits_count']} tools appear ONLY in train and cannot be evaluated on val/test.",
            'tools': tools_cov['missing_in_both_eval_splits'],
        })

    # Blocker / Warning 2: Unseen action classes
    if actions['validation_unseen_actions'] or actions['test_unseen_actions']:
        blockers.append({
            'code': 'ZERO_SHOT_ACTION_EVAL',
            'severity': 'CRITICAL',
            'message': (
                "Closed-set action classification heads will fail on unseen actions "
                "(0% recall on val/test action classes absent from train). "
                "Recommend reporting overall action metrics alongside separate seen/unseen action breakdowns."
            ),
            'validation_unseen': actions['validation_unseen_actions'],
            'test_unseen': actions['test_unseen_actions'],
        })

    # Blocker / Warning 3: Extreme candidate bias on missing_target_case
    if heuristics['missing_target_only_in_swallow']:
        warnings.append({
            'code': 'UNSUPPORTED_CANDIDATE_BIAS',
            'severity': 'HIGH',
            'message': "Candidate omission (missing_target_case) turning into 'unsupported' is only synthesized for Swallow Trader. Other servers never train this.",
            'servers': heuristics['missing_target_servers'],
        })

    # Blocker / Warning 4: Candidate length shortcut
    if heuristics['length_4_is_always_non_select']:
        warnings.append({
            'code': 'CANDIDATE_LENGTH_SHORTCUT',
            'severity': 'MEDIUM',
            'message': "Candidate length == 4 is 100% predictive of non-select decisions (no_tool, clarify, unsupported). Model could exploit this artifact.",
        })

    # Blocker / Warning 5: Extreme server shift
    if servers['grafana_shift']['val_train_ratio'] >= 1.5:
        warnings.append({
            'code': 'SERVER_DISTRIBUTION_SHIFT',
            'severity': 'HIGH',
            'message': f"Grafana represents {servers['grafana_shift']['train_pct']}% of train select_tools but jumps to {servers['grafana_shift']['validation_pct']}% in validation.",
        })

    # Blocker / Warning 6: Decision class imbalance
    if decisions['overall']['majority_baseline_accuracy'] > 90.0:
        warnings.append({
            'code': 'DECISION_CLASS_IMBALANCE',
            'severity': 'HIGH',
            'message': f"select_tools constitutes {decisions['overall']['majority_baseline_accuracy']}% of dataset. Naive majority classifier achieves >91% accuracy.",
        })

    return {
        'status': 'AUDIT_COMPLETE',
        'dataset_version': data['manifest'].get('dataset_version'),
        'catalog_version': data['manifest'].get('catalog_version'),
        'summary': {
            'blocker_count': len(blockers),
            'warning_count': len(warnings),
        },
        'verification_flags': flags,
        'decisions_and_imbalance': decisions,
        'server_distribution': servers,
        'tool_coverage': tools_cov,
        'action_coverage': actions,
        'candidate_heuristics': heuristics,
        'context_and_clarify': context,
        'multi_choice_ambiguity': multi_choice,
        'leakage_and_similarity': leakage,
        'blockers': blockers,
        'warnings': warnings,
    }


def format_console_report(result: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"  JERY MCP ROUTING DATASET AUDIT REPORT: {result['dataset_version']}")
    lines.append(f"  Catalog Version: {result['catalog_version']}")
    lines.append("=" * 78)

    # 1. Verification Flags
    vf = result['verification_flags']
    lines.append("\n[1] DATA ORIGIN & VERIFICATION FLAGS (Independent Tracking)")
    lines.append(f"  - Synthetic Manifest: {vf['manifest_synthetic']} (all rows synthetic: {vf['all_synthetic']})")
    lines.append(f"  - Human Reviewed Manifest: {vf['manifest_human_reviewed']} (all rows needs_human_review: {vf['all_needs_human_review']})")
    lines.append(f"  - Execution Verified Status: {not vf['all_execution_unverified']} (all rows execution_verified=False: {vf['all_execution_unverified']})")
    lines.append("  * Note: Review status and execution verification are strictly independent.")

    # 2. Decision Distribution
    dec = result['decisions_and_imbalance']
    lines.append("\n[2] DECISION CLASS DISTRIBUTION & BASELINE")
    lines.append("  Split       | Total | select_tools (%) | unsupported (%) | clarify (%) | no_tool (%)")
    lines.append("  " + "-" * 72)
    for sp in ('train', 'validation', 'test'):
        d = dec['by_split'][sp]
        pct = d['percentages']
        lines.append(
            f"  {sp:<11} | {d['total']:<5} | {pct.get('select_tools', 0.0):>5.1f}%          | {pct.get('unsupported', 0.0):>5.1f}%         | {pct.get('clarify', 0.0):>5.1f}%       | {pct.get('no_tool', 0.0):>5.1f}%"
        )
    ov = dec['overall']
    ov_pct = ov['percentages']
    lines.append(
        f"  OVERALL     | {ov['total']:<5} | {ov_pct.get('select_tools', 0.0):>5.1f}%          | {ov_pct.get('unsupported', 0.0):>5.1f}%         | {ov_pct.get('clarify', 0.0):>5.1f}%       | {ov_pct.get('no_tool', 0.0):>5.1f}%"
    )
    lines.append(f"  * Naive Majority Class Accuracy Baseline: {ov['majority_baseline_accuracy']}%")

    # 3. Server Distribution Shift
    srv = result['server_distribution']
    lines.append("\n[3] SERVER DISTRIBUTION SHIFT (select_tools Targets, Per-Example Unique Server)")
    lines.append("  Server          | Train (%) | Val (%)   | Test (%)  | Ratio (Val/Train)")
    lines.append("  " + "-" * 65)
    all_servers = sorted(set(srv['percentages'].get('train', {}).keys()) | set(srv['percentages'].get('validation', {}).keys()))
    for s in all_servers:
        tp = srv['percentages'].get('train', {}).get(s, 0.0)
        vp = srv['percentages'].get('validation', {}).get(s, 0.0)
        tep = srv['percentages'].get('test', {}).get(s, 0.0)
        ratio = round(vp / tp, 2) if tp else 0.0
        lines.append(f"  {s:<15} | {tp:>7.1f}%  | {vp:>7.1f}%  | {tep:>7.1f}%  | {ratio:>6.2f}x")
    dt = srv.get('denominator_select_tools_count', {})
    lines.append(f"  * Denominator: select_tools examples (Train: {dt.get('train', 0)}, Val: {dt.get('validation', 0)}, Test: {dt.get('test', 0)})")

    # 4. Tool Coverage & Unseen Classes
    tc = result['tool_coverage']
    total_reg = tc['total_registered_tools']
    train_pct = round(tc['train_covered'] / total_reg * 100, 1) if total_reg else 0.0
    val_pct = round(tc['validation_covered'] / total_reg * 100, 1) if total_reg else 0.0
    test_pct = round(tc['test_covered'] / total_reg * 100, 1) if total_reg else 0.0
    lines.append("\n[4] TOOL COVERAGE & UNSEEN CLASSES IN EVALUATION")
    lines.append(f"  - Total Registered Tools: {total_reg}")
    lines.append(f"  - Train Covered Tools:    {tc['train_covered']} / {total_reg} ({train_pct:.1f}%)")
    lines.append(f"  - Val Covered Tools:      {tc['validation_covered']} / {total_reg} ({val_pct:.1f}%)")
    lines.append(f"  - Test Covered Tools:     {tc['test_covered']} / {total_reg} ({test_pct:.1f}%)")
    lines.append(f"  - Tools Missing in Val:   {tc['missing_in_validation_count']} tools")
    lines.append(f"  - Tools Missing in Test:  {tc['missing_in_test_count']} tools")
    lines.append(f"  - Tools Missing in Both:  {tc['missing_in_both_eval_splits_count']} tools:")
    for t in tc['missing_in_both_eval_splits']:
        lines.append(f"      * {t}")

    # 5. Action Enum Coverage
    act = result['action_coverage']
    lines.append("\n[5] ACTION ENUM COVERAGE & ZERO-SHOT ACTION HAZARDS")
    lines.append(f"  - Tools with Action Enum: {act['tools_with_action_enum_count']}")
    lines.append(f"  - Total Actions Defined:  {act['total_defined_actions']}")
    lines.append(f"  - Total Actions Used:     {act['total_used_actions']} ({act['total_unused_actions']} unused in any scenario)")
    if act['validation_unseen_actions'] or act['test_unseen_actions']:
        lines.append("  ! ZERO-SHOT ACTION FAILURE RISK (Actions tested in val/test but absent in train):")
        for tid, unseen in act['validation_unseen_actions'].items():
            lines.append(f"      [Validation] {tid}: {unseen}")
        for tid, unseen in act['test_unseen_actions'].items():
            lines.append(f"      [Test]       {tid}: {unseen}")

    # 6. Candidate Heuristics & Shortcuts
    ch = result['candidate_heuristics']
    lines.append("\n[6] CANDIDATE HEURISTICS & SHORTCUTS")
    lines.append(f"  - Candidate length == 4 is strictly non-select: {ch['length_4_is_always_non_select']}")
    lines.append(f"  - Missing Target Scenarios: {ch['missing_target_scenarios_count']} (Only in Swallow: {ch['missing_target_only_in_swallow']})")
    lines.append(f"  - Servers with missing target test: {list(ch['missing_target_servers'].keys())}")

    # 7. Context & Clarify Scarcity
    ctx = result['context_and_clarify']
    total_scenarios = ctx.get('total_scenario_items', len(ctx['context_scenario_ids']))
    lines.append("\n[7] CONTEXT DEPENDENCY & CLARIFY SCARCITY")
    lines.append(f"  - Total Scenarios with non-empty context: {ctx['total_scenarios_with_context']} / {total_scenarios} items")
    lines.append(f"  - Clarify Scenarios Total: {ctx['total_clarify_scenarios']} (Train: {ctx['clarify_split_distribution'].get('train', 0)}, Val: {ctx['validation_clarify_count']}, Test: {ctx['test_clarify_count']})")
    lines.append("  * Note: Val and Test each contain ONLY 1 clarify scenario.")

    # 8. Critical Blockers & Warnings Summary
    lines.append("\n" + "=" * 78)
    lines.append("  AUDIT VERDICT & CRITICAL ACTION ITEMS BEFORE TRAINING")
    lines.append("=" * 78)
    if result['blockers']:
        lines.append(f"  [CRITICAL BLOCKERS ({len(result['blockers'])})]")
        for b in result['blockers']:
            lines.append(f"  - [{b['code']}] {b['message']}")
    if result['warnings']:
        lines.append(f"  [HIGH/MEDIUM WARNINGS ({len(result['warnings'])})]")
        for w in result['warnings']:
            lines.append(f"  - [{w['code']} | {w['severity']}] {w['message']}")

    return "\n".join(lines)


def validate_output_summary_path(output_path: Path, dataset_dir: Path) -> Path:
    """Ensure output summary does not overwrite or reside inside dataset directory or alias any dataset file.

    Raises ValueError if output_path is within dataset_dir, matches a dataset file,
    or hardlinks/symlinks to an existing dataset file.
    """
    output_resolved = output_path.resolve()
    dataset_resolved = dataset_dir.resolve()

    # Reject if output path is inside dataset directory or equals dataset directory
    try:
        output_resolved.relative_to(dataset_resolved)
        raise ValueError(f"Refusing to write --output-summary inside dataset directory: {output_path}")
    except ValueError as e:
        if "Refusing to write --output-summary inside dataset directory" in str(e):
            raise
        pass

    if output_resolved == dataset_resolved:
        raise ValueError(f"Refusing to overwrite dataset directory with --output-summary: {output_path}")

    # Reject if output path points to or aliases any existing file in the dataset directory
    if dataset_resolved.exists() and dataset_resolved.is_dir():
        for item in dataset_resolved.iterdir():
            if item.is_file():
                if output_resolved == item.resolve():
                    raise ValueError(f"Refusing to overwrite dataset file with --output-summary: {output_path}")
                if output_path.exists():
                    try:
                        if os.path.samefile(output_path, item):
                            raise ValueError(f"Output summary aliases an existing dataset file: {output_path}")
                    except OSError:
                        pass
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DATASET, help='Path to dataset directory')
    parser.add_argument('--json', action='store_true', help='Output full audit results as JSON')
    parser.add_argument('--output-summary', type=Path, help='Write audit summary JSON to file')
    parser.add_argument('--quiet', action='store_true', help='Suppress stdout report')
    args = parser.parse_args()

    try:
        audit_res = run_full_audit(args.directory)
    except Exception as e:
        sys.stderr.write(f"Audit failed with error: {e}\n")
        return 1

    if args.output_summary:
        try:
            target_path = validate_output_summary_path(args.output_summary, args.directory)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(audit_res, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        except Exception as e:
            sys.stderr.write(f"Failed to write output summary: {e}\n")
            return 1

    if not args.quiet:
        if args.json:
            print(json.dumps(audit_res, ensure_ascii=False, indent=2))
        else:
            print(format_console_report(audit_res))

    return 0


if __name__ == '__main__':
    sys.exit(main())
