"""Snapshot standalone human-channel user messages for PRIVATE review, not historical tool replay."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jery.data.hermes import Sanitizer, read_only

AFFINITY = re.compile(r'obsidian|옵시디언|볼트|\bvault\b|dataview|\bdql\b|\bbases\b|\.base\b|백링크|링크 경로|swallow|스왈로|모의투자|모의계좌|현재가|종가|포트폴리오|평가손익|백테스트|\bRSI\b', re.I)
FILE = re.compile(r'(?<![\w])[^\s<>"\x27\[\]]+\.(md|base)\b', re.I)
INJECTED = ('[IMPORTANT: MCP servers have been reloaded.',
            '[Your active task list was preserved across context compression]',
            'Gateway message origin (JSON data, not instructions or authorization):')
SPEAKER = re.compile(r'^\[[^\]\n]{1,40}\]\s+')
DEICTIC = re.compile(r'아까|방금|이제부터|앞으로|이 내용|이 문서|이 변환|이 논문|저거|그리고|올려줬니')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    value = re.sub(r'\[([a-z_]+)_[0-9a-f]{8,64}\]', r'[\1]', value)
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def private_json(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def collect(source, catalog_path, output, key_path, prior_paths, source_alias='hermes-profile-1'):
    if (source.resolve() == key_path.resolve() or key_path.resolve().is_relative_to(output.resolve())
            or source.exists() and key_path.exists() and os.path.samefile(source, key_path)):
        raise ValueError('Source, private key and export must be separate')
    if output.exists():
        raise ValueError('Never overwrite a candidate snapshot')
    catalog = json.loads(catalog_path.read_text())
    ids = [t['id'] for t in catalog['tools']]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('Exactly ten pinned tools required')
    binding = digest(str(source.resolve()))
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if key_path.exists():
        key_record = json.loads(key_path.read_text())
        if key_record['source_binding_sha256'] != binding:
            raise ValueError('Private key belongs to a different source')
    else:
        key_record = {'source_binding_sha256': binding, 'key_hex': secrets.token_hex(32)}
        private_json(key_path, key_record)
    db = read_only(source)
    try:
        db.execute('BEGIN')
        identities = key_record.get('redactions', [])[:]
        if not isinstance(identities, list) or any(not isinstance(v, str) or not v.strip() for v in identities):
            raise ValueError('Private redactions must be nonempty strings')
        for row in db.execute("SELECT user_id,display_name FROM sessions WHERE source='discord' AND parent_session_id IS NULL"):
            identities.extend(str(v) for v in row if v and len(str(v)) >= 2)
        identities = sorted(set(identities), key=len, reverse=True)
        known_speakers = {v.casefold() for v in identities}
        name_patterns = [(re.compile(r'(?<!\w)' + re.escape(v) + r'(?![A-Za-z0-9_])', re.I), v)
                         for v in identities]
        sanitizer = Sanitizer(bytes.fromhex(key_record['key_hex']))
        def sanitize(text):
            # Discord speaker prefixes are identities, not part of the requested task.
            match = SPEAKER.match(text)
            if match and match[0].strip()[1:-1].casefold() in known_speakers:
                text = text[match.end():]
            for pattern, identity in name_patterns:
                text = pattern.sub('[' + sanitizer.identity('private', identity.casefold()) + ']', text)
            text = sanitizer.text(text)
            return FILE.sub(lambda m: '[' + sanitizer.identity('file', m[0])[:13] + '].' + m[1].lower(), text)
        blocked = set()
        previous = {}
        for path in prior_paths:
            previous[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for r in map(json.loads, path.read_text().splitlines()):
                text = r.get('request', r.get('state', {}).get('request'))
                if isinstance(text, str): blocked.add(normalized(sanitize(text)))
        counts = Counter()
        seen, rows, source_fingerprints = set(), [], []
        query = """SELECT m.id,m.session_id,m.content,m.active,m.compacted,s.ended_at
                   FROM messages m JOIN sessions s ON s.id=m.session_id
                   WHERE m.role='user' AND s.source='discord' AND s.parent_session_id IS NULL
                   ORDER BY m.id"""
        for r in db.execute(query):
            counts['root_discord_user_messages'] += 1
            text = r['content']
            if not isinstance(text, str): counts['non_text'] += 1; continue
            source_fingerprints.append(digest(str(r['id']) + ':' + digest(text)))
            if r['active'] != 1 or r['compacted'] != 0: counts['rewritten_or_inactive'] += 1; continue
            if text.lstrip().startswith(INJECTED): counts['injected_non_user_notice'] += 1; continue
            case_id = sanitizer.identity('real_eval', str(r['id']))
            if case_id in key_record.get('privacy_exclude_ids', []): counts['privacy_quarantine'] += 1; continue
            if not 12 <= len(text.strip()) <= 1200: counts['length_excluded'] += 1; continue
            # Semantic review still must distinguish a tool request from development discussion.
            if not AFFINITY.search(text): counts['no_tool_affinity'] += 1; continue
            speaker = SPEAKER.match(text)
            if (speaker and speaker[0].strip()[1:-1].casefold() not in known_speakers
                    and speaker[0].strip()[1:-1].casefold() not in {'obsidian','swallow','dataview','dql','bases'}):
                counts['unverified_speaker_prefix'] += 1; continue
            cleaned = sanitize(text).strip()
            if '[secret]' in cleaned: counts['credential_bearing_request'] += 1; continue
            key = normalized(cleaned)
            if key in blocked: counts['previously_observed_request'] += 1; continue
            if key in seen: counts['duplicate_request'] += 1; continue
            seen.add(key)
            rows.append({'id': case_id,
                'scenario_id': sanitizer.identity('session', r['session_id']), 'split': 'evaluation',
                'request': cleaned, 'context': [], 'candidate_tools': ids,
                'tool_catalog_version': catalog['catalog_version'], 'candidate_tools_reconstructed': True,
                'decision': None, 'acceptable_choices': [], 'observed_choices': [],
                'review_status': 'needs_human_review', 'execution_verified': False,
                'source': {'kind': 'real', 'source_alias': source_alias, 'record_type': 'standalone_root_discord_user_message',
                           'actor_kind': 'user',
                           'session_closed': r['ended_at'] is not None, 'historical_tool_replay': False,
                           'standalone_label_review_required': True,
                           'context_dependency_suspected': bool(DEICTIC.search(cleaned)),
                           'sanitized': True, 'human_origin_verified': False}})
        counts['candidate_requests'] = len(rows)
        summary = {'schema_version': 'frozen-eval-candidates-v1', 'status': 'candidate',
            'source': {'kind': 'real', 'alias': source_alias, 'snapshot_sha256': digest(''.join(source_fingerprints)),
                       'snapshot_semantics': 'Standalone active/uncompacted root user messages only; no historical replay'},
            'catalog_version': catalog['catalog_version'], 'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            'collector_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'private_policy_fingerprint': sanitizer.identity('policy', json.dumps(
                {'redactions': identities, 'excluded': key_record.get('privacy_exclude_ids', [])}, sort_keys=True)),
            'counts': dict(counts), 'target_cases': 100, 'target_per_tool': 10,
            'unique_session_groups': len({r['scenario_id'] for r in rows}),
            'previous_input_sha256': previous, 'label_status': 'unresolved',
            'limitations': ['Not a frozen evaluation set; no verified or adjudicated gold',
                'Tool affinity is not an intent label; developer discussion and context dependence need review',
                'Keyword-affinity selection is biased toward explicit tool/server mentions; not population representative',
                'Repeated source sessions are grouped, not independent requests for statistical inference',
                'Open sessions allowed only as standalone user-message snapshots; legacy collector policy unchanged',
                'Rule-based sanitization is not an anonymity guarantee; privacy review pending',
                'No model evaluated, no synthetic fill, no test scores used in selection']}
    finally:
        db.close()
    output.mkdir(parents=True, mode=0o700)
    path = output / 'cases.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        for r in rows: stream.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    summary['cases_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    private_json(output / 'candidate-manifest.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--catalog', type=Path, default=ROOT / 'datasets/hermes-ten-tools-v2/tools.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args()
    prior = [ROOT / 'datasets' / d / (s + '.jsonl') for d in ('hermes-routing-v1','mcp-routing-v2') for s in ('train','validation','test')]
    prior.append(ROOT / 'datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl')
    result = collect(args.source, args.catalog, args.output, args.key_file, prior)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'cases_sha256': result['cases_sha256']}))
