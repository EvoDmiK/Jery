"""Read completed Hermes sessions without writing to the source database."""
import hashlib
import hmac
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


class Held(ValueError):
    """A fixed reason code; never contains source text."""


class Sanitizer:
    def __init__(self, key, identities=()):
        self.key = key
        if any(not isinstance(s, str) or not s.strip() for s in identities):
            raise ValueError('Redactions must be nonempty strings')
        self.identities = sorted(identities, key=len, reverse=True)

    def identity(self, kind, value):
        digest = hmac.new(self.key, str(value).encode(), hashlib.sha256).hexdigest()[:32]
        return f'{kind}_{digest}'

    def text(self, value):
        if not isinstance(value, str):
            raise Held('non_text_content')
        text = value
        for identity in self.identities:
            text = text.replace(identity, '[' + self.identity('private', identity) + ']')
        # Whole credential lines are discarded rather than guessing where a secret ends.
        text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----', '[secret]', text)
        text = re.sub(r'(?im)(?:authorization|cookie|password|passwd|secret|token|credentials?|(?:api|access|refresh|auth)[_-]?(?:key|token))\s*[\"\x27]?\s*[:=][^\n]*', '[secret]', text)
        text = re.sub(r'(?i)\bBearer\s+\S+|\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_\-]+|\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+', '[secret]', text)
        patterns = (
            ('email', r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
            ('url', r'https?://[^\s<>\"\x27]+'),
            ('path', r'(?:[A-Za-z]:\\|(?<!\w)/|\~/)[^\s<>\"\x27,;]+'),
            ('relative_path', r'(?<![^\s<>\"\x27,;])(?:[^\s/<>\"\x27,;]+/)+[^\s/<>\"\x27,;]+'),
            ('document', r'(?<![^\s/<>\"\x27])[^\s/<>\"\x27]+\.(?:md|pdf|docx?|xlsx?|csv|jsonl?|txt)\b'),
            ('phone', r'(?<!\w)(?:\+?82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\w)'),
            ('user_id', r'(?<!\d)\d{17,20}(?!\d)'),
            ('personal_id', r'(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)'),
            ('ip', r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        )
        for kind, pattern in patterns:
            text = re.sub(pattern, lambda m: '[' + self.identity(kind, m[0]) + ']', text, flags=re.I)
        return text


def load_catalog(path):
    catalog = json.loads(Path(path).read_text())
    if not isinstance(catalog.get('catalog_version'), str) or not catalog['catalog_version']:
        raise ValueError('Missing catalog version')
    tools = {}
    for tool in catalog['tools']:
        if not re.fullmatch(r'mcp__[A-Za-z0-9_-]+__[A-Za-z0-9_-]+', tool['id']):
            raise ValueError('Invalid tool ID')
        if tool['id'] in tools:
            raise ValueError('Duplicate tool ID')
        tools[tool['id']] = tool
    if not tools:
        raise ValueError('Empty catalog')
    return catalog, tools


def read_only(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    return connection


class HermesSource:
    def __init__(self, path, sanitizer, catalog, reconstructed=False):
        self.path, self.sanitizer = Path(path).resolve(), sanitizer
        self.catalog, self.tools = catalog, {t['id']: t for t in catalog['tools']}
        self.reconstructed = reconstructed
        self.aliases = {}
        for tool in catalog['tools']:
            # Hermes MCP names use mcp_{server}_{name}; canonical dataset IDs use __.
            for alias in (tool['id'], f"mcp_{tool['server']}_{tool['name']}"):
                if alias in self.aliases and self.aliases[alias] != tool['id']:
                    raise ValueError('Ambiguous tool alias')
                self.aliases[alias] = tool['id']
        with closing(read_only(self.path)) as db:
            for table, required in (
                ('sessions', {'id', 'parent_session_id', 'ended_at'}),
                ('messages', {'id', 'session_id', 'role', 'content', 'tool_calls', 'tool_call_id'}),
            ):
                columns = {r['name'] for r in db.execute(f'PRAGMA table_info({table})')}
                if not required <= columns:
                    raise ValueError('Unsupported Hermes database schema')

    def scan(self, seen, limit):
        # ponytail: rescan completed message IDs per fetch; add a local source index if logs outgrow this.
        with closing(read_only(self.path)) as db:
            db.execute('BEGIN')
            sessions = {r['id']: dict(r) for r in db.execute('SELECT id, parent_session_id, ended_at FROM sessions')}
            rows = db.execute("SELECT m.id, m.session_id FROM messages m JOIN sessions s ON s.id=m.session_id WHERE m.role='assistant' AND s.ended_at IS NOT NULL ORDER BY m.id")
            selected, quarantined, keys, cached = [], [], [], {}
            total_bytes = 0
            for row in rows:
                key = self.sanitizer.identity('example', row['id'])
                if key in seen:
                    continue
                keys.append(key)
                try:
                    sid = row['session_id']
                    if sid not in cached:
                        cached[sid] = [dict(r) for r in db.execute('SELECT * FROM messages WHERE session_id=? ORDER BY id', (sid,))]
                    messages = cached[sid]
                    index = next(i for i, m in enumerate(messages) if m['id'] == row['id'])
                    example = self.example(key, sid, sessions, messages, index)
                    size = len(json.dumps(example, ensure_ascii=False).encode())
                    if size > 256 * 1024:
                        raise Held('oversized_record')
                    if total_bytes + size > 8 * 1024 * 1024:
                        keys.pop()
                        break
                    total_bytes += size
                    selected.append(example)
                except Held as exc:
                    quarantined.append((key, str(exc)))
                except (ValueError, TypeError, KeyError):
                    quarantined.append((key, 'malformed_record'))
                if len(keys) >= limit:
                    break
            return selected, quarantined, keys

    def example(self, key, sid, sessions, messages, index):
        row = messages[index]
        if any(isinstance(m.get('content'), str) and len(m['content'].encode()) > 256 * 1024 for m in messages[:index]):
            raise Held('oversized_record')
        if any(m.get('active', 1) != 1 or m.get('compacted', 0) or m.get('_compressed_summary', 0) for m in messages[:index + 1]):
            raise Held('rewritten_history')
        if not self.reconstructed:
            raise Held('missing_historical_candidates')
        parents = set()
        root = sid
        while True:
            if root in parents:
                raise Held('cyclic_session_parent')
            parents.add(root)
            if root not in sessions:
                raise Held('missing_session_parent')
            parent = sessions[root]['parent_session_id']
            if not parent:
                break
            root = parent
        user = next((i for i in range(index - 1, -1, -1) if messages[i]['role'] == 'user'), None)
        if user is None or not messages[user]['content']:
            raise Held('missing_request')
        calls = json.loads(row['tool_calls']) if row['tool_calls'] else []
        if not isinstance(calls, list):
            raise Held('malformed_calls')
        choices, results, call_ids = [], [], set()
        following = []
        for message in messages[index + 1:]:
            if message['role'] in ('assistant', 'user'):
                break
            if message['role'] == 'tool' and message.get('active', 1) == 1 and not message.get('compacted', 0):
                following.append(message)
        for call in calls:
            fn = call['function']
            tool_id = self.aliases.get(fn['name'])
            if tool_id is None:
                raise Held('unknown_or_non_mcp_tool')
            arguments = fn.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise Held('malformed_arguments')
            choice = {'tool': tool_id}
            action = arguments.get('action')
            schema = self.tools[tool_id]['input_schema']
            action_schema = schema.get('properties', {}).get('action')
            if action is not None:
                if not action_schema or action not in action_schema.get('enum', []):
                    raise Held('invalid_action')
                choice['action'] = action
            elif 'action' in schema.get('required', []):
                raise Held('missing_action')
            choices.append(choice)
            call_id = call.get('id')
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                raise Held('invalid_call_identity')
            call_ids.add(call_id)
            matches = [m for m in following if m['tool_call_id'] == call_id]
            # Only explicit JSON status is evidence; free text and timeouts remain unknown.
            status = 'unknown'
            if len(matches) == 1:
                try:
                    result = json.loads(matches[0]['content'])
                    if isinstance(result, dict):
                        if result.get('is_error') is True or result.get('error') or result.get('success') is False:
                            status = 'error'
                        elif result.get('success') is True or result.get('is_error') is False:
                            status = 'ok'
                except (ValueError, TypeError):
                    pass
            results.append(status)
        outcome = 'unknown'
        if results and 'unknown' not in results:
            outcome = results[0] if len(set(results)) == 1 else 'mixed'
        context = []
        for i, message in enumerate(messages[:index]):
            if i == user or message['role'] not in ('user', 'assistant', 'tool'):
                continue
            content = message['content']
            if content:
                context.append({'role': message['role'], 'content': self.sanitizer.text(content)})
        return {
            'id': key, 'session_group': self.sanitizer.identity('session', root),
            'request': self.sanitizer.text(messages[user]['content']), 'context': context,
            'candidate_tools': list(self.tools), 'tool_catalog_version': self.catalog['catalog_version'],
            'candidate_tools_reconstructed': True, 'observed_choices': choices,
            'outcome': outcome, 'source': 'hermes_log',
        }
