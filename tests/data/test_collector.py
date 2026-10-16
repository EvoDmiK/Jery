"""Synthetic SQLite fixtures only; no private Hermes content is stored here."""
import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from jery.data.collector import Collector
from jery.data.hermes import Sanitizer, read_only

CATALOG = {
    'catalog_version': 'test-v1',
    'tools': [{'id': 'mcp__vault__read', 'server': 'vault', 'name': 'read',
               'input_schema': {'properties': {'action': {'enum': ['search', 'read']}}, 'required': ['action']}}],
}


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.db'
        self.state = self.root / 'collector.db'
        self.catalog = self.root / 'tools.json'
        self.catalog.write_text(json.dumps(CATALOG))
        with sqlite3.connect(self.source) as db:
            db.executescript('''
                CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, ended_at REAL);
                CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
                    content TEXT, tool_calls TEXT, tool_call_id TEXT, active INTEGER DEFAULT 1,
                    compacted INTEGER DEFAULT 0);
                INSERT INTO sessions VALUES ('root-private', NULL, 100), ('child-private', 'root-private', 200), ('live', NULL, NULL);
            ''')
        self.add('root-private', 'user', 'Find alice@example.test in /Users/alice/private.md for Alice')
        self.first = self.add('root-private', 'assistant', 'CURRENT_REASON_NOT_INPUT', [self.call('one')])
        self.add('root-private', 'tool', '{"success": true, "text": "CURRENT_RESULT_NOT_INPUT"}', call_id='one')
        self.add('root-private', 'assistant', 'CURRENT_ANSWER_NOT_INPUT')
        self.add('child-private', 'user', 'Translate this sentence')
        self.add('child-private', 'assistant', 'Translation')
        self.add('live', 'user', 'Still running')
        self.add('live', 'assistant', 'Will finish later')
        self.before = self.source.read_bytes()
        self.collector = self.new()

    def new(self, **kwargs):
        return Collector(self.source, kwargs.pop('state', self.state), self.catalog,
                         reconstructed=kwargs.pop('reconstructed', True), identities=['Alice'], **kwargs)

    @staticmethod
    def call(identity, action='search', name='mcp_vault_read'):
        return {'id': identity, 'function': {'name': name, 'arguments': json.dumps({'action': action})}}

    def add(self, sid, role, content, calls=None, call_id=None):
        with sqlite3.connect(self.source) as db:
            return db.execute('INSERT INTO messages(session_id,role,content,tool_calls,tool_call_id) VALUES (?,?,?,?,?)',
                              (sid, role, content, json.dumps(calls) if calls is not None else None, call_id)).lastrowid

    def ack(self, batch):
        return self.collector.commit(self.collector.submission(batch))

    def test_source_read_only_privacy_context_and_parent_group(self):
        batch = self.collector.fetch()
        self.assertEqual(len(batch['examples']), 3)
        first, final, child = batch['examples']
        text = json.dumps(first)
        for private in ('alice@example.test', '/Users/alice', 'Alice', 'root-private', 'CURRENT_'):
            self.assertNotIn(private, text)
        self.assertEqual(first['context'], [])
        self.assertEqual(first['outcome'], 'ok')
        self.assertEqual(final['observed_choices'], [])
        self.assertEqual(first['session_group'], child['session_group'])
        self.assertEqual(first['candidate_tools'], ['mcp__vault__read'])
        self.assertTrue(first['candidate_tools_reconstructed'])
        self.ack(batch)
        self.assertEqual(self.before, self.source.read_bytes())
        with read_only(self.source) as db:
            with self.assertRaises(sqlite3.OperationalError):
                db.execute('DELETE FROM messages')
        self.assertEqual(self.source.read_bytes(), self.before)
        count = self.collector.export(self.root / 'queue.jsonl')
        self.assertEqual(count, 3)
        exported = [json.loads(line) for line in (self.root / 'queue.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['decision'] is None and r['acceptable_choices'] == [] and r['review_status'] == 'needs_human_review' for r in exported))
        with self.assertRaises(FileExistsError):
            self.collector.export(self.root / 'queue.jsonl')

    def test_pending_restart_concurrent_get_and_ack(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(lambda _: self.collector.fetch(), range(4)))
        self.assertTrue(all(b == batches[0] for b in batches))
        restarted = self.new()
        self.assertEqual(restarted.fetch(), batches[0])
        submitted = self.collector.submission(batches[0])
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(lambda _: restarted.commit(submitted), range(4)))
        self.assertTrue(all(r == receipts[0] for r in receipts))
        later = restarted.fetch()
        restarted.commit(restarted.submission(later))
        restarted.commit(submitted)
        with sqlite3.connect(self.state) as db:
            self.assertEqual(db.execute('SELECT checkpoint FROM consumers').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM review_candidates').fetchone()[0], 3)

    def test_commit_rejects_tampering_and_foreign_consumers(self):
        batch = self.collector.fetch()
        original = self.collector.submission(batch)
        for mutate in (
            lambda b: b.update(next_cursor='cursor_999'),
            lambda b: b.update(consumer='other'),
            lambda b: b['examples'][0].update(decision='select_tools'),
            lambda b: b['examples'][0]['observed_choices'][0].update(action='delete'),
            lambda b: b['examples'][0].update(request='changed'),
            lambda b: b.update(examples=[]),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                self.collector.commit(changed)
        self.assertEqual(self.collector.fetch(), batch)
        with sqlite3.connect(self.state) as db:
            self.assertEqual(db.execute('SELECT checkpoint FROM consumers').fetchone()[0], 0)
        self.ack(batch)

    def test_partial_storage_failure_rolls_back(self):
        batch = self.collector.fetch()
        with sqlite3.connect(self.state) as db:
            db.execute('''CREATE TRIGGER fail_second BEFORE INSERT ON review_candidates
                WHEN (SELECT COUNT(*) FROM review_candidates) >= 1
                BEGIN SELECT RAISE(ABORT, 'simulated disk error'); END''')
        with self.assertRaises(sqlite3.IntegrityError):
            self.ack(batch)
        with sqlite3.connect(self.state) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM review_candidates').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM seen').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT checkpoint FROM consumers').fetchone()[0], 0)
            db.execute('DROP TRIGGER fail_second')
        self.assertEqual(self.collector.fetch(), batch)
        self.ack(batch)

    def test_preserve_human_review_across_recollection(self):
        batch = self.collector.fetch()
        self.ack(batch)
        example = self.collector.submission(batch)['examples'][0]
        example.update(decision='unsupported', review_status='human_reviewed', review_comment='keep')
        with sqlite3.connect(self.state) as db:
            db.execute('UPDATE review_candidates SET payload=? WHERE id=?', (json.dumps(example), example['id']))
        other = self.collector.fetch('second-consumer')
        self.collector.commit(self.collector.submission(other))
        with sqlite3.connect(self.state) as db:
            stored = json.loads(db.execute('SELECT payload FROM review_candidates WHERE id=?', (example['id'],)).fetchone()[0])
        self.assertEqual(stored, example)

    def test_missing_candidates_quarantines_and_empty_ack_advances(self):
        held = self.new(state=self.root / 'held.db', reconstructed=False)
        batch = held.fetch()
        self.assertEqual(batch['examples'], [])
        self.assertEqual(held.fetch(), batch)
        held.commit(held.submission(batch))
        with sqlite3.connect(held.state_path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM quarantine').fetchone()[0], 3)
            self.assertEqual(db.execute('SELECT checkpoint FROM consumers').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT DISTINCT reason FROM quarantine').fetchone()[0], 'missing_historical_candidates')
        self.assertEqual(held.fetch()['examples'], [])

    def test_open_session_arrives_after_older_rows_consumed(self):
        self.ack(self.collector.fetch())
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE sessions SET ended_at=300 WHERE id='live'")
        batch = self.collector.fetch()
        self.assertEqual([e['request'] for e in batch['examples']], ['Still running'])

    def test_bad_calls_held_and_outcomes_remain_separate(self):
        self.ack(self.collector.fetch())
        self.add('root-private', 'assistant', '', [self.call('bad', action='delete')])
        self.add('root-private', 'assistant', '', [self.call('native', name='terminal')])
        self.add('root-private', 'assistant', '', [self.call('a'), self.call('b')])
        self.add('root-private', 'tool', '{"success": true}', call_id='a')
        self.add('root-private', 'tool', '{"error": "timeout"}', call_id='b')
        self.add('root-private', 'assistant', '', [self.call('unknown')])
        self.add('root-private', 'tool', 'plain text cannot prove success', call_id='unknown')
        batch = self.collector.fetch()
        self.assertEqual([e['outcome'] for e in batch['examples']], ['mixed', 'unknown'])
        with sqlite3.connect(self.state) as db:
            self.assertEqual({r[0] for r in db.execute('SELECT reason FROM quarantine')}, {'invalid_action', 'unknown_or_non_mcp_tool'})

    def test_rewritten_and_oversized_history_is_held(self):
        self.ack(self.collector.fetch())
        with sqlite3.connect(self.source) as db:
            db.execute("INSERT INTO sessions VALUES ('rewritten', NULL, 1), ('large', NULL, 1)")
        self.add('rewritten', 'user', 'old request')
        self.add('rewritten', 'assistant', 'old answer')
        self.add('large', 'user', '가' * 100000)
        self.add('large', 'assistant', 'answer')
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE messages SET active=0 WHERE session_id='rewritten' AND role='user'")
        batch = self.collector.fetch()
        self.assertEqual(batch['examples'], [])
        with sqlite3.connect(self.state) as db:
            self.assertEqual({r[0] for r in db.execute('SELECT reason FROM quarantine')}, {'rewritten_history', 'oversized_record'})
        self.ack(batch)

    def test_state_source_binding_and_alias_rejection(self):
        with self.assertRaises(ValueError):
            self.new(reconstructed=False)
        with self.assertRaises(ValueError):
            self.new(state=self.source)
        import os
        alias = self.root / 'alias.db'
        os.link(self.source, alias)
        with self.assertRaises(ValueError):
            self.new(state=alias)
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_sensitive_patterns(self):
        sanitizer = Sanitizer(b'test-key', ['Private Person'])
        raw = 'token: opaque-private-value\nPrivate Person alice@example.test /Users/alice/a.md notes/secret.md 010-1234-5678\nAuthorization: Bearer abc\napi_key="privatevalue"\nhttps://x.test/?token=abc\nsk-abcdefghijklmnop\n'
        cleaned = sanitizer.text(raw)
        self.assertEqual(sanitizer.text('가' * 80000), '가' * 80000)
        for value in ('opaque-private-value', 'Private Person', 'alice@example.test', '/Users/alice', 'secret.md', '010-1234-5678', 'abc', 'privatevalue', 'x.test', 'abcdefghijklmnop'):
            self.assertNotIn(value, cleaned)
        self.assertEqual(sanitizer.text('alice@example.test'), sanitizer.text('alice@example.test'))


if __name__ == '__main__':
    unittest.main()
