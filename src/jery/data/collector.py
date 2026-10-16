"""Durable pending batches and atomic review queue/checkpoint commits."""
import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .hermes import HermesSource, Sanitizer, load_catalog

SCHEMA = 'jery-log-candidates-v1'


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class Collector:
    def __init__(self, source_path, state_path, catalog_path, *, reconstructed=False, identities=()):
        self.state_path = Path(state_path).resolve()
        source_path = Path(source_path).resolve()
        if self.state_path == source_path or (self.state_path.exists() and os.path.samefile(source_path, self.state_path)):
            raise ValueError('Source and state must be different files')
        catalog, _ = load_catalog(catalog_path)
        identities = list(identities)
        # Binding prevents a changed source, policy, or candidate snapshot from silently reusing IDs/cursors.
        binding = hashlib.sha256(encode([str(source_path), catalog, reconstructed, identities]).encode()).hexdigest()
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.state_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.state_path, 0o600)
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            existing = db.execute("SELECT value FROM config WHERE key='binding'").fetchone()
            if existing and existing[0] != binding:
                raise ValueError('State belongs to a different source or policy')
            db.execute('INSERT OR IGNORE INTO config VALUES (?,?)', ('binding', binding))
            db.execute('INSERT OR IGNORE INTO config VALUES (?,?)', ('schema_version', 'jery-collector-state-v1'))
            db.execute('INSERT OR IGNORE INTO config VALUES (?,?)', ('key', secrets.token_hex(32)))
            key = bytes.fromhex(db.execute("SELECT value FROM config WHERE key='key'").fetchone()[0])
            db.execute('CREATE TABLE IF NOT EXISTS catalogs (version TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO catalogs VALUES (?,?)', (catalog['catalog_version'], encode(catalog)))
            db.execute('CREATE TABLE IF NOT EXISTS consumers (consumer TEXT PRIMARY KEY, checkpoint INTEGER NOT NULL DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, consumer TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, source_keys TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0, UNIQUE(consumer, sequence))')
            db.execute('CREATE TABLE IF NOT EXISTS seen (consumer TEXT NOT NULL, source_key TEXT NOT NULL, PRIMARY KEY(consumer, source_key))')
            db.execute('CREATE TABLE IF NOT EXISTS quarantine (id TEXT PRIMARY KEY, reason TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS review_candidates (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.source = HermesSource(source_path, Sanitizer(key, identities), catalog, reconstructed)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.state_path, timeout=30, isolation_level=None)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def consumer(value):
        import re
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value):
            raise ValueError('Invalid consumer')
        return value

    def fetch(self, consumer='jery-dataset-v1', limit=100):
        self.consumer(consumer)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Limit must be between 1 and 1000')
        with self.transaction() as db:
            pending = db.execute('SELECT payload FROM batches WHERE consumer=? AND committed=0', (consumer,)).fetchone()
            if pending:
                return json.loads(pending[0])
            db.execute('INSERT OR IGNORE INTO consumers(consumer) VALUES (?)', (consumer,))
            sequence = db.execute('SELECT checkpoint FROM consumers WHERE consumer=?', (consumer,)).fetchone()[0] + 1
            seen = {r[0] for r in db.execute('SELECT source_key FROM seen WHERE consumer=?', (consumer,))}
            examples, held, keys = self.source.scan(seen, limit)
            for key, reason in held:
                db.execute('INSERT OR IGNORE INTO quarantine VALUES (?,?)', (key, reason))
            payload = {
                'schema_version': SCHEMA, 'consumer': consumer,
                'batch_id': self.source.sanitizer.identity('batch', f'{consumer}:{sequence}'),
                'next_cursor': f'cursor_{sequence}', 'sanitized': True, 'examples': examples,
            }
            db.execute('INSERT INTO batches(id,consumer,sequence,payload,source_keys) VALUES (?,?,?,?,?)',
                       (payload['batch_id'], consumer, sequence, encode(payload), encode(keys)))
            return payload

    @staticmethod
    def submission(batch):
        payload = {k: v for k, v in batch.items() if k != 'sanitized'}
        payload['examples'] = [dict(r, decision=None, acceptable_choices=[], review_status='needs_human_review') for r in batch['examples']]
        return payload

    def commit(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError('Expected batch object')
        consumer = self.consumer(submitted.get('consumer'))
        batch_id = submitted.get('batch_id')
        if not isinstance(batch_id, str):
            raise ValueError('Missing batch ID')
        with self.transaction() as db:
            row = db.execute('SELECT sequence,payload,source_keys,committed FROM batches WHERE id=? AND consumer=?', (batch_id, consumer)).fetchone()
            if row is None:
                raise ValueError('Unknown batch')
            sequence, payload, source_keys, committed = row
            batch = json.loads(payload)
            # Compare against the server snapshot, rejecting changed text, labels, tools, actions and cursor.
            if submitted != self.submission(batch):
                raise ValueError('Batch differs from issued snapshot')
            receipt = {'batch_id': batch_id, 'next_cursor': batch['next_cursor'], 'committed': True}
            if committed:
                return receipt
            checkpoint = db.execute('SELECT checkpoint FROM consumers WHERE consumer=?', (consumer,)).fetchone()[0]
            if sequence != checkpoint + 1:
                raise ValueError('Out of order batch')
            for example in submitted['examples']:
                # Human-edited payloads already in the queue are intentionally preserved.
                db.execute('INSERT OR IGNORE INTO review_candidates VALUES (?,?)', (example['id'], encode(example)))
            db.executemany('INSERT OR IGNORE INTO seen VALUES (?,?)', [(consumer, key) for key in json.loads(source_keys)])
            db.execute('UPDATE consumers SET checkpoint=? WHERE consumer=?', (sequence, consumer))
            db.execute('UPDATE batches SET committed=1 WHERE id=?', (batch_id,))
            return receipt

    def status(self):
        with self.transaction() as db:
            return {
                'schema_version': 'jery-collector-state-v1',
                'review_candidates': db.execute('SELECT COUNT(*) FROM review_candidates').fetchone()[0],
                'quarantine': dict(db.execute('SELECT reason, COUNT(*) FROM quarantine GROUP BY reason')),
                'pending_batches': db.execute('SELECT COUNT(*) FROM batches WHERE committed=0').fetchone()[0],
            }

    def export(self, path):
        """Export the current review queue; never overwrite an existing file."""
        target = Path(path).resolve()
        # Read a coherent snapshot before opening the output (which might alias a database).
        with self.transaction() as db:
            rows = list(db.execute('SELECT payload FROM review_candidates ORDER BY id'))
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            for row in rows:
                stream.write(row[0] + '\n')
        return len(rows)
