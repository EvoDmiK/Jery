import json
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_laya_full_dataset import build


class FullExportTest(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        source = ROOT / 'datasets/mcp-routing-v2'
        catalog = (source / 'tools.json').read_bytes()
        (root / 'tools.json').write_bytes(catalog)
        manifest = {'sha256': {'tools.json': hashlib.sha256(catalog).hexdigest()}, 'split_counts': {}}
        for split in ('train', 'validation', 'test'):
            rows = [json.loads(line) for line in (source / (split + '.jsonl')).read_text().splitlines()]
            selected = [next(r for r in rows if len(r['candidate_tools']) == size) for size in (96, 8)]
            content = ''.join(json.dumps(r) + '\n' for r in selected).encode()
            (root / (split + '.jsonl')).write_bytes(content)
            manifest['sha256'][split + '.jsonl'] = hashlib.sha256(content).hexdigest()
            manifest['split_counts'][split] = len(selected)
        (root / 'manifest.json').write_text(json.dumps(manifest))

    def test_full_coverage_and_original_files_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, out = root / 'source', root / 'out'
            self.fixture(source)
            original = {p: p.read_bytes() for p in source.iterdir()}
            manifest = build(source, out)
            for split in ('train', 'validation', 'test'):
                raw = {r['id']: r for r in map(json.loads, (source / (split + '.jsonl')).read_text().splitlines())}
                grouped = {}
                for row in map(json.loads, (out / (split + '.jsonl')).read_text().splitlines()):
                    meta = row['metadata']; grouped.setdefault(meta['source_id'], []).append(meta)
                    self.assertEqual(set(row['state']), {'request', 'context', 'candidates'})
                    self.assertNotIn('source_id', json.dumps(row['questions']))
                self.assertEqual(set(grouped), set(raw))
                for rid, windows in grouped.items():
                    tools = [t for w in windows for t in w['window_tools']]
                    self.assertEqual(len(tools), len(set(tools)))
                    self.assertEqual(set(tools), set(raw[rid]['candidate_tools']))
                    self.assertEqual([w['window_index'] for w in windows], list(range(len(windows))))
                    self.assertTrue(all(w['window_count'] == len(windows) for w in windows))
                self.assertEqual(manifest['splits'][split]['source_rows'], 2)
                self.assertEqual(manifest['splits'][split]['rows'], 13)
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            with self.assertRaises(ValueError): build(source, out)

    def test_partitions_and_model_input_do_not_use_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'
            self.fixture(source); build(source, root / 'before')
            manifest = json.loads((source / 'manifest.json').read_text())
            for split in ('train', 'validation', 'test'):
                path = source / (split + '.jsonl')
                rows = list(map(json.loads, path.read_text().splitlines()))
                for row in rows:
                    row['decision'] = 'no_tool'; row['acceptable_choices'] = []
                content = ''.join(json.dumps(r) + '\n' for r in rows).encode()
                path.write_bytes(content); manifest['sha256'][path.name] = hashlib.sha256(content).hexdigest()
            (source / 'manifest.json').write_text(json.dumps(manifest))
            build(source, root / 'after')
            for split in ('train', 'validation', 'test'):
                a = list(map(json.loads, (root / 'before' / (split + '.jsonl')).read_text().splitlines()))
                b = list(map(json.loads, (root / 'after' / (split + '.jsonl')).read_text().splitlines()))
                for before, after in zip(a, b):
                    self.assertEqual(before['state'], after['state'])
                    self.assertEqual(before['questions'], after['questions'])
                    self.assertEqual(before['metadata']['window_tools'], after['metadata']['window_tools'])


if __name__ == '__main__':
    unittest.main()
