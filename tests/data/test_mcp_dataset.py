"""Check reproducibility and reject corrupted or leaking routing datasets."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_mcp_dataset import build
from validate_mcp_dataset import validate


class DatasetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'dataset'
        shutil.copytree(ROOT / 'datasets/mcp-routing-v2', self.root)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def rehash(self, name):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        manifest['sha256'][name] = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        self.write('manifest.json', manifest)

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return validate(self.root)

    def test_saved_inputs_reproduce_exactly(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir() if p.suffix in ('.json', '.jsonl')}
        build(self.root)
        self.check()
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in before})

    def test_modified_file_fails_checksum(self):
        with (self.root / 'train.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(AssertionError, 'Checksum'):
            self.check()

    def test_candidate_outside_list_is_rejected_even_with_updated_checksum(self):
        path = self.root / 'train.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(r for r in rows if r['acceptable_choices'])
        target = row['acceptable_choices'][0]['tool']
        row['candidate_tools'].remove(target)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        self.rehash('train.jsonl')
        with self.assertRaisesRegex(AssertionError, 'outside candidates'):
            self.check()

    def test_family_cannot_be_split_and_failed_build_preserves_outputs(self):
        path = self.root / 'scenarios.json'
        seeds = json.loads(path.read_text())
        variants = [s for s in seeds if s['scenario_id'] == 'obsidian_vault.list_resources.1']
        self.assertGreater(len(variants), 1)
        variants[0]['split'] = 'test'
        self.write('scenarios.json', seeds)
        before = (self.root / 'train.jsonl').read_bytes()
        with self.assertRaisesRegex(AssertionError, 'Scenario leakage'):
            build(self.root)
        self.assertEqual(before, (self.root / 'train.jsonl').read_bytes())

    def test_invalid_action_fails_before_build(self):
        seeds = json.loads((self.root / 'scenarios.json').read_text())
        row = next(s for s in seeds if any('action' in c for c in s['acceptable_choices']))
        row['acceptable_choices'][0]['action'] = 'invented_action'
        self.write('scenarios.json', seeds)
        with self.assertRaisesRegex(AssertionError, 'Invalid action'):
            build(self.root)

    def test_server_label_must_match_tool_catalog(self):
        labels = json.loads((self.root / 'labels.json').read_text())
        labels[0]['server'] = 'wrong_server'
        self.write('labels.json', labels)
        self.rehash('labels.json')
        with self.assertRaisesRegex(AssertionError, 'Server label mismatch'):
            self.check()


if __name__ == '__main__':
    unittest.main()
