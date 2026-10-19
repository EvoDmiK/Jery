import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('language_probe', Path(__file__).resolve().parents[2] / 'scripts/evaluate_laya_language_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class LanguageProbeTest(unittest.TestCase):
    def test_only_request_changes_and_source_is_preserved(self):
        row = {'state': {'request': '요약해줘', 'context': [], 'candidates': [{'name': 'same'}]},
               'questions': {'same': 'schema'}, 'gold': {'same': 'label'},
               'metadata': {'source_id': 'a', 'window_count': 1}}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.jsonl'
            source.write_text(json.dumps(row) + '\n')
            original = source.read_bytes()
            specification = {'source_sha256': hashlib.sha256(original).hexdigest(),
                             'pairs': [{'source_id': 'a', 'english_request': 'Summarize it'}]}
            ko, en = probe.paired_rows(source, specification)
            expected = copy.deepcopy(row)
            expected['state']['request'] = 'Summarize it'
            self.assertEqual(ko, [row])
            self.assertEqual(en, [expected])
            self.assertEqual(source.read_bytes(), original)
            specification['source_sha256'] = 'wrong'
            with self.assertRaises(ValueError): probe.paired_rows(source, specification)
