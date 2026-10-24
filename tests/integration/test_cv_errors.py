import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from compare_laya_cv_errors import compare, metrics


class OofComparisonTest(unittest.TestCase):
    def test_paired_predictions_validate_coverage_gold_and_selected_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def write(path, obj, lines=False):
                p = root / path; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(''.join(json.dumps(r) + '\n' for r in obj) if lines else json.dumps(obj))
                return hashlib.sha256(p.read_bytes()).hexdigest()
            folds = []
            for fold in (1, 2):
                row = {'metadata': {'source_id': str(fold), 'scenario_id': str(fold),
                       'acceptable_choices': [{'tool': 'a'}], 'alias_to_tool': {'T0': 'a'},
                       'original_candidate_tools': ['a', 'b'], 'review_status': 'needs_human_review'},
                       'gold': {'route_in_window': {'label': 'T0'}}, 'state': {'request': 'synthetic'}}
                sha = write(f'folds/fold-{fold}/validation.jsonl', [row], True)
                folds.append({'fold': fold, 'splits': {'validation': {'sha256': sha}}})
            fold_sha = write('folds/folds.json', {'n_splits': 2, 'folds': folds})
            for index in (1, 9):
                children = []
                for fold in (1, 2):
                    name = f'{index}-{fold}'; correct = index == 9
                    children.append({'index': fold, 'run_id': name, 'local_id': name})
                    write(f'out/{name}/validation_cases.jsonl', [{'source_id': str(fold), 'tool': 'a' if correct else 'b', 'decision': 'select_tools'}], True)
                    write(f'out/{name}/report.json', {'validation': {'case_accuracy': int(correct)}, 'selection': {'selected_epoch': 1}})
                write(f'state/trial-{index:02d}/folds/state.json', {'status': 'finished', 'children': children})
            write('state/state.json', {'parent_run_id': 'parent', 'fold_summary_sha256': fold_sha,
                  'plan': {'cv': {'folds_dir': 'folds'}}, 'config': {'outputs_dir': 'out'},
                  'children': [{'index': i, 'run_id': str(i), 'status': 'finished'} for i in (1, 9)]})
            summary, cases = compare(root, root / 'state')
            self.assertEqual(summary['paired'], {'improved': 2})
            self.assertEqual(summary['cases'], 2)
            prediction = root / 'out/9-1/validation_cases.jsonl'
            good = prediction.read_text()
            for bad, message in [('', 'coverage'), (good + good, 'coverage'),
                                 (good.replace('"a"', '"outside"'), 'candidate'),
                                 (good.replace('"a"', '"b"'), 'selected-head')]:
                prediction.write_text(bad)
                with self.assertRaisesRegex(ValueError, message): compare(root, root / 'state')
                prediction.write_text(good)

    def test_macro_f1_averages_tools_not_decisions(self):
        cases = [{'gold': g, 'pred': p} for g, p in [('a', 'a'), ('a', 'b'), ('b', 'b')]]
        result = metrics(cases, 'pred')
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1'], 2 / 3)
        self.assertEqual(result['per_tool']['a']['support'], 2)


if __name__ == '__main__':
    unittest.main()
