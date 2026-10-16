"""Add the training-catalog TF-IDF state to router artifacts saved before it was persisted.

PYTHONPATH=src python3 -m jery.model.augment_index --catalog datasets/mcp-routing-v2/tools.json ARTIFACT...

Only the `index` key (and provenance in meta) is added. Config, weights and actions are untouched;
the catalog file must match the tools.json hash recorded in the artifact at training time.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from .features import TfidfIndex
from .predict import read_artifact


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def augment(artifact, catalog_path):
    artifact, catalog_path = Path(artifact), Path(catalog_path)
    data = read_artifact(artifact)
    expected = data.get('meta', {}).get('sha256', {}).get('tools.json')
    actual = sha256(catalog_path)
    if expected != actual:
        raise SystemExit('%s: catalog %s does not match training tools.json (%s)' % (artifact, actual, expected))
    state = TfidfIndex(json.loads(catalog_path.read_text())).state()
    before = sha256(artifact)
    if 'index' in data:
        if data['index'] != state:
            raise SystemExit('%s: existing index state differs from the training catalog' % artifact)
        return {'artifact': str(artifact), 'status': 'already_has_index', 'sha256': before}
    frozen = {k: v for k, v in data.items() if k not in ('index', 'meta')}
    data['index'] = state
    data['meta']['index_augmented'] = {
        'at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'from_catalog_sha256': actual, 'previous_artifact_sha256': before,
        'note': 'index state recomputed from the original training catalog; weights unchanged'}
    backup = artifact.with_name(artifact.name + '.pre-index')
    if not backup.exists():
        shutil.copyfile(artifact, backup)
    tmp = artifact.with_name(artifact.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True)
    check = read_artifact(tmp)
    if {k: v for k, v in check.items() if k not in ('index', 'meta')} != frozen:
        tmp.unlink()
        raise SystemExit('%s: augmentation changed model parameters' % artifact)
    tmp.replace(artifact)
    return {'artifact': str(artifact), 'status': 'augmented', 'previous_sha256': before,
            'sha256': sha256(artifact), 'backup': str(backup)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('artifacts', nargs='+')
    args = parser.parse_args(argv)
    for artifact in args.artifacts:
        print(json.dumps(augment(artifact, args.catalog), ensure_ascii=False))


if __name__ == '__main__':
    main()
