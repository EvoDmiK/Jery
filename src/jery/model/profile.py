"""Split profile of label coverage for reporting (analysis only; never used as model features).

PYTHONPATH=src python3 -m jery.model.profile [--data-dir datasets/mcp-routing-v2]
"""
import argparse
from collections import Counter
import json
from pathlib import Path

SPLITS = ('train', 'validation', 'test')


def dataset_profile(splits, catalog):
    servers = {t['id']: t['server'] for t in catalog['tools']}
    tools = {s: {c['tool'] for r in rows for c in r['acceptable_choices']} for s, rows in splits.items()}
    train_actions = {(c['tool'], c['action']) for r in splits['train'] for c in r['acceptable_choices'] if c.get('action')}
    selected_requests = {}
    for rows in splits.values():
        for r in rows:
            if r['decision'] == 'select_tools':
                selected_requests.setdefault(r['request'], set()).update(servers[c['tool']] for c in r['acceptable_choices'])
    out = {'train_only_tools': sorted(tools['train'] - tools['validation'] - tools['test']), 'splits': {}}
    for s, rows in splits.items():
        select = [r for r in rows if r['decision'] == 'select_tools']
        share = Counter(servers[r['acceptable_choices'][0]['tool']] for r in select)
        unsupported = [r for r in rows if r['decision'] == 'unsupported']
        # Missing-target: the same request is select_tools elsewhere, but its tool was removed from candidates.
        missing = Counter(sorted(selected_requests[r['request']])[0] for r in unsupported if r['request'] in selected_requests)
        unseen = Counter('%s:%s' % (c['tool'], c['action']) for r in rows for c in r['acceptable_choices']
                         if c.get('action') and (c['tool'], c['action']) not in train_actions)
        out['splits'][s] = {
            'examples': len(rows),
            'decision_support': dict(Counter(r['decision'] for r in rows)),
            'gold_tools': len(tools[s]),
            'select_server_share': {k: round(v / len(select), 3) for k, v in sorted(share.items())},
            'unsupported_missing_target_by_server': dict(missing),
            'unsupported_out_of_catalog': sum(1 for r in unsupported if r['request'] not in selected_requests),
            'unseen_train_actions': dict(unseen) if s != 'train' else {},
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='datasets/mcp-routing-v2')
    args = parser.parse_args(argv)
    d = Path(args.data_dir)
    splits = {s: [json.loads(x) for x in (d / (s + '.jsonl')).read_text().splitlines() if x.strip()] for s in SPLITS}
    print(json.dumps(dataset_profile(splits, json.loads((d / 'tools.json').read_text())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
