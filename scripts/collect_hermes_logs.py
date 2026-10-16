"""Run the local collector API or one offline collection/ACK cycle."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jery.data.collector import Collector
from jery.runtime.collector_api import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve', 'once', 'export', 'status'))
    parser.add_argument('--source', required=True, help='Hermes state.db (read-only)')
    parser.add_argument('--state', required=True, help='Separate private collector SQLite file')
    parser.add_argument('--catalog', required=True, help='Curated tools.json snapshot')
    parser.add_argument('--reconstruct-candidates', action='store_true', help='Explicitly use all catalog tools as reconstructed candidates')
    parser.add_argument('--redactions', help='Private JSON list of names/identifiers to pseudonymize')
    parser.add_argument('--consumer', default='jery-dataset-v1')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--output', help='New local JSONL file, required for export')
    args = parser.parse_args()
    identities = json.loads(Path(args.redactions).read_text()) if args.redactions else []
    if not isinstance(identities, list):
        parser.error('--redactions must contain a JSON list')
    collector = Collector(args.source, args.state, args.catalog, reconstructed=args.reconstruct_candidates, identities=identities)
    if args.command == 'serve':
        with make_server(collector, os.environ.get('JERY_COLLECTOR_TOKEN', ''), args.host, args.port) as server:
            print(f'Collector listening on {args.host}:{server.server_port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    elif args.command == 'once':
        batch = collector.fetch(args.consumer, args.limit)
        collector.commit(collector.submission(batch))
        print(json.dumps({'committed': True, 'candidate_count': len(batch['examples']), **collector.status()}))
    elif args.command == 'status':
        print(json.dumps(collector.status()))
    else:
        if not args.output:
            parser.error('--output is required for export')
        print(json.dumps({'exported': collector.export(args.output)}))


if __name__ == '__main__':
    main()
