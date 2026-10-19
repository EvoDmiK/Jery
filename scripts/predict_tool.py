"""Classify a local request using a trained Jery model; never execute MCP tools."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jery.runtime.prediction import model_input, validate_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'datasets/mcp-routing-v2/tools.json')
    parser.add_argument('--context', type=Path, help='JSON array of prior messages; default empty')
    parser.add_argument('--candidates', type=Path, help='JSON array of allowed tool IDs; default catalog tools')
    parser.add_argument('--model', type=Path, help='Saved model; default configured latest artifact')
    args = parser.parse_args()
    try:
        catalog = json.loads(args.catalog.read_text())
        context = json.loads(args.context.read_text()) if args.context else []
        candidates = json.loads(args.candidates.read_text()) if args.candidates else [t['id'] for t in catalog['tools']]
        features = model_input({'request': args.request, 'context': context, 'candidate_tools': candidates}, catalog)
        from jery.model import predict
        result = predict(**features, model=args.model)
        validate_prediction(result, features['candidate_tools'], features['catalog'])
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
