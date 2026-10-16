"""Public prediction adapter for docs/contracts/prediction-v1.md.

predict(request, context, candidate_tools, catalog, model=None)
  -> {"decision", "choices": [{"tool", "action"?}], "ranked_tools": [tool_id, ...], "confidence"?}
"""
import gzip
import json
import os
from pathlib import Path

from jery.runtime.prediction import model_input as contract_input, validate_prediction

from .routers import LinearRouter, RetrievalRouter

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / 'models/mcp-routing-v2/latest/model.json.gz'
_cache = {}


def save_model(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.to_dict()
    data['meta'] = meta or {}
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True)


def read_artifact(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def load_model(path):
    """Restore a router, including its training-time TF-IDF state; no catalog is needed or refit."""
    data = read_artifact(path)
    cls = RetrievalRouter if data['config']['kind'] == 'retrieval' else LinearRouter
    return cls.from_dict(data)


def _cached_model(path):
    """Cache by resolved path plus file identity so a replaced or rewritten artifact is reloaded."""
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _cache:
        for old in [k for k in _cache if k[0] == key[0]]:
            del _cache[old]
        _cache[key] = load_model(path)
    return _cache[key]


def _as_catalog(catalog):
    """Accept tools.json, a list of specs or {id: spec}; the runtime contract takes {'tools': [...]}."""
    if isinstance(catalog, dict) and 'tools' in catalog:
        return catalog
    if isinstance(catalog, dict):
        return {'tools': [dict(spec, id=tool_id) for tool_id, spec in catalog.items()]}
    if isinstance(catalog, list):
        return {'tools': catalog}
    raise ValueError('catalog must be tools.json, a spec list or an id -> spec mapping')


def to_contract(internal):
    """Internal router output (acceptable_choices + scored ranking) -> prediction-v1."""
    out = {'decision': internal['decision'],
           'choices': [dict(c) for c in internal['acceptable_choices']],
           'ranked_tools': [r['tool'] for r in internal.get('ranked_tools') or []]}
    if 'confidence' in internal:
        out['confidence'] = internal['confidence']
    return out


def predict(request, context, candidate_tools, catalog, model=None):
    """Route one turn under prediction-v1.

    request: nonempty str. context: [{"role": "user"|"assistant"|"tool", "content": str}] before the
    tool choice. candidate_tools: unique tool ids (may be empty). catalog: tools.json dict (or a spec
    list / {id: spec}); only the candidates' id/server/name/description/input_schema reach the model.
    model: a loaded router, an artifact path, or None for $JERY_MODEL_PATH, else
    models/mcp-routing-v2/latest/model.json.gz.

    Raises ValueError for invalid inputs or a prediction that violates the contract (tool outside the
    candidates, action outside the schema enum, bad ranking).
    """
    features = contract_input({'request': request, 'context': context, 'candidate_tools': candidate_tools},
                              _as_catalog(catalog))
    if model is None or isinstance(model, (str, Path)):
        model = _cached_model(model or os.environ.get('JERY_MODEL_PATH') or DEFAULT_MODEL)
    internal = model.predict({k: features[k] for k in ('request', 'context', 'candidate_tools')},
                             features['catalog'])
    return validate_prediction(to_contract(internal), features['candidate_tools'], features['catalog'])
