"""Prediction adapter: predict(request, context, candidate_tools, catalog) -> label-shaped dict."""
import gzip
import json
import os
from pathlib import Path

from .features import DECISIONS, index_catalog
from .routers import LinearRouter, RetrievalRouter

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / 'models/mcp-routing-v2/latest/model.json.gz'
_cache = {}


def save_model(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.to_dict() if hasattr(model, 'to_dict') else {'config': model.config()}
    data['meta'] = meta or {}
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True)


def load_model(path, catalog):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    cfg = data['config']
    if cfg['kind'] == 'retrieval':
        return RetrievalRouter(catalog, threshold=cfg['threshold'], fallback=cfg['fallback'])
    return LinearRouter.from_dict(data, catalog)


def predict(request, context, candidate_tools, catalog, model=None):
    """Route one turn.

    request: str; context: list of {"role", "content"} turns before the tool choice;
    candidate_tools: list of tool ids (mcp__{server}__{name}); catalog: tools.json dict, list of
    specs or {id: spec}. model: a loaded router, a model path, or None for $JERY_MODEL_PATH /
    models/mcp-routing-v2/latest/model.json.gz.

    Returns {"decision": one of DECISIONS, "acceptable_choices": [{"tool", "action"?}] (empty
    unless select_tools; the tool is always in candidate_tools), "ranked_tools": [{"tool", "score"}]}.
    """
    if not isinstance(request, str):
        raise TypeError('request must be a string')
    catalog = index_catalog(catalog)
    if model is None or isinstance(model, (str, Path)):
        path = str(model or os.environ.get('JERY_MODEL_PATH') or DEFAULT_MODEL)
        if path not in _cache:
            _cache[path] = load_model(path, catalog)
        model = _cache[path]
    inp = {'request': request, 'context': list(context or []), 'candidate_tools': list(candidate_tools or [])}
    result = model.predict(inp, catalog)
    allowed = set(inp['candidate_tools'])
    if result['decision'] not in DECISIONS or any(c['tool'] not in allowed for c in result['acceptable_choices']):
        raise AssertionError('router returned a selection outside the candidates')
    return result
