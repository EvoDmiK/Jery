"""Model inputs and text features. Only request, context and candidate tool specs are read."""
from collections import Counter
import math
import re
import unicodedata

MODEL_INPUT_KEYS = ('request', 'context', 'candidate_tools')
DECISIONS = ('select_tools', 'no_tool', 'clarify', 'unsupported')
NULL_DECISIONS = DECISIONS[1:]
WORD = re.compile(r'\w+')


def model_input(row):
    """Return the only row fields a model may see; labels, ids, split and provenance are dropped."""
    return {
        'request': row['request'],
        'context': list(row.get('context') or []),
        'candidate_tools': list(row.get('candidate_tools') or []),
    }


def index_catalog(catalog):
    """Accept tools.json, a list of tool specs or an id -> spec mapping."""
    if isinstance(catalog, dict) and 'tools' in catalog:
        catalog = catalog['tools']
    if isinstance(catalog, dict):
        return {tool_id: dict(spec, id=tool_id) for tool_id, spec in catalog.items()}
    return {spec['id']: spec for spec in catalog or []}


def unique_candidates(candidate_tools):
    return list(dict.fromkeys(candidate_tools or []))


def server_of(tool_id, spec=None):
    if spec and spec.get('server'):
        return spec['server']
    parts = tool_id.split('__')
    return parts[1] if len(parts) >= 3 else ''


def tokens(text):
    """Word unigrams plus character 2/3-grams inside words; robust for Korean without a tokenizer."""
    text = unicodedata.normalize('NFKC', text or '').lower()
    out = []
    for word in WORD.findall(text):
        out.append('w:' + word)
        padded = '<' + word + '>'
        for n in (2, 3):
            out.extend('c:' + padded[i:i + n] for i in range(len(padded) - n + 1))
    return out


def query_text(request, context):
    return [request] + [str(turn.get('content', '')) for turn in context or [] if isinstance(turn, dict)]


def query_features(request, context):
    """Sparse, L2-normalised log-tf features; context tokens are kept apart from request tokens."""
    counts = Counter('r' + t for t in tokens(request))
    for turn in context or []:
        if isinstance(turn, dict):
            counts.update('x' + t for t in tokens(str(turn.get('content', ''))))
    feats = {f: 1.0 + math.log(c) for f, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    feats = {f: v / norm for f, v in feats.items()}
    feats['bias'] = 1.0
    feats['has_context'] = 1.0 if context else 0.0
    return feats


def tool_text(tool_id, spec):
    """Text from the candidate's public spec only: server, name, description and input schema."""
    spec = spec or {}
    parts = [server_of(tool_id, spec), spec.get('name') or tool_id.split('__')[-1]]
    parts.append((spec.get('description') or '')[:1500])
    for name, prop in ((spec.get('input_schema') or {}).get('properties') or {}).items():
        parts.append(name)
        if isinstance(prop, dict):
            parts.append(str(prop.get('description', ''))[:200])
            parts.extend(str(v) for v in prop.get('enum', []) if isinstance(v, str))
    return ' '.join(p.replace('_', ' ') for p in parts if p)


class TfidfIndex:
    """Character n-gram TF-IDF over tool specs. IDF uses catalog text only, never labels."""

    def __init__(self, catalog):
        self.catalog = index_catalog(catalog)
        docs = {tool_id: Counter(tokens(tool_text(tool_id, spec))) for tool_id, spec in self.catalog.items()}
        df = Counter(t for counts in docs.values() for t in counts)
        n = max(len(docs), 1)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.default_idf = math.log(1 + n) + 1.0
        self.vectors = {tool_id: self._vector(counts) for tool_id, counts in docs.items()}

    def _vector(self, counts):
        vec = {t: (1.0 + math.log(c)) * self.idf.get(t, self.default_idf) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def query_vector(self, request, context):
        return self._vector(Counter(t for text in query_text(request, context) for t in tokens(text)))

    def tool_vector(self, tool_id, catalog=None):
        if tool_id in self.vectors:
            return self.vectors[tool_id]
        spec = (catalog or {}).get(tool_id)
        return self._vector(Counter(tokens(tool_text(tool_id, spec))))

    @staticmethod
    def cosine(a, b):
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())
