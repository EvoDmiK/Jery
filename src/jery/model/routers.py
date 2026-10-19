"""Two small local routers: spec retrieval with a threshold, and a candidate-masked linear softmax."""
import math
import random

from .features import (
    NULL_DECISIONS, TfidfIndex, index_catalog, model_input, query_features, server_of, tokens,
    unique_candidates,
)


def _result(decision, ranked, action=None, confidence=None):
    """Internal prediction in label shape (acceptable_choices + scored ranking).

    jery.model.predict converts this to the public prediction-v1 contract at the boundary.
    """
    choices = []
    if decision == 'select_tools' and ranked:
        choice = {'tool': ranked[0]['tool']}
        if action:
            choice['action'] = action
        choices.append(choice)
    out = {'decision': decision, 'acceptable_choices': choices, 'ranked_tools': ranked}
    if confidence is not None:
        out['confidence'] = confidence
    return out


def schema_actions(spec):
    """Valid action enum from a candidate's input schema."""
    action = ((spec or {}).get('input_schema') or {}).get('properties', {}).get('action', {})
    return set(action.get('enum') or []) if isinstance(action, dict) else set()


def _restore_index(data):
    if 'index' not in data:
        raise ValueError('model artifact has no saved TF-IDF index state; '
                         'run python3 -m jery.model.augment_index with the original training catalog')
    return TfidfIndex(state=data['index'])


class RetrievalRouter:
    """Rank candidates by TF-IDF cosine to their spec; below `threshold` return `fallback`."""

    kind = 'retrieval'

    def __init__(self, catalog=None, threshold=0.0, fallback='unsupported', index=None):
        self.index = index or TfidfIndex(catalog)
        self.threshold = threshold
        self.fallback = fallback

    def rank(self, inp, catalog=None):
        catalog = index_catalog(catalog) if catalog is not None else self.index.catalog
        query = self.index.query_vector(inp['request'], inp['context'])
        ranked = [{'tool': t, 'score': round(self.index.cosine(query, self.index.tool_vector(t, catalog)), 6)}
                  for t in unique_candidates(inp['candidate_tools'])]
        ranked.sort(key=lambda r: -r['score'])
        return ranked

    def predict(self, inp, catalog=None):
        ranked = self.rank(inp, catalog)
        decision = 'select_tools' if ranked and ranked[0]['score'] >= self.threshold else self.fallback
        return _result(decision, ranked)

    def config(self):
        return {'kind': self.kind, 'threshold': self.threshold, 'fallback': self.fallback}

    def to_dict(self):
        return {'config': self.config(), 'index': self.index.state()}

    @classmethod
    def from_dict(cls, data):
        cfg = {k: v for k, v in data['config'].items() if k != 'kind'}
        return cls(index=_restore_index(data), **cfg)


class LinearRouter:
    """Softmax over {candidates} + {no_tool, clarify, unsupported}.

    Tool score = w_tool·φ(q) + w_server·φ(q) + v·pair(q, spec). Decision scores see φ(q) and
    candidate-set statistics, so `unsupported` wins when the needed tool is absent from candidates.
    """

    kind = 'linear'

    def __init__(self, catalog=None, epochs=10, lr=0.2, l2=0.0, use_spec=True, seed=13, index=None):
        self.index = index or TfidfIndex(catalog)
        self.epochs, self.lr, self.l2, self.use_spec, self.seed = epochs, lr, l2, use_spec, seed
        self.w = {}
        self.actions = {}

    def config(self):
        return {'kind': self.kind, 'epochs': self.epochs, 'lr': self.lr, 'l2': self.l2,
                'use_spec': self.use_spec, 'seed': self.seed}

    # ----- features -----
    def _options(self, inp, catalog):
        """[(label, [(class_key, extra_pairs)])] for every candidate plus the null decisions."""
        phi = query_features(inp['request'], inp['context'])
        candidates = unique_candidates(inp['candidate_tools'])
        sims = {}
        if self.use_spec:
            qvec = self.index.query_vector(inp['request'], inp['context'])
            qtok = set(t for t in tokens(' '.join([inp['request']] + [str(c.get('content', '')) for c in inp['context'] if isinstance(c, dict)])) if t.startswith('w:'))
            for t in candidates:
                spec = catalog.get(t) or {}
                name_tok = set(tokens((spec.get('name') or t.split('__')[-1]).replace('_', ' ')))
                name_tok = {x for x in name_tok if x.startswith('w:')}
                sims[t] = (self.index.cosine(qvec, self.index.tool_vector(t, catalog)),
                           len(name_tok & qtok) / max(len(name_tok), 1),
                           1.0 if 'w:' + server_of(t, spec).split('_')[0] in qtok else 0.0)
        options = []
        for t in candidates:
            pairs = [('pair:sim', sims[t][0]), ('pair:name', sims[t][1]), ('pair:server', sims[t][2])] if sims else []
            options.append((t, ['T:' + t, 'S:' + server_of(t, catalog.get(t))], pairs))
        top = max((s[0] for s in sims.values()), default=0.0)
        cand_stats = [('cand:empty', 1.0 if not candidates else 0.0), ('cand:log_n', math.log1p(len(candidates)) / 5)]
        if self.use_spec:
            cand_stats.append(('cand:max_sim', top))
        for d in NULL_DECISIONS:
            options.append((d, ['D:' + d], [(f + '|' + d, v) for f, v in cand_stats]))
        return phi, options

    def _score(self, phi, classes, pairs):
        w = self.w
        s = sum(w.get(f + '|' + c, 0.0) * v for c in classes for f, v in phi.items())
        return s + sum(w.get(f, 0.0) * v for f, v in pairs)

    @staticmethod
    def _softmax(scores):
        m = max(scores)
        e = [math.exp(s - m) for s in scores]
        z = sum(e)
        return [x / z for x in e]

    def _step(self, phi, classes_list, gold):
        """One SGD step on -log sum_{g in gold} p_g for parallel option lists."""
        scores = [self._score(phi, c, p) for c, p in classes_list]
        probs = self._softmax(scores)
        mass = sum(probs[i] for i in gold) or 1e-12
        w, lr, l2 = self.w, self.lr, self.l2
        for i, (classes, pairs) in enumerate(classes_list):
            g = probs[i] - (probs[i] / mass if i in gold else 0.0)
            if abs(g) < 1e-5:
                continue
            for c in classes:
                for f, v in phi.items():
                    k = f + '|' + c
                    old = w.get(k, 0.0)
                    w[k] = old - lr * (g * v + l2 * old)
            for f, v in pairs:
                old = w.get(f, 0.0)
                w[f] = old - lr * (g * v + l2 * old)

    def fit(self, rows):
        """Fit on training rows; only model_input() fields plus the training labels are read."""
        catalog = self.index.catalog
        prepared = []
        for row in rows:
            phi, options = self._options(model_input(row), catalog)
            labels = [o[0] for o in options]
            if row['decision'] == 'select_tools':
                good = {c['tool'] for c in row['acceptable_choices']}
            else:
                good = {row['decision']}
            gold = {i for i, label in enumerate(labels) if label in good}
            if not gold:
                continue
            acts = [(c['tool'], c['action']) for c in row['acceptable_choices'] if c.get('action')]
            for tool, action in acts:
                self.actions.setdefault(tool, set()).add(action)
            prepared.append((phi, [(o[1], o[2]) for o in options], gold, acts))
        self.actions = {t: sorted(a) for t, a in self.actions.items()}
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(prepared)
            for phi, options, gold, acts in prepared:
                self._step(phi, options, gold)
                for tool, action in acts:
                    names = self.actions[tool]
                    if len(names) > 1:
                        self._step(phi, [(['A:' + tool + ':' + a], []) for a in names], {names.index(action)})
        self.w = {k: v for k, v in self.w.items() if abs(v) > 1e-6}
        return self

    def predict(self, inp, catalog=None):
        catalog = index_catalog(catalog) if catalog is not None else self.index.catalog
        phi, options = self._options(inp, catalog)
        scores = [self._score(phi, o[1], o[2]) for o in options]
        probs = self._softmax(scores)
        ranked = sorted(({'tool': o[0], 'score': round(p, 6)} for o, p in zip(options, probs)
                         if o[0] not in NULL_DECISIONS), key=lambda r: -r['score'])
        best = max(range(len(options)), key=lambda i: scores[i])
        label = options[best][0]
        if label in NULL_DECISIONS:
            return _result(label, ranked, confidence=round(probs[best], 6))
        valid = schema_actions(catalog.get(label))
        names = [a for a in self.actions.get(label, []) if a in valid]
        action = None
        if len(names) == 1:
            action = names[0]
        elif names:
            action = max(names, key=lambda a: self._score(phi, ['A:' + label + ':' + a], []))
        return _result('select_tools', ranked, action, confidence=round(probs[best], 6))

    # ----- persistence -----
    def to_dict(self):
        return {'config': self.config(), 'weights': self.w, 'actions': self.actions, 'index': self.index.state()}

    @classmethod
    def from_dict(cls, data):
        cfg = {k: v for k, v in data['config'].items() if k != 'kind'}
        model = cls(index=_restore_index(data), **cfg)
        model.w = data['weights']
        model.actions = data['actions']
        return model
