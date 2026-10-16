"""Shared boundary for routing inputs and model outputs; never extracts labels."""
import copy
import math

DECISIONS = frozenset(('select_tools', 'no_tool', 'clarify', 'unsupported'))


def model_input(example, catalog):
    """Copy only request, prior messages, candidate IDs and their tool specifications."""
    if not isinstance(example, dict) or not isinstance(catalog, dict):
        raise ValueError('Example and catalog must be objects')
    request = example.get('request')
    if not isinstance(request, str) or not request.strip():
        raise ValueError('A nonempty request is required')
    context = example.get('context')
    if not isinstance(context, list):
        raise ValueError('Context must be a message list')
    messages = []
    for message in context:
        if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant', 'tool') or not isinstance(message.get('content'), str):
            raise ValueError('Invalid context message')
        messages.append({'role': message['role'], 'content': message['content']})
    candidates = example.get('candidate_tools')
    if not isinstance(candidates, list) or any(not isinstance(t, str) for t in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError('Candidate IDs must be a unique string list')
    tools = catalog.get('tools')
    if not isinstance(tools, list):
        raise ValueError('Catalog must contain tools')
    by_id = {}
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get('id'), str) or tool['id'] in by_id:
            raise ValueError('Invalid or duplicate catalog ID')
        by_id[tool['id']] = tool
    specs = []
    for tool_id in candidates:
        if tool_id not in by_id:
            raise ValueError('Candidate missing from catalog')
        spec = by_id[tool_id]
        if not isinstance(spec.get('input_schema'), dict) or any(not isinstance(spec.get(k), str) for k in ('server', 'name', 'description')):
            raise ValueError('Candidate requires a schema and description')
        specs.append(copy.deepcopy({k: spec[k] for k in ('id', 'server', 'name', 'description', 'input_schema')}))
    return {'request': request, 'context': messages, 'candidate_tools': candidates[:], 'catalog': {'tools': specs}}


def validate_prediction(prediction, candidate_tools, catalog):
    """Reject invalid decisions, tools, actions, rankings and numeric confidence."""
    if not isinstance(prediction, dict) or not isinstance(prediction.get('decision'), str) or prediction['decision'] not in DECISIONS:
        raise ValueError('Invalid routing decision')
    choices = prediction.get('choices')
    if not isinstance(choices, list) or bool(choices) != (prediction['decision'] == 'select_tools'):
        raise ValueError('Choices must match the routing decision')
    tools = {t['id']: t for t in catalog['tools']}
    seen = set()
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) - {'tool', 'action'}:
            raise ValueError('Invalid choice fields')
        tool_id = choice.get('tool')
        if not isinstance(tool_id, str) or tool_id not in candidate_tools or tool_id not in tools:
            raise ValueError('Selected tool is outside candidates')
        action = choice.get('action')
        if 'action' in choice:
            actions = tools[tool_id]['input_schema'].get('properties', {}).get('action', {}).get('enum', [])
            if not isinstance(action, str) or action not in actions:
                raise ValueError('Invalid action')
        key = tool_id, action
        if key in seen:
            raise ValueError('Duplicate choice')
        seen.add(key)
    if 'ranked_tools' in prediction:
        ranked = prediction['ranked_tools']
        if not isinstance(ranked, list) or any(not isinstance(t, str) or t not in candidate_tools for t in ranked) or len(set(ranked)) != len(ranked):
            raise ValueError('Invalid ranking')
    if 'confidence' in prediction:
        confidence = prediction['confidence']
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be finite and between zero and one')
    return prediction
