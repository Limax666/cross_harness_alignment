"""Match raw assistant API tool calls to an independently captured dispatch log.

This is an evidence check, not a tool authorization gate or an environment
sandbox. The caller must capture the API responses before middleware and keep
the dispatch log independently; a reconstructed transcript is insufficient.
"""

import copy
import json
import threading


class GuardedChatProxy:
    """Validate a non-streaming chat exchange before returning it to an agent.

    The upstream callable and this instance must be held by trusted code. This
    class does not provide HTTP transport, process isolation, or endpoint ACLs.
    """

    def __init__(self, gate, model, tools, upstream):
        if not isinstance(gate, ToolCallGate) or not isinstance(model, str) or not model:
            raise ValueError("gate and fixed model are required")
        if not isinstance(tools, list) or not callable(upstream):
            raise ValueError("fixed tool schemas and upstream callable are required")
        names = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("invalid fixed tool schema")
            name = tool["function"].get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("invalid fixed tool name")
            names.append(name)
        if len(names) != len(set(names)) or frozenset(names) != gate._allowed:
            raise ValueError("tool schemas differ from gate authorization")
        self._gate, self._model, self._upstream = gate, model, upstream
        self._tools = _canonical(tools)
        self._raw_requests = []
        self._raw_messages = []
        self._lock = threading.Lock()

    @property
    def raw_messages(self):
        with self._lock:
            return copy.deepcopy(self._raw_messages)

    @property
    def raw_requests(self):
        with self._lock:
            return copy.deepcopy(self._raw_requests)

    def forward(self, request):
        with self._lock:
            try:
                self._gate.require_ready()
                if not isinstance(request, dict) or request.get("model") != self._model:
                    raise ValueError("unexpected model")
                if request.get("stream", False) is not False or type(request.get("n", 1)) is not int or request.get("n", 1) != 1:
                    raise ValueError("streaming or multiple choices are unsupported")
                if not isinstance(request.get("messages"), list) or _canonical(request.get("tools")) != self._tools:
                    raise ValueError("messages or fixed tool schemas invalid")
                captured_request = copy.deepcopy(request)
                response = copy.deepcopy(self._upstream(copy.deepcopy(captured_request)))
                if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
                    raise ValueError("expected one model choice")
                choice = response["choices"][0]
                if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                    raise ValueError("invalid model choice")
                message = copy.deepcopy(choice["message"])
                has_calls = bool(message.get("tool_calls"))
                if (choice.get("finish_reason") == "tool_calls") != has_calls:
                    raise ValueError("inconsistent tool-call finish reason")
                self._gate.accept_response(message)
                self._raw_requests.append(captured_request)
                self._raw_messages.append(message)
                return copy.deepcopy(response)
            except BaseException:
                self._gate.abort()
                raise


class GuardedToolDispatcher:
    """Call a trusted native recorder only after a successful gate claim."""

    def __init__(self, gate, recorder):
        self._gate = gate
        self._recorder = recorder
        self._executed = []
        self._observations = {}
        self._lock = threading.Lock()

    @property
    def executed(self):
        with self._lock:
            return copy.deepcopy(self._executed)

    @property
    def observations(self):
        with self._lock:
            return copy.deepcopy(self._observations)

    def dispatch(self, name, arguments):
        with self._lock:
            try:
                args = copy.deepcopy(arguments)
                call_id = self._gate.claim(name, args)
                result = self._recorder.tool(name, copy.deepcopy(args))
                if not isinstance(result, dict) or not isinstance(result.get("observation"), str):
                    raise ValueError("native tool observation is missing")
                self._executed.append({"id": call_id, "name": name, "arguments": args})
                self._observations[call_id] = result["observation"]
                self._gate.complete(call_id)
                return result
            except BaseException:
                self._gate.abort()
                raise


class ToolCallGate:
    """One episode's pre-dispatch gate; the proxy must feed raw API responses.

    This object must be owned by the trusted proxy/dispatcher, inaccessible to
    the agent. It provides neither transport nor sandboxing by itself.
    """

    def __init__(self, allowed_tools):
        if not isinstance(allowed_tools, (set, frozenset)) or any(
            not isinstance(name, str) or not name for name in allowed_tools
        ):
            raise ValueError("allowed_tools must be a set of non-empty names")
        self._allowed = frozenset(allowed_tools)
        self._pending = []
        self._active = None
        self._seen_ids = set()
        self._failed = False
        self._closed = False
        self._saw_response = False
        self._lock = threading.Lock()

    def _require_open(self):
        if self._failed or self._closed:
            raise ValueError("tool gate is closed or failed")

    @property
    def finished(self):
        with self._lock:
            return self._closed and not self._failed

    def require_ready(self):
        with self._lock:
            self._require_open()
            if self._pending or self._active is not None:
                self._failed = True
                raise ValueError("tool dispatch is still pending")

    def accept_response(self, message):
        with self._lock:
            self._require_open()
            try:
                if self._pending or self._active is not None or not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("pending dispatches or invalid assistant response")
                calls = message.get("tool_calls")
                if calls is None:
                    calls = []
                if not isinstance(calls, list):
                    raise ValueError("invalid tool calls")
                parsed = []
                new_ids = set()
                for call in calls:
                    call_id, name, args = _parse_call(call)
                    if call_id in self._seen_ids or call_id in new_ids or name not in self._allowed:
                        raise ValueError("duplicate call ID or unauthorized tool")
                    new_ids.add(call_id)
                    parsed.append((call_id, name, args))
                self._pending = parsed
                self._seen_ids.update(new_ids)
                self._saw_response = True
            except (TypeError, ValueError, RecursionError) as exc:
                self._failed = True
                raise ValueError("invalid or unauthorized API response") from exc

    def claim(self, name, arguments):
        with self._lock:
            self._require_open()
            try:
                if not isinstance(arguments, dict) or not self._pending or self._active is not None:
                    raise ValueError("unexpected tool dispatch")
                call_id, expected_name, expected_args = self._pending[0]
                if name != expected_name or _canonical(arguments) != expected_args:
                    raise ValueError("dispatch differs from raw API response")
                self._pending.pop(0)
                self._active = call_id
                return call_id
            except (TypeError, ValueError, RecursionError) as exc:
                self._failed = True
                raise ValueError("tool dispatch rejected") from exc

    def complete(self, call_id):
        with self._lock:
            self._require_open()
            if self._active is None or self._active != call_id:
                self._failed = True
                raise ValueError("completion does not match an active call")
            self._active = None

    def finish(self):
        with self._lock:
            self._require_open()
            if self._pending or self._active is not None or not self._saw_response:
                self._failed = True
                raise ValueError("missing response or undispatched tool calls")
            self._closed = True

    def abort(self):
        with self._lock:
            self._failed = True


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _canonical(value):
    _check_json_value(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"))


def _check_json_value(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for child in value.values():
            _check_json_value(child)
    elif isinstance(value, list):
        for child in value:
            _check_json_value(child)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("not a JSON value")


def _parse_call(call):
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ValueError("invalid tool call")
    call_id, function = call.get("id"), call.get("function")
    if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
        raise ValueError("invalid call ID or function")
    name, raw = function.get("name"), function.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(raw, str):
        raise ValueError("invalid name or raw arguments")
    args = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    return call_id, name, _canonical(args)


def attest_tool_calls(messages, executed):
    if not isinstance(messages, list) or not isinstance(executed, list):
        return False
    observed = []
    seen_ids = set()
    saw_assistant = False
    try:
        for message in messages:
            if not isinstance(message, dict):
                return False
            saw_assistant |= message.get("role") == "assistant"
            calls = message.get("tool_calls")
            if calls is None:
                continue
            if message.get("role") != "assistant" or not isinstance(calls, list):
                return False
            for call in calls:
                call_id, name, args = _parse_call(call)
                if call_id in seen_ids:
                    return False
                seen_ids.add(call_id)
                observed.append((call_id, name, args))
        actual = []
        for item in executed:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                return False
            if not isinstance(item.get("name"), str) or not item["name"] or not isinstance(item.get("arguments"), dict):
                return False
            actual.append((item["id"], item["name"], _canonical(item["arguments"])))
        return saw_assistant and observed == actual
    except (TypeError, ValueError, RecursionError):
        return False


def attest_tool_observations(messages, requests, observations):
    """Check each raw response's tool results reached the next model request."""
    if not isinstance(messages, list) or not isinstance(requests, list) or not isinstance(observations, dict):
        return False
    if len(messages) != len(requests) or not messages:
        return False
    all_ids = set()
    try:
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                return False
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                return False
            if not calls:
                continue
            parsed = [_parse_call(call) for call in calls]
            ids = [call[0] for call in parsed]
            if index + 1 >= len(requests) or len(ids) != len(set(ids)) or any(call_id in all_ids for call_id in ids):
                return False
            all_ids.update(ids)
            next_request = requests[index + 1]
            context = next_request.get("messages") if isinstance(next_request, dict) else None
            if not isinstance(context, list) or any(not isinstance(item, dict) for item in context):
                return False
            matching = []
            for position, item in enumerate(context):
                if item.get("role") == "assistant" and isinstance(item.get("tool_calls"), list):
                    if [_parse_call(call) for call in item["tool_calls"]] == parsed:
                        matching.append(position)
            if len(matching) != 1:
                return False
            for call_id in ids:
                replies = [(position, item) for position, item in enumerate(context)
                           if item.get("role") == "tool" and item.get("tool_call_id") == call_id]
                if (len(replies) != 1 or replies[0][0] <= matching[0] or
                        call_id not in observations or replies[0][1].get("content") != observations[call_id]):
                    return False
        return all_ids == set(observations)
    except (TypeError, ValueError, KeyError, RecursionError):
        return False
