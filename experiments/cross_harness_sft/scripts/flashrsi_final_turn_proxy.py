"""Permit an explicitly tool-free final turn while preserving fail-closed gates."""
import copy
from flashrsi_tool_attestation import GuardedChatProxy


class FinalTurnChatProxy(GuardedChatProxy):
    def forward(self, request):
        if not isinstance(request, dict):
            return super().forward(request)
        final = request.get('tools') in (None, []) and request.get('tool_choice') == 'none'
        if not final:
            return super().forward(request)
        with self._lock:
            try:
                self._gate.require_ready()
                if request.get('model') != self._model or not isinstance(request.get('messages'), list):
                    raise ValueError('unexpected model or messages')
                if request.get('stream', False) is not False or type(request.get('n', 1)) is not int or request.get('n', 1) != 1:
                    raise ValueError('streaming or multiple choices unsupported')
                captured = copy.deepcopy(request)
                response = copy.deepcopy(self._upstream(copy.deepcopy(captured)))
                if not isinstance(response, dict) or not isinstance(response.get('choices'), list) or len(response['choices']) != 1:
                    raise ValueError('expected one choice')
                choice = response['choices'][0]
                if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                    raise ValueError('invalid choice')
                message = copy.deepcopy(choice['message'])
                if message.get('tool_calls') or choice.get('finish_reason') == 'tool_calls':
                    raise ValueError('tool calls are forbidden in the final turn')
                self._gate.accept_response(message)
                self._raw_requests.append(captured)
                self._raw_messages.append(message)
                return response
            except BaseException:
                self._gate.abort()
                raise
