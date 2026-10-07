"""Qwen tool-call rendering for the registered benchmark MCP contracts."""

BENCHMARK_MCP_CHAT_TEMPLATE = r"""
{%- if tools %}
{{- '<|im_start|>system\nAvailable tools (call only these functions):\n' -}}
{{- tools | tojson -}}
{{- '\nThe exact callable names are those in the schemas above. A bare native name is invalid. Use <tool_call><function=EXACT_TOOL_NAME><parameter=ARG_NAME>JSON_VALUE</parameter></function></tool_call>, including every required parameter. Do not invent names or arguments. After each tool result, continue the task and provide a final answer.<|im_end|>\n' -}}
{%- endif %}
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' -}}
{{- message.get('content', '') or '' -}}
{%- for call in message.get('tool_calls', []) %}
{{- '<tool_call><function=' + call['function']['name'] + '>' -}}
{%- for key, value in call['function']['arguments'].items() %}
{{- '<parameter=' + key + '>' -}}{{- value | tojson -}}{{- '</parameter>' -}}
{%- endfor %}
{{- '</function></tool_call>' -}}
{%- endfor %}
{{- '<|im_end|>\n' -}}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' -}}{%- endif %}
""".strip()
