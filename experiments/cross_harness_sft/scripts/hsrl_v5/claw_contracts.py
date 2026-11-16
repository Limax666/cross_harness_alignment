"""Data-only executable/script requirements for the pinned Python tool image."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import shlex

# Declared Python tool image: configs/hsrl_tools_v4.Dockerfile, Python slim
# base + pinned pip packages. This is a static declaration, not a runtime probe.
DECLARED_PROGRAMS = frozenset({'python', 'python3', 'sys.executable', 'bash', 'sh', 'gzip', 'pytest'})
ABSOLUTE_PROGRAMS = {'/usr/local/bin/python', '/usr/local/bin/python3', '/bin/bash', '/usr/bin/bash',
                     '/bin/sh', '/usr/bin/sh', '/bin/gzip', '/usr/bin/gzip'}
EXTERNAL_PROGRAMS = ('node', 'npm', 'npx', 'java', 'javac', 'gcc', 'g++', 'docker', 'ffmpeg',
                     'pdftotext', 'git', 'Rscript', 'octave', 'matlab', 'rclone', 'kubectl',
                     'terraform', 'speedtest', 'lspci', 'lsusb', 'sysctl', 'openssl', 'curl', 'wget')


def checker_execution_contract(checker: Path, query: str) -> dict:
    source = checker.read_text()
    calls, dynamic, unavailable = [], [], set()
    parse_error = None
    if any(token in source for token in ('subprocess', 'os.system', 'os.popen')):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            tree = None
            parse_error = {'type': type(exc).__name__, 'line': exc.lineno, 'message': str(exc)}
        if tree is not None:
            aliases = {}
            direct = {}
            functions = {'run', 'check_output', 'check_call', 'call', 'Popen', 'getoutput', 'getstatusoutput'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for item in node.names:
                        aliases[item.asname or item.name] = item.name
                elif isinstance(node, ast.ImportFrom) and node.module == 'subprocess':
                    direct.update({item.asname or item.name: 'subprocess.' + item.name for item in node.names if item.name in functions})
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call = ast.unparse(node.func)
                call = direct.get(call, call)
                for alias, module in aliases.items():
                    if call.startswith(alias + '.'):
                        call = module + call[len(alias):]
                        break
                if call not in {'subprocess.' + name for name in functions} | {'os.system', 'os.popen'}:
                    continue
                arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == 'args'), None)
                if arg is None:
                    continue
                shell = call in {'os.system', 'os.popen', 'subprocess.getoutput', 'subprocess.getstatusoutput'} or any(k.arg == 'shell' and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords)
                expression = ast.unparse(arg)
                program = None
                kind = 'dynamic_command_pending'
                if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
                    first = arg.elts[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        program, kind = first.value, 'literal_argv_program'
                    elif ast.unparse(first) == 'sys.executable':
                        program, kind = 'sys.executable', 'python_executable'
                    else:
                        kind = 'dynamic_program_pending'
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    try:
                        program = shlex.split(arg.value)[0]
                    except (ValueError, IndexError):
                        program = arg.value
                    kind = 'shell_command_pending' if shell else 'string_command_pending'
                evidence = {'line': node.lineno, 'call': call, 'program_or_expression': program or expression,
                            'argument_expression': expression, 'kind': kind, 'shell': shell}
                calls.append(evidence)
                if kind.endswith('_pending') or shell:
                    dynamic.append(evidence)
                if program is not None and program not in DECLARED_PROGRAMS and program not in ABSOLUTE_PROGRAMS:
                    unavailable.add(program)
    requests = []
    for clause in re.split(r'\n|(?<=[.!?;])\s+', query):
        if not re.search(r'\b(?:run|execute|invoke|launch|use|using)\b', clause, re.I):
            continue
        if re.search(r'\b(?:do not|never|without|no need to)\b.{0,40}\b(?:run|execute|invoke|launch|use)\b', clause, re.I):
            continue
        binary_names = [name for name in EXTERNAL_PROGRAMS if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', clause)]
        script_request = bool(re.search(r'\.py\b|\.sh\b|\.js\b|\bscript|command.line|CLI\b', clause, re.I))
        if binary_names or script_request:
            requests.append({'source': 'user_query', 'text': clause, 'requested_programs': binary_names,
                             'script_request': script_request})
            unavailable.update(binary_names)
    pending = bool(unavailable or dynamic or parse_error)
    return {'version': 'clawgym_checker_executables_and_requested_scripts_v1',
            'image_contract': 'hsrl_tools_v4_python_slim_plus_pinned_pip_Dockerfile',
            'declared_programs': sorted(DECLARED_PROGRAMS),
            'unavailable_or_unverified_programs': sorted(unavailable),
            'process_calls': calls, 'dynamic_command_evidence': dynamic,
            'requested_script_and_binary_evidence': requests, 'parse_error': parse_error,
            'checker_type_evidence': {'interface': 'original_hook_JSON_numeric_scalar_or_dict_runtime_shape_unverified',
                'json_serialization_lines': [n for n, text in enumerate(source.splitlines(), 1) if 'json.dumps(' in text or 'json.dump(' in text],
                'workspace_argument_lines': [n for n, text in enumerate(source.splitlines(), 1) if 'sys.argv' in text]},
            'status': 'pending' if pending else 'declared_programs_supported_runtime_script_validation_pending',
            'source_code_executed': False, 'failure_semantics': 'missing_or_unknown_binary_is_invalid_pending_not_utility_zero'}


def apply_execution_contract(record: dict) -> None:
    import json
    path = Path(record['source_path'])
    checker = Path(record['private_checker_path'])
    if not checker.is_file():
        return
    contract = checker_execution_contract(checker, json.loads(path.read_text())['user_query'])
    record['source_contract']['checker_execution_contract'] = contract
    record['capability_contract']['checker_execution_contract'] = contract
    if contract['status'] == 'pending':
        record['exclusion_reasons'] = sorted({*record['exclusion_reasons'], 'checker_executable_or_script_contract_pending'})
        record['qualified'] = record['structurally_admissible'] = False
        record['capability_contract']['status'] = 'pending_or_unsupported'
