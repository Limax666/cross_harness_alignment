"""Bind official FlashRSI Codex/Claude adapters to a fixed AgentDojo MCP tool."""
import json
from pathlib import Path
from flashrsi.agents.codex import Codex
from flashrsi.agents.claude_code import ClaudeCode

ROOT=Path(__file__).resolve().parents[1]
MCP_SCRIPT=str(ROOT/'scripts/flashrsi_native_mcp_stdio.py')
POLICY=ROOT/'configs/flashrsi_b0_safety_policy.txt'


def make_native_snapshot(harness,config):
    agent={'codex':Codex,'claude_code':ClaudeCode}[harness](config)
    base=agent.snapshot
    with base.check_out() as directory:
        prompt=directory/('AGENTS.md' if harness=='codex' else 'CLAUDE.md')
        original=prompt.read_text() if prompt.exists() else ''
        prompt.write_text(original.rstrip()+'\n\n## Trusted safety policy\n'+POLICY.read_text().strip()+'\n')
        if harness=='codex':
            settings=directory/'config.toml';content=settings.read_text() if settings.exists() else ''
            if '[mcp_servers.benchmark]' in content:raise ValueError('unexpected existing benchmark MCP server')
            settings.write_text(content+'\n[mcp_servers.benchmark]\ncommand = "/usr/bin/python3"\n'
                +'args = '+json.dumps(['-I','-B',MCP_SCRIPT])+'\n'
                +'env_vars = '+json.dumps(['FLASHRSI_GATEWAY_URL','FLASHRSI_GATEWAY_TOKEN','FLASHRSI_NATIVE_SCHEMAS'])+'\n')
        else:
            config_path=directory/'.mcp.json'
            value=json.loads(config_path.read_text()) if config_path.exists() else {'mcpServers':{}}
            if value.get('mcpServers'):raise ValueError('unexpected existing MCP servers')
            value['mcpServers']={'benchmark':{'command':'/usr/bin/python3','args':['-I','-B',MCP_SCRIPT]}}
            config_path.write_text(json.dumps(value,indent=2)+'\n')
        return base.commit(directory)
