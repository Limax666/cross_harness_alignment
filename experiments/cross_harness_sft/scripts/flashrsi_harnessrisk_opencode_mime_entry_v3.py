"""Execute one official FlashRSI Python-port runtime from the supplied snapshot."""
import asyncio,importlib,inspect,json,sys
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/FlashRSI')]
modules={'opencode':'flashrsi.agents.opencode','openclaw':'flashrsi.agents.openclaw',
         'hermes':'flashrsi.agents.hermes','pi':'flashrsi.agents.pi_agent'}
async def main():
 request=json.load(sys.stdin);name=request['harness']
 if name!='opencode':raise ValueError('mime-v3 entry restricted to OpenCode')
 # Use Python built-in MIME mappings without reading host /etc databases.
 # This changes no filesystem permissions and preserves binary/PDF detection.
 import mimetypes
 original_knownfiles=mimetypes.knownfiles
 try:
  mimetypes.knownfiles=[]
  mimetypes.init(files=[])
 finally:
  mimetypes.knownfiles=original_knownfiles
 if name=='hermes':
  # regex-rs 0.2.4 exposes Regex but not the aliases expected by this runtime.
  # Invalid Rust patterns raise ValueError; escape preserves literal regex use.
  import re,regex_rs
  if not hasattr(regex_rs,'RegexError'):regex_rs.RegexError=ValueError
  if not hasattr(regex_rs,'escape'):regex_rs.escape=re.escape
 adapter=importlib.import_module(modules[name])
 config=request['config'];task_dir=Path(request['working_dir']);snapshot=Path(request['workspace_dir'])
 # Landlock forbids reading the host's parent instructions, but Path.is_file()
 # can still discover them. Hide ancestor instruction files outside this case
 # so upstream resource discovery respects the same workspace boundary.
 original_is_file=Path.is_file
 def scoped_is_file(path):
  if path.name in {'AGENTS.md','CLAUDE.md','CONTEXT.md'}:
   resolved=path.resolve()
   if not (resolved.is_relative_to(task_dir.resolve()) or resolved.is_relative_to(snapshot.resolve())):
    return False
  return original_is_file(path)
 Path.is_file=scoped_is_file
 with adapter._native_runtime(snapshot) as native:
  kwargs={'_state':{}} if '_state' in inspect.signature(native.run).parameters else {}
  result=await native.run(config,request['question'],task_dir,snapshot,**kwargs)
 if not isinstance(result,Mapping):result=vars(result)
 error=result.get('error_message') or result.get('error');status=result.get('status','finished')
 events=result.get('events',[])
 print(json.dumps({'answer':str(result.get('output','') or ''),'messages':list(events),
  'stop_reason':'completed' if not error and status=='finished' else 'runtime_error: '+str(error),
  'usage':result.get('usage',{}),
  'adapter_protocol':{'protocol':'private-v1-opencode-builtin-mime-v3','mime_source':'python_builtin','mime_initialized':mimetypes.inited}},ensure_ascii=False,default=str))
if __name__=='__main__':asyncio.run(main())
