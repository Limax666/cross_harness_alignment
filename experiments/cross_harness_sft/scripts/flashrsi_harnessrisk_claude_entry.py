"""Execute Claude Code with the supplied FlashRSI snapshot in the sandbox."""
import asyncio,hashlib,json,sys,os,shutil,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'vendor/FlashRSI')]
from flashrsi.agents.claude_code import ClaudeCode
from flashrsi.agents.codex import Codex
from flashrsi.core import Snapshot
async def main():
 request=json.load(sys.stdin);snapshot=Snapshot.load(request['snapshot_json']);config=request['config']
 if request.get('harness')=='codex':
  # Codex copies its versioned core into this writable per-case scratch area.
  # Create executable files with their final modes; seccomp continues to deny chmod.
  scratch=Path(os.environ['HOME']).resolve();os.umask(0o022)
  original_copytree=shutil.copytree;original_chmod=Path.chmod;original_copystat=shutil.copystat
  def under_scratch(path):return scratch in Path(path).resolve().parents
  def safe_copy(src,dst,*,follow_symlinks=True):
   if not under_scratch(dst) or not follow_symlinks:raise PermissionError('copy outside per-case scratch')
   mode=stat.S_IMODE(os.stat(src,follow_symlinks=False).st_mode)
   with open(src,'rb') as source, os.fdopen(os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode),'wb') as dest:
    shutil.copyfileobj(source,dest)
   return dst
  def safe_copytree(src,dst,symlinks=False,ignore=None,copy_function=shutil.copy2,
                    ignore_dangling_symlinks=False,dirs_exist_ok=False):
   if not under_scratch(dst):raise PermissionError('copytree outside per-case scratch')
   result=original_copytree(src,dst,symlinks=symlinks,ignore=ignore,copy_function=safe_copy,
                            ignore_dangling_symlinks=ignore_dangling_symlinks,dirs_exist_ok=dirs_exist_ok)
   home=Path(dst)
   bundled=home/'python_core/resource_assets/system_skills'
   if home.name=='codex' and bundled.is_dir():
    target=home/'skills/.system'
    if target.exists():raise FileExistsError('unexpected pre-existing system skills')
    fingerprint=hashlib.sha256()
    for file in sorted(bundled.rglob('*')):
     if file.is_file():
      fingerprint.update(file.relative_to(bundled).as_posix().encode());fingerprint.update(file.read_bytes())
    target.parent.mkdir(parents=True,exist_ok=True)
    safe_copytree(bundled,target)
    (target/'.codex-python-system-skills.marker').write_text(fingerprint.hexdigest())
   return result
  def safe_copystat(src,dst,*args,**kwargs):
   if not under_scratch(dst):return original_copystat(src,dst,*args,**kwargs)
   # Scratch is privately owned and filesystem-restricted. copytree creates
   # directories using the process umask; no chmod is permitted by seccomp.
   if Path(dst).is_dir():return
   if stat.S_IMODE(os.stat(src).st_mode)!=stat.S_IMODE(os.stat(dst).st_mode):raise PermissionError('copy mode mismatch')
  def safe_chmod(path,mode,*args,**kwargs):
   if not under_scratch(path):return original_chmod(path,mode,*args,**kwargs)
   if stat.S_IMODE(path.stat().st_mode)!=mode:raise PermissionError('scratch mode change forbidden')
  shutil.copytree=safe_copytree;shutil.copystat=safe_copystat;Path.chmod=safe_chmod
  agent=Codex(config)
 else:agent=ClaudeCode(config)
 result=await agent._execute(request['question'],Path(request['working_dir']),snapshot,Path(request['workspace_dir']))
 print(json.dumps({'answer':result.output or '', 'messages':list(result.messages),
   'stop_reason':'completed' if result.status=='finished' else 'runtime_error: '+str(result.error_message),
   'usage':dict(result.usage)},ensure_ascii=False,default=str))
if __name__=='__main__':asyncio.run(main())
