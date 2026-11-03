import asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rl_generation_deadline import generate_with_deadline

def test_success_does_not_abort():
 async def go():
  called=[]
  async def gen():return 'ok'
  async def abort():called.append(1)
  assert await generate_with_deadline(gen(),abort,0.1)=='ok'
  assert not called
 asyncio.run(go())

def test_timeout_aborts_and_raises_without_fabricating_output():
 async def go():
  called=[]
  async def gen():await asyncio.sleep(10)
  async def abort():called.append(1)
  try:await generate_with_deadline(gen(),abort,0.01)
  except TimeoutError:pass
  else:raise AssertionError('expected deadline failure')
  assert called==[1]
 asyncio.run(go())

def test_unconfirmed_abort_fails_closed():
 async def go():
  async def gen():await asyncio.sleep(10)
  async def abort():raise RuntimeError('abort unconfirmed')
  try:await generate_with_deadline(gen(),abort,0.01)
  except RuntimeError as e:assert str(e)=='abort unconfirmed'
  else:raise AssertionError('abort failure was hidden')
 asyncio.run(go())
