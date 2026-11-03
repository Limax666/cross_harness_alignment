"""Request deadline: explicitly abort the backend and never invent sampled tokens."""
import asyncio

async def generate_with_deadline(generation, abort, timeout):
    if timeout <= 0:
        raise ValueError('generation deadline must be positive')
    task=asyncio.ensure_future(generation)
    try:
        return await asyncio.wait_for(asyncio.shield(task),timeout)
    except asyncio.TimeoutError as exc:
        try:
            await asyncio.wait_for(abort(),10)
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        raise TimeoutError('backend generation deadline expired; request aborted, rollout unscorable') from exc
