#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenarios for test_worker_loop_close.py, each run as a whole process.

usage: python worker_loop_close_scenarios.py <scenario>

Each scenario runs ``asyncio.run(main())`` and leaves a pipeline worker in one
state when main returns. ``asyncio.run`` then cancels every task still alive,
the worker's own tasks among them, all at once, and waits for all of them
without a bound. Once the loop has closed, the process reports the worker's
state and exits 0. If the close is still waiting ``GRACE_SECS`` after main
returned, a watchdog prints every pending task's await chain and ends the
process with exit code 3.

The scenarios wait on the worker's state, never on a fixed sleep:

- prestart: the StartFrame is held mid-pipeline, so the worker has not started.
- idle: the orphan NEW-P-20 left behind. The pipeline is built the way
  run_test builds it, it has started and idles, and nothing awaits its runner.
- detached: Mesa's WebRTC shape. A started pipeline runs in a task of its own,
  which a server's shutdown does not wait for.
- ending: a graceful end in flight. The EndFrame is held mid-pipeline, as a
  farewell still playing holds it.
- orderly (control): as idle, but main stops the runner with runner.cancel()
  and waits for it before returning.
- cancelling (control): Mesa's telephony shape. The request task running the
  pipeline was cancelled and not awaited, and the CancelFrame is held
  mid-pipeline, as a service still closing its connection holds it.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

from loguru import logger

import pipecat
from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.workers.runner import WorkerRunner

GRACE_SECS = 10.0
# Bounds only a scenario that never reaches its state. Pipeline setup takes
# about 3 s on an idle machine, and several times that on a loaded one.
STATE_TIMEOUT_SECS = 30.0

logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    filter=lambda record: record["name"] in ("pipecat.pipeline.worker", "pipecat.workers.runner"),
)

_workers: list[PipelineWorker] = []
# Tasks a scenario orphans, kept here so that only the loop's close ends them.
_orphans: list[asyncio.Task] = []
_loop: asyncio.AbstractEventLoop | None = None
_main_returned = threading.Event()
_loop_closed = threading.Event()


class _Passes(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _Holds(_Passes):
    """Holds the first frame of one type for good, and passes the rest on."""

    def __init__(self, frame_type: type):
        super().__init__()
        self._frame_type = frame_type
        self.holding = asyncio.Event()

    async def process_frame(self, frame, direction):
        if isinstance(frame, self._frame_type) and not self.holding.is_set():
            await FrameProcessor.process_frame(self, frame, direction)
            self.holding.set()
            await asyncio.Event().wait()
            return
        await super().process_frame(frame, direction)


def _await_chain(coro) -> str:
    links = []
    while coro is not None:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            links.append(type(coro).__name__)
            break
        links.append(
            f"{frame.f_code.co_name}({Path(frame.f_code.co_filename).name}:{frame.f_lineno})"
        )
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return " > ".join(links)


def _report(prefix: str) -> None:
    for worker in _workers:
        print(
            f"{prefix} {worker}: started={worker.started_at is not None} "
            f"has_finished={worker.has_finished()} finished_event={worker._finished_event.is_set()}",
            flush=True,
        )


def _dump() -> None:
    print(
        f"HUNG: the loop's close is still waiting {GRACE_SECS:.0f}s after main returned", flush=True
    )
    for task in asyncio.all_tasks(_loop):
        if not task.done():
            print(f"  pending {task.get_name()}: {_await_chain(task.get_coro())}", flush=True)
    _report("  state")


def _watchdog() -> None:
    _main_returned.wait()
    if _loop_closed.wait(GRACE_SECS):
        return
    if _loop is not None:
        _loop.call_soon_threadsafe(_dump)
    time.sleep(2)
    os._exit(3)


async def _until(predicate, what: str) -> None:
    deadline = time.monotonic() + STATE_TIMEOUT_SECS
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError(f"timed out waiting until {what}")
        await asyncio.sleep(0.01)


def _worker(*processors: FrameProcessor, **kwargs) -> PipelineWorker:
    worker = PipelineWorker(Pipeline(list(processors)), **kwargs)
    _workers.append(worker)
    return worker


async def _run_pipeline_worker(worker: PipelineWorker) -> None:
    """Run a worker the way Mesa's run_pipeline_worker() does."""
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    await runner.run(auto_end=True)


async def _orphan_like_run_test() -> tuple[PipelineWorker, WorkerRunner, asyncio.Task]:
    """Start the pipeline run_test builds, under a runner nothing awaits."""
    source = QueuedFrameProcessor(queue=asyncio.Queue(), queue_direction=FrameDirection.UPSTREAM)
    sink = QueuedFrameProcessor(queue=asyncio.Queue(), queue_direction=FrameDirection.DOWNSTREAM)
    worker = _worker(source, _Passes(), sink, cancel_on_idle_timeout=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    _orphans.append(task)
    await _until(lambda: worker.started_at is not None, "the pipeline started")
    return worker, runner, task


async def prestart() -> None:
    holder = _Holds(StartFrame)
    worker = _worker(holder)
    _orphans.append(asyncio.create_task(_run_pipeline_worker(worker)))
    await _until(holder.holding.is_set, "the StartFrame is held")


async def idle() -> None:
    await _orphan_like_run_test()


async def detached() -> None:
    worker = _worker(_Passes())
    _orphans.append(asyncio.create_task(_run_pipeline_worker(worker), name="pipeline-task"))
    await _until(lambda: worker.started_at is not None, "the pipeline started")


async def ending() -> None:
    holder = _Holds(EndFrame)
    worker = _worker(holder)
    _orphans.append(asyncio.create_task(_run_pipeline_worker(worker)))
    await _until(lambda: worker.started_at is not None, "the pipeline started")
    await worker.queue_frame(EndFrame())
    await _until(holder.holding.is_set, "the EndFrame is held")


async def orderly() -> None:
    _, runner, task = await _orphan_like_run_test()
    await runner.cancel(reason="orderly stop")
    await asyncio.wait({task}, timeout=STATE_TIMEOUT_SECS)
    if not task.done():
        raise RuntimeError("runner.cancel() did not stop the runner")


async def cancelling() -> None:
    holder = _Holds(CancelFrame)
    worker = _worker(holder)
    request = asyncio.create_task(_run_pipeline_worker(worker), name="request-task")
    _orphans.append(request)
    await _until(lambda: worker.started_at is not None, "the pipeline started")
    request.cancel()
    await _until(holder.holding.is_set, "the CancelFrame is held")


SCENARIOS = {fn.__name__: fn for fn in (prestart, idle, detached, ending, orderly, cancelling)}


async def _main(scenario: str) -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    try:
        await SCENARIOS[scenario]()
        _report("RETURNING")
    finally:
        _main_returned.set()


if __name__ == "__main__":
    print(f"PIPECAT {Path(pipecat.__file__).resolve()}", flush=True)
    threading.Thread(target=_watchdog, daemon=True).start()
    asyncio.run(_main(sys.argv[1]))
    _loop_closed.set()
    _report("CLOSED")
