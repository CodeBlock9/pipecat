#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""run_test leaves nothing running when either of its two sides fails.

run_test runs the pipeline's runner next to a pusher, which waits for the
pipeline to start and then queues the test's frames. It used to await the two
with asyncio.gather, which does not cancel one side when the other fails: a
pusher that missed its start raised TimeoutError out of run_test and left the
runner, and the pipeline worker under it, running.

Each test forces one failure deterministically and checks that run_test raises
the original error with nothing it started still running. A regression leaves
those tasks behind, so every test stops them itself, bounded, before its event
loop closes: a worker the loop's close cancels together with its own tasks
never finishes, and the test run would hang instead of failing.
"""

import asyncio
import time

import pytest

import pipecat.tests.utils as test_utils
from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import run_test
from pipecat.workers.runner import WorkerRunner


class _HoldsStart(FrameProcessor):
    """Holds the StartFrame for ``hold`` seconds, then passes everything on."""

    def __init__(self, hold: float):
        super().__init__()
        self._hold = hold

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await asyncio.sleep(self._hold)
        await self.push_frame(frame, direction)


class _EndsBeforeStart(FrameProcessor):
    """Cancels its pipeline on the StartFrame and drops it, so it never starts."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.pipeline_worker.cancel(reason="ends before it starts")
            return
        await self.push_frame(frame, direction)


class _Passes(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _FailingRunner(WorkerRunner):
    async def run(self, *args, **kwargs):
        raise RuntimeError("the runner failed")


def _record_runners(monkeypatch, runner_class=WorkerRunner) -> list[WorkerRunner]:
    """Make run_test build its runner from ``runner_class``, and record it."""
    runners = []

    class Recording(runner_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runners.append(self)

    monkeypatch.setattr(test_utils, "WorkerRunner", Recording)
    return runners


def _still_running(before: set[asyncio.Task]) -> list[asyncio.Task]:
    return [task for task in asyncio.all_tasks() - before if not task.done()]


async def _stop_leftovers(runners: list[WorkerRunner], before: set[asyncio.Task]):
    """Stop what a regressed run_test left running, bounded.

    The runner is cancelled the way run_test should have, so its worker winds
    down with its own tasks alive. Only the pusher is cancelled directly.
    """
    for runner in runners:
        await runner.cancel(reason="test cleanup")
    leftover = _still_running(before)
    for task in leftover:
        if task.get_coro().__qualname__.endswith("push_frames"):
            task.cancel()
    if leftover:
        await asyncio.wait(leftover, timeout=10)


@pytest.mark.asyncio
async def test_a_missed_start_raises_and_leaves_nothing_running(monkeypatch):
    runners = _record_runners(monkeypatch)
    before = asyncio.all_tasks()
    try:
        with pytest.raises(TimeoutError):
            await run_test(
                _HoldsStart(hold=0.5), frames_to_send=[TextFrame("hi")], start_timeout=0.05
            )
        assert _still_running(before) == []
    finally:
        await _stop_leftovers(runners, before)


@pytest.mark.asyncio
async def test_a_failed_runner_raises_its_error_and_leaves_nothing_running(monkeypatch):
    runners = _record_runners(monkeypatch, _FailingRunner)
    before = asyncio.all_tasks()
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="the runner failed"):
            await run_test(_Passes(), frames_to_send=[TextFrame("hi")], start_timeout=30)
        # The pusher is cancelled rather than left to sit out its start_timeout.
        assert time.monotonic() - started < 15
        assert _still_running(before) == []
    finally:
        await _stop_leftovers(runners, before)


@pytest.mark.asyncio
async def test_a_pipeline_that_ends_before_it_starts_raises_at_once(monkeypatch):
    runners = _record_runners(monkeypatch)
    before = asyncio.all_tasks()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await run_test(_EndsBeforeStart(), frames_to_send=[TextFrame("hi")], start_timeout=30)
        # The runner has ended, so the pusher's wait for a start is over: it
        # used to sit out the whole start_timeout before raising.
        assert time.monotonic() - started < 15
        assert _still_running(before) == []
    finally:
        await _stop_leftovers(runners, before)
