#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The worker counts the heartbeats that got through.

`on_heartbeat_timeout` fires every `heartbeats_monitor_secs` for as long as a
stall lasts, and it fires the same way whether the pipeline has stopped or is
merely slow. The count is what separates the two: between two timeouts it
either moved, and frames are getting through, or it did not, and nothing is.
"""

import asyncio

import pytest

from pipecat.frames.frames import HeartbeatFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor


class _Passthrough(FrameProcessor):
    pass


def _worker() -> PipelineWorker:
    return PipelineWorker(Pipeline([_Passthrough()]))


def test_a_fresh_worker_has_received_none():
    assert _worker().heartbeats_received == 0


@pytest.mark.asyncio
async def test_each_heartbeat_the_monitor_consumes_advances_the_count():
    worker = _worker()
    monitor = asyncio.create_task(worker._heartbeat_monitor_handler())
    try:
        for _ in range(3):
            await worker._heartbeat_queue.put(HeartbeatFrame(timestamp=worker._clock.get_time()))
        for _ in range(200):
            if worker.heartbeats_received == 3:
                break
            await asyncio.sleep(0.01)
    finally:
        monitor.cancel()

    assert worker.heartbeats_received == 3


@pytest.mark.asyncio
async def test_the_count_is_unchanged_while_nothing_traverses_the_pipeline():
    """The stalled case, which is what a consumer reads it for."""
    worker = _worker()
    worker._params.heartbeats_monitor_secs = 0.02
    fired = []

    async def _timeout(_worker):
        fired.append(worker.heartbeats_received)

    worker.add_event_handler("on_heartbeat_timeout", _timeout)
    monitor = asyncio.create_task(worker._heartbeat_monitor_handler())
    try:
        for _ in range(200):
            if len(fired) >= 3:
                break
            await asyncio.sleep(0.01)
    finally:
        monitor.cancel()

    assert len(fired) >= 3
    assert len(set(fired)) == 1, f"the count moved while nothing got through: {fired}"
