#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A cancellation can say how long it is worth waiting for the frame to land.

`_wait_for_pipeline_end` waits for the `CancelFrame` to reach the end of the
pipeline before firing `on_pipeline_finished`, bounded by the worker's
`cancel_timeout_secs`. That is the right default: processors get to flush.

It is the wrong bound for a caller that already knows the pipeline cannot
forward frames -- an output transport that has given up on a dead socket, say.
The frame will not arrive, so the full timeout is dead time in which the worker
still holds every resource the call owns.
"""

import asyncio

import pytest

from pipecat.frames.frames import CancelFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import CANCEL_TIMEOUT_SECS, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor


class _Stalled(FrameProcessor):
    """A processor that never forwards, like an output transport that gave up."""


def _worker() -> PipelineWorker:
    return PipelineWorker(Pipeline([_Stalled()]))


def test_the_default_is_the_workers_own_timeout():
    worker = _worker()
    assert worker._pending_cancel_timeout_secs is None
    assert worker._cancel_timeout_secs == CANCEL_TIMEOUT_SECS


@pytest.mark.asyncio
async def test_a_bounded_cancel_does_not_wait_the_full_timeout():
    """The give-up case: the frame cannot arrive, so waiting buys nothing."""
    worker = _worker()
    finished = []

    @worker.event_handler("on_pipeline_finished")
    async def _finished(_worker, frame):
        finished.append(frame)

    worker._pending_cancel_timeout_secs = 0.05
    loop = asyncio.get_running_loop()
    started = loop.time()
    await worker._wait_for_pipeline_end(CancelFrame(reason="audio_output_write_failed"))
    elapsed = loop.time() - started

    assert elapsed < 1.0, (
        f"waited {elapsed:.1f}s for a frame that cannot arrive; the default is "
        f"{CANCEL_TIMEOUT_SECS}s"
    )


@pytest.mark.asyncio
async def test_an_unbounded_cancel_still_waits_for_the_frame():
    """Every other ending keeps the flush it depends on."""
    worker = _worker()
    worker._cancel_timeout_secs = 0.3

    loop = asyncio.get_running_loop()
    started = loop.time()
    await worker._wait_for_pipeline_end(CancelFrame(reason="end_call"))
    elapsed = loop.time() - started

    assert elapsed >= 0.3


@pytest.mark.asyncio
async def test_cancel_records_the_bound_for_this_cancellation():
    worker = _worker()
    worker._cancelled = True  # skip the queueing; the bound is what is under test

    await worker.cancel(reason="audio_output_write_failed", cancel_timeout_secs=0.05)

    assert worker._pending_cancel_timeout_secs == 0.05
