#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the cancel-timeout warning naming where the CancelFrame stopped."""

import io

import pytest
from loguru import logger

from pipecat.frames.frames import CancelFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams, _leaf_processors
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager


class _CancelBlocker(FrameProcessor):
    """Swallows the ``CancelFrame`` so it never reaches the end of the pipeline."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, CancelFrame):
            return
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


@pytest.mark.asyncio
async def test_cancel_timeout_names_the_processors_that_never_saw_the_frame():
    blocker = _CancelBlocker(name="CancelBlocker")
    downstream = IdentityFilter(name="Downstream")
    worker = PipelineWorker(
        Pipeline([blocker, downstream]),
        idle_timeout_secs=0.2,
        cancel_timeout_secs=0.3,
    )

    log_output = io.StringIO()
    handler_id = logger.add(log_output, level="WARNING", format="{message}")
    try:
        await worker.run(WorkerParams(task_manager=TaskManager()))
    finally:
        logger.remove(handler_id)

    log_text = log_output.getvalue()
    assert "timeout waiting for" in log_text
    assert "CancelBlocker" in log_text
    assert "Downstream" in log_text


@pytest.mark.asyncio
async def test_the_report_names_only_the_processors_still_waiting():
    first = IdentityFilter(name="First")
    second = IdentityFilter(name="Second")
    worker = PipelineWorker(Pipeline([first, second]))

    first._cancelling = True

    report = worker._cancel_progress_report()

    assert "First" not in report
    assert "Second" in report


@pytest.mark.asyncio
async def test_the_report_says_so_when_every_processor_saw_the_frame():
    worker = PipelineWorker(Pipeline([IdentityFilter(name="Only")]))

    for processor in _leaf_processors(worker._pipeline):
        processor._cancelling = True

    assert worker._cancel_progress_report() == "Every processor has seen it."


@pytest.mark.asyncio
async def test_the_report_summarizes_a_long_tail():
    processors = [IdentityFilter(name=f"P{i}") for i in range(12)]
    worker = PipelineWorker(Pipeline(processors))

    report = worker._cancel_progress_report()

    assert "P0" in report
    assert "more" in report
