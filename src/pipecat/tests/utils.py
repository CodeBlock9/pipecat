#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Testing utilities for Pipecat pipeline components."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    HeartbeatFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import CANCEL_TIMEOUT_SECS, PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

# How long run_test waits for its pipeline to stop once either side has failed:
# the worker bounds its cancellation by CANCEL_TIMEOUT_SECS, and the rest covers
# the cleanup after it.
_STOP_TIMEOUT_SECS = CANCEL_TIMEOUT_SECS + 5.0


@dataclass
class SleepFrame(SystemFrame):
    """A system frame that introduces a sleep delay in the test pipeline.

    This frame is used by the test framework to control timing between
    frame processing, allowing tests to separate system frames from
    data or control frames.

    Parameters:
        sleep: Duration to sleep in seconds before processing the next frame.
    """

    sleep: float = 0.2


class HeartbeatsObserver(BaseObserver):
    """Observer that monitors heartbeat frames from a specific processor.

    This observer watches for HeartbeatFrames from a target processor and
    invokes a callback when they are detected, useful for testing timing
    and lifecycle events.
    """

    def __init__(
        self,
        *,
        target: FrameProcessor,
        heartbeat_callback: Callable[[FrameProcessor, HeartbeatFrame], Awaitable[None]],
        **kwargs,
    ):
        """Initialize the heartbeats observer.

        Args:
            target: The frame processor to monitor for heartbeat frames.
            heartbeat_callback: Async callback function to invoke when heartbeats are detected.
            **kwargs: Additional arguments passed to the parent observer.
        """
        super().__init__(**kwargs)
        self._target = target
        self._callback = heartbeat_callback

    async def on_push_frame(self, data: FramePushed):
        """Handle frame push events and detect heartbeats from target processor.

        Args:
            data: The frame push event data containing source and frame information.
        """
        src = data.source
        frame = data.frame

        if src == self._target and isinstance(frame, HeartbeatFrame):
            await self._callback(self._target, frame)


class QueuedFrameProcessor(FrameProcessor):
    """A processor that captures frames in a queue for testing purposes.

    This processor intercepts frames flowing in a specific direction and
    stores them in a queue for later inspection during testing, while
    still allowing the frames to continue through the pipeline.
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue,
        queue_direction: FrameDirection,
        ignore_start: bool = True,
    ):
        """Initialize the queued frame processor.

        Args:
            queue: The asyncio queue to store captured frames.
            queue_direction: The direction of frames to capture (UPSTREAM or DOWNSTREAM).
            ignore_start: Whether to ignore StartFrames when capturing.
        """
        super().__init__(enable_direct_mode=True)
        self._queue = queue
        self._queue_direction = queue_direction
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and capture them in the queue if they match the direction.

        Args:
            frame: The frame to process.
            direction: The direction the frame is flowing.
        """
        await super().process_frame(frame, direction)

        if direction == self._queue_direction:
            if not isinstance(frame, StartFrame) or not self._ignore_start:
                await self._queue.put(frame)
        await self.push_frame(frame, direction)


async def run_test(
    processor: FrameProcessor,
    *,
    enable_rtvi: bool = False,
    expected_down_frames: Sequence[type] | None = None,
    expected_up_frames: Sequence[type] | None = None,
    frames_to_send: Sequence[Frame],
    frames_to_send_direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ignore_start: bool = True,
    observers: list[BaseObserver] | None = None,
    pipeline_params: PipelineParams | None = None,
    send_end_frame: bool = True,
    start_timeout: float = 5.0,
) -> tuple[Sequence[Frame], Sequence[Frame]]:
    """Run a test pipeline with the specified processor and validate frame flow.

    This function creates a test pipeline with the given processor, sends the
    specified frames through it, and validates that the expected frames are
    received in both upstream and downstream directions.

    Args:
        processor: The frame processor to test.
        enable_rtvi: Whether RTVI should be enabled in this test.
        expected_down_frames: Expected frame types flowing downstream (optional).
        expected_up_frames: Expected frame types flowing upstream (optional).
        frames_to_send: Sequence of frames to send through the processor.
        frames_to_send_direction: Direction to send frames_to_send. Downstream
            frames are pushed from the beginning of the pipeline, upstream frames
            from the end. Defaults to DOWNSTREAM.
        ignore_start: Whether to ignore StartFrames in frame validation.
        observers: Optional list of observers to attach to the pipeline.
        pipeline_params: Optional pipeline parameters.
        send_end_frame: Whether to send an EndFrame at the end of the test.
        start_timeout: How long to wait, in seconds, for the pipeline to start
            before giving up.

    Returns:
        Tuple containing (downstream_frames, upstream_frames) that were received.

    Raises:
        AssertionError: If the received frames don't match the expected frame types.
        TimeoutError: If the pipeline doesn't start within ``start_timeout``, or
            ends before it starts.
        RuntimeError: If, after either side has failed, the pipeline doesn't stop
            within the worker's cancel timeout and a margin for its cleanup.
    """
    observers = observers or []
    pipeline_params = pipeline_params or PipelineParams()

    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=ignore_start,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=ignore_start,
    )

    pipeline = Pipeline([source, processor, sink])

    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=enable_rtvi,
        observers=observers,
        params=pipeline_params,
    )

    pipeline_started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_pipeline_started(worker, frame):
        pipeline_started.set()

    async def push_frames():
        # Processors drop frames that arrive before StartFrame, and upstream
        # frames enter at the sink, which StartFrame reaches last.
        await asyncio.wait_for(pipeline_started.wait(), timeout=start_timeout)
        for frame in frames_to_send:
            if isinstance(frame, SleepFrame):
                await asyncio.sleep(frame.sleep)
            else:
                await worker.queue_frame(frame, frames_to_send_direction)

        if send_end_frame:
            await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)

    # Neither side may outlive the other. gather() would leave the runner, and
    # the worker under it, running after the pusher had failed.
    running = asyncio.create_task(runner.run())
    pushing = asyncio.create_task(push_frames())
    try:
        await asyncio.wait((running, pushing), return_when=asyncio.FIRST_COMPLETED)
        if pushing.done():
            pushing.result()
            await running
        else:
            running.result()
            if not pipeline_started.is_set():
                raise TimeoutError("the pipeline ended before it started")
    finally:
        pushing.cancel()
        if not running.done():
            await runner.cancel(reason="run_test is stopping")
        _, still_running = await asyncio.wait((running, pushing), timeout=_STOP_TIMEOUT_SECS)
        if still_running:
            raise RuntimeError(f"the test pipeline did not stop within {_STOP_TIMEOUT_SECS}s")

    #
    # Down frames
    #
    received_down_frames: list[Frame] = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame) or not send_end_frame:
            received_down_frames.append(frame)

    if expected_down_frames is not None:
        down_frames_printed = "["
        for frame in received_down_frames:
            down_frames_printed += f"{frame.__class__.__name__}, "
        down_frames_printed += "]"
        expected_frames_printed = "["
        for frame in expected_down_frames:
            expected_frames_printed += f"{frame.__name__}, "
        expected_frames_printed += "]"
        print("received DOWN frames =", down_frames_printed)
        print("expected DOWN frames =", expected_frames_printed)

        assert len(received_down_frames) == len(expected_down_frames)

        for real, expected in zip(received_down_frames, expected_down_frames):
            assert isinstance(real, expected)

    #
    # Up frames
    #
    received_up_frames: list[Frame] = []
    while not received_up.empty():
        frame = await received_up.get()
        received_up_frames.append(frame)

    if expected_up_frames is not None:
        print("received UP frames =", received_up_frames)
        print("expected UP frames =", expected_up_frames)

        assert len(received_up_frames) == len(expected_up_frames)

        for real, expected in zip(received_up_frames, expected_up_frames):
            assert isinstance(real, expected)

    return (received_down_frames, received_up_frames)
