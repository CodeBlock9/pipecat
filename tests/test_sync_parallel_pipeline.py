#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
)
from pipecat.pipeline.sync_parallel_pipeline import FrameOrder, SyncParallelPipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


@dataclass
class TaggedFrame(Frame):
    """A simple tagged frame for testing pipeline ordering."""

    tag: str = ""

    def __str__(self):
        return f"{self.name}(tag: {self.tag})"


@dataclass
class PingFrame(SystemFrame):
    """A system frame no processor handles: it only travels."""


class EmitTaggedFrameProcessor(FrameProcessor):
    """Emits a TaggedFrame for every TextFrame it receives.

    Used to produce distinguishable output from different pipelines so tests
    can verify ordering.
    """

    def __init__(self, tag: str, *, delay: float = 0, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._delay = delay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            await self.push_frame(TaggedFrame(tag=self._tag))
        else:
            await self.push_frame(frame, direction)


class Collector(FrameProcessor):
    """Records the name of every frame it receives, StartFrames included."""

    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(type(frame).__name__)


class FarewellOnEnd(FrameProcessor):
    """A branch that still has output to flush when its EndFrame arrives."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            await self.push_frame(TextFrame("bye"), direction)
        await self.push_frame(frame, direction)


async def _run_collecting(
    branches, frames, frame_order: FrameOrder = FrameOrder.ARRIVAL
) -> list[str]:
    """Queue frames into a SyncParallelPipeline and return what came out of it.

    Unlike `run_test`, nothing is filtered: a duplicated StartFrame shows.
    """
    sync = SyncParallelPipeline(*branches, frame_order=frame_order)
    collector = Collector()
    sync.link(collector)
    setup = frame_processor_setup(TaskManager(loop=asyncio.get_running_loop()))
    await sync.setup(setup)
    await collector.setup(setup)
    try:
        for frame in frames:
            await sync.queue_frame(frame)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        return list(collector.frames)
    finally:
        await sync.cleanup()
        await collector.cleanup()


class TestSyncParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_multiple_frames(self):
        """Identical frames from multiple paths should be deduplicated."""
        pipeline = SyncParallelPipeline([IdentityFilter()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        expected_down_frames = [TextFrame, TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_arrival_order(self):
        """With FrameOrder.ARRIVAL, a slow first pipeline's frames should
        arrive after a fast second pipeline's frames."""
        pipeline = SyncParallelPipeline(
            [EmitTaggedFrameProcessor("slow", delay=0.05)],
            [EmitTaggedFrameProcessor("fast")],
            frame_order=FrameOrder.ARRIVAL,
        )

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        tags = [f.tag for f in down_frames if isinstance(f, TaggedFrame)]
        assert tags == [
            "fast",
            "slow",
            "fast",
            "slow",
        ], f"Expected fast before slow in each batch, got {tags}"

    async def test_pipeline_order(self):
        """With FrameOrder.PIPELINE and multiple input frames, each batch
        should follow pipeline definition order regardless of processing speed."""
        pipeline = SyncParallelPipeline(
            [EmitTaggedFrameProcessor("slow", delay=0.05)],
            [EmitTaggedFrameProcessor("fast")],
            frame_order=FrameOrder.PIPELINE,
        )

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        tags = [f.tag for f in down_frames if isinstance(f, TaggedFrame)]
        assert tags == [
            "slow",
            "fast",
            "slow",
            "fast",
        ], f"Expected pipeline definition order (slow, fast) in each batch, got {tags}"

    async def test_default_is_arrival(self):
        """The default frame_order should be ARRIVAL."""
        pipeline = SyncParallelPipeline([IdentityFilter()])
        assert pipeline._frame_order == FrameOrder.ARRIVAL


class TestSyncParallelPipelineLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_system_frames_are_forwarded_once(self):
        """A system frame fanned out to the branches is not replayed at the next sync."""
        frames = await _run_collecting(
            [[IdentityFilter()], [IdentityFilter()]],
            [StartFrame(), TextFrame("hello"), InterruptionFrame(), TextFrame("again")],
        )
        self.assertEqual(
            frames, ["StartFrame", "TextFrame", "InterruptionFrame", "TextFrame"], frames
        )

    async def test_system_frames_are_forwarded_once_in_pipeline_order(self):
        """Pipeline order releases output from its own list, which drops the copies too."""
        frames = await _run_collecting(
            [[IdentityFilter()], [IdentityFilter()]],
            [StartFrame(), TextFrame("hello"), InterruptionFrame(), TextFrame("again")],
            frame_order=FrameOrder.PIPELINE,
        )
        self.assertEqual(
            frames, ["StartFrame", "TextFrame", "InterruptionFrame", "TextFrame"], frames
        )

    async def test_upstream_system_frames_are_forwarded_once(self):
        """An upstream system frame's copies are dropped at the next upstream sync."""
        sync = SyncParallelPipeline([IdentityFilter()], [IdentityFilter()])
        upstream = Collector()
        downstream = Collector()
        upstream.link(sync)
        sync.link(downstream)
        setup = frame_processor_setup(TaskManager(loop=asyncio.get_running_loop()))
        for processor in (upstream, sync, downstream):
            await processor.setup(setup)
        try:
            await sync.queue_frame(StartFrame())
            await asyncio.sleep(0.05)
            for frame in (PingFrame(), TextFrame("hello"), TextFrame("again")):
                await sync.queue_frame(frame, FrameDirection.UPSTREAM)
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.2)
            self.assertEqual(upstream.frames, ["PingFrame", "TextFrame", "TextFrame"])
        finally:
            for processor in (upstream, sync, downstream):
                await processor.cleanup()

    async def test_end_frame_survives_a_branch_that_flushes_output_first(self):
        """Both farewells arrive, then the EndFrame, once."""
        frames = await _run_collecting(
            [[FarewellOnEnd()], [FarewellOnEnd()]], [StartFrame(), TextFrame("hello"), EndFrame()]
        )
        self.assertEqual(frames.count("EndFrame"), 1, frames)
        self.assertEqual(frames[-1], "EndFrame", frames)
        self.assertEqual(frames.count("TextFrame"), 3, f"hello and both farewells: {frames}")

    async def test_end_frame_goes_last_in_pipeline_order(self):
        """In pipeline order the EndFrame follows the last branch's farewell, not the first's."""
        frames = await _run_collecting(
            [[FarewellOnEnd()], [FarewellOnEnd()]],
            [StartFrame(), TextFrame("hello"), EndFrame()],
            frame_order=FrameOrder.PIPELINE,
        )
        self.assertEqual(frames.count("EndFrame"), 1, frames)
        self.assertEqual(frames[-1], "EndFrame", frames)
        self.assertEqual(frames.count("TextFrame"), 3, f"hello and both farewells: {frames}")

    async def test_end_frame_arrives_when_nothing_is_flushed(self):
        """With no late output the EndFrame is forwarded."""
        frames = await _run_collecting(
            [[IdentityFilter()], [IdentityFilter()]], [StartFrame(), TextFrame("hello"), EndFrame()]
        )
        self.assertEqual(frames.count("EndFrame"), 1, frames)
        self.assertEqual(frames[-1], "EndFrame", frames)


if __name__ == "__main__":
    unittest.main()
