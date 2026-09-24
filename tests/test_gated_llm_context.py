#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, StartFrame
from pipecat.processors.aggregators.gated_llm_context import GatedLLMContextAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.sync.event_notifier import EventNotifier
from tests.frame_processor_helpers import frame_processor_setup


class TestGatedStartFrame(unittest.IsolatedAsyncioTestCase):
    async def _pushed_for(self, frame) -> list[str]:
        """Return the frames the real gate pushes for ``frame``, after a StartFrame.

        Only ``push_frame`` is captured; the gate runs on a real task manager and
        is torn down through its real ``cleanup()``.
        """
        gate = GatedLLMContextAggregator(notifier=EventNotifier())
        await gate.setup(frame_processor_setup(TaskManager(loop=asyncio.get_running_loop())))
        pushed: list[str] = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(type(frame).__name__)

        gate.push_frame = capture
        try:
            await gate.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
            if not isinstance(frame, StartFrame):
                pushed.clear()
                await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
            return pushed
        finally:
            await gate.cleanup()

    async def test_start_frame_is_forwarded_once(self):
        """The StartFrame branch pushes it, and nothing pushes it again."""
        pushed = await self._pushed_for(StartFrame())
        self.assertEqual(pushed, ["StartFrame"], f"pushed: {pushed}")

    async def test_control_end_frame_is_forwarded_once(self):
        """The EndFrame branch is part of the elif chain."""
        pushed = await self._pushed_for(EndFrame())
        self.assertEqual(pushed, ["EndFrame"], f"pushed: {pushed}")


if __name__ == "__main__":
    unittest.main()
