#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What a raising observer callback costs: that one event, never its queue.

``WorkerObserver`` delivers each observer's events from its own queue and task.
A callback that raises must not end that task: the observer's later events
would be lost, and ``wait_until_idle`` (the drain a pipeline runs at its end)
would never resolve.
"""

import asyncio
import unittest

from loguru import logger

from pipecat.frames.frames import Frame, TextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager


class RaisingOnceObserver(BaseObserver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0
        self.pushed: list[Frame] = []

    async def on_push_frame(self, data: FramePushed):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad observer")
        self.pushed.append(data.frame)


class AlwaysRaisingObserver(BaseObserver):
    async def on_push_frame(self, data: FramePushed):
        raise ValueError("bad observer")


class HealthyObserver(BaseObserver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushed: list[Frame] = []

    async def on_push_frame(self, data: FramePushed):
        self.pushed.append(data.frame)


def _pushed(frame: Frame) -> FramePushed:
    return FramePushed(
        source=None,  # type: ignore[arg-type]
        destination=None,  # type: ignore[arg-type]
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


class TestObserverFailure(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_callback_does_not_stop_later_deliveries(self):
        """The proxy task survives, the queue drains, and the next event arrives."""
        raising = RaisingOnceObserver()
        healthy = HealthyObserver()
        proxy = WorkerObserver(observers=[raising, healthy])
        await proxy.setup(TaskManager(loop=asyncio.get_running_loop()))
        try:
            await proxy.on_push_frame(_pushed(TextFrame("first")))
            await proxy.on_push_frame(_pushed(TextFrame("second")))
            try:
                await asyncio.wait_for(proxy.wait_until_idle(), timeout=1.0)
            except TimeoutError:
                self.fail("wait_until_idle never resolved after one observer callback raised")
            self.assertFalse(
                proxy._proxies[raising].task.done(), "the raising observer's proxy task exited"
            )
            self.assertEqual([f.text for f in raising.pushed], ["second"])
            self.assertEqual([f.text for f in healthy.pushed], ["first", "second"])
        finally:
            await proxy.cleanup()

    async def test_a_healthy_observer_drains(self):
        """With no failure, idle resolves and both events arrive."""
        healthy = HealthyObserver()
        proxy = WorkerObserver(observers=[healthy])
        await proxy.setup(TaskManager(loop=asyncio.get_running_loop()))
        try:
            await proxy.on_push_frame(_pushed(TextFrame("first")))
            await proxy.on_push_frame(_pushed(TextFrame("second")))
            await asyncio.wait_for(proxy.wait_until_idle(), timeout=1.0)
            self.assertEqual([f.text for f in healthy.pushed], ["first", "second"])
        finally:
            await proxy.cleanup()

    async def test_only_the_first_failure_is_logged_as_an_error(self):
        """Two failures produce one error line, with its traceback; the second is at debug.

        Observers see every push between every processor pair, so a traceback
        per failed event would be formatted on the loop the audio shares, and
        an observer raising on every frame would flood the log.
        """
        raising = AlwaysRaisingObserver()
        proxy = WorkerObserver(observers=[raising])
        await proxy.setup(TaskManager(loop=asyncio.get_running_loop()))
        records = []
        handler_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
        try:
            await proxy.on_push_frame(_pushed(TextFrame("first")))
            await proxy.on_push_frame(_pushed(TextFrame("second")))
            await asyncio.wait_for(proxy.wait_until_idle(), timeout=1.0)
        finally:
            logger.remove(handler_id)
            await proxy.cleanup()

        failures = [r for r in records if "raised handling" in r["message"]]
        self.assertEqual([r["level"].name for r in failures], ["ERROR", "DEBUG"])
        self.assertIn("raised handling FramePushed: bad observer", failures[0]["message"])
        self.assertIsNotNone(failures[0]["exception"], "the first failure has no traceback")
        self.assertIsNone(failures[1]["exception"])
