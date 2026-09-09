#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What ``WorkerObserver`` puts on each observer's queue, and what it does not.

The proxy fans every frame event out to every registered observer. On a
telephony call that is about 13,875 queue puts a second across five observers,
and every one of them wakes a task. Two things narrow it: an event is not
queued for an observer that never overrode the handler it would be delivered
to, and a push event is not queued for an observer that declared the frame
types its handler acts on and did not name this one. Audio is about 98% of
what an observer is handed, and none of the observers in a Mesa pipeline acts
on any of it.
"""

import asyncio
import unittest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager


class PushOnlyObserver(BaseObserver):
    """Overrides only ``on_push_frame`` — the shape every Mesa observer has."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushed: list[Frame] = []

    async def on_push_frame(self, data: FramePushed):
        self.pushed.append(data.frame)


class BothHandlersObserver(BaseObserver):
    """Overrides both handlers, as ``StartupTimingObserver`` does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processed: list[Frame] = []
        self.pushed: list[Frame] = []

    async def on_process_frame(self, data: FrameProcessed):
        self.processed.append(data.frame)

    async def on_push_frame(self, data: FramePushed):
        self.pushed.append(data.frame)


class SilentObserver(BaseObserver):
    """Overrides nothing. Every event it is sent runs ``pass``."""


class DeclaredObserver(PushOnlyObserver):
    """Names the frame types its push handler acts on."""

    observed_frame_types = (TextFrame,)


class SubclassAwareObserver(PushOnlyObserver):
    """Declares a base class, and must still be sent the subclasses."""

    observed_frame_types = (TTSAudioRawFrame.__mro__[1],)  # OutputAudioRawFrame


def _processed(frame: Frame) -> FrameProcessed:
    return FrameProcessed(
        processor=None,  # type: ignore[arg-type]
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def _pushed(frame: Frame) -> FramePushed:
    return FramePushed(
        source=None,  # type: ignore[arg-type]
        destination=None,  # type: ignore[arg-type]
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def _record_puts(proxy) -> list:
    """Everything the fan-out queues for one observer, in order.

    The queue is what the fan-out actually touches, so wrapping it measures
    exactly what these tests are about. Wrapping the observer instead would
    change the thing under test: an observer that gained an override because a
    test wrapped it would stop being filtered.
    """
    recorded: list = []
    original = proxy.queue.put

    async def put(item):
        recorded.append(item)
        await original(item)

    proxy.queue.put = put
    return recorded


async def _started_proxy(observers: list[BaseObserver]) -> WorkerObserver:
    worker_observer = WorkerObserver(
        observers=observers, task_manager=TaskManager(loop=asyncio.get_running_loop())
    )
    await worker_observer.start()
    return worker_observer


class TestHandlerFanOut(unittest.IsolatedAsyncioTestCase):
    async def test_an_observer_is_not_sent_events_it_does_not_handle(self):
        push_only = PushOnlyObserver()
        both = BothHandlersObserver()
        silent = SilentObserver()
        proxy = await _started_proxy([push_only, both, silent])
        try:
            queued = {
                observer: _record_puts(proxy._proxies[observer])
                for observer in (push_only, both, silent)
            }

            await proxy.on_process_frame(_processed(TextFrame("hello")))
            await proxy.on_push_frame(_pushed(TextFrame("hello")))
            await proxy.wait_until_idle()

            self.assertEqual([type(e) for e in queued[push_only]], [FramePushed])
            self.assertEqual([type(e) for e in queued[both]], [FrameProcessed, FramePushed])
            self.assertEqual(queued[silent], [])
        finally:
            await proxy.stop()

    async def test_the_pipeline_started_signal_reaches_every_observer(self):
        """It carries no frame at all, so no type dispatch may touch it."""
        push_only = PushOnlyObserver()
        silent = SilentObserver()
        proxy = await _started_proxy([push_only, silent])
        try:
            queued = {
                observer: _record_puts(proxy._proxies[observer]) for observer in (push_only, silent)
            }

            await proxy.on_pipeline_started()
            await proxy.wait_until_idle()

            self.assertEqual(len(queued[push_only]), 1)
            self.assertEqual(len(queued[silent]), 1)
        finally:
            await proxy.stop()


class TestDeclaredPushTypes(unittest.IsolatedAsyncioTestCase):
    async def test_a_declared_away_frame_never_reaches_the_observer(self):
        declared = DeclaredObserver()
        undeclared = PushOnlyObserver()
        proxy = await _started_proxy([declared, undeclared])
        try:
            audio = InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=8000, num_channels=1)
            await proxy.on_push_frame(_pushed(audio))
            await proxy.on_push_frame(_pushed(TextFrame("hello")))
            await proxy.wait_until_idle()

            self.assertEqual([type(f) for f in declared.pushed], [TextFrame])
            # An observer that declares nothing keeps receiving everything.
            self.assertEqual([type(f) for f in undeclared.pushed], [InputAudioRawFrame, TextFrame])
        finally:
            await proxy.stop()

    async def test_a_declared_base_class_still_admits_its_subclasses(self):
        """Every observer body branches with isinstance; the filter must agree."""
        observer = SubclassAwareObserver()
        proxy = await _started_proxy([observer])
        try:
            await proxy.on_push_frame(
                _pushed(TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=8000, num_channels=1))
            )
            await proxy.on_push_frame(_pushed(TextFrame("hello")))
            await proxy.wait_until_idle()

            self.assertEqual([type(f) for f in observer.pushed], [TTSAudioRawFrame])
        finally:
            await proxy.stop()

    async def test_a_malformed_declaration_is_ignored_rather_than_obeyed(self):
        """Observers arrive from packages versioned separately from this one."""
        observer = PushOnlyObserver()
        observer.observed_frame_types = "StartFrame"  # type: ignore[assignment]
        proxy = await _started_proxy([observer])
        try:
            await proxy.on_push_frame(_pushed(TextFrame("hello")))
            await proxy.wait_until_idle()

            self.assertEqual([type(f) for f in observer.pushed], [TextFrame])
        finally:
            await proxy.stop()


class TestAddObserverAfterStart(unittest.IsolatedAsyncioTestCase):
    async def test_an_observer_added_after_start_gets_a_proxy(self):
        """``start()`` with no observers left an empty dict, which is falsy."""
        proxy = await _started_proxy([])
        try:
            late = PushOnlyObserver()
            proxy.add_observer(late)

            await proxy.on_push_frame(_pushed(StartFrame()))
            await proxy.wait_until_idle()

            self.assertEqual([type(f) for f in late.pushed], [StartFrame])
        finally:
            await proxy.stop()
