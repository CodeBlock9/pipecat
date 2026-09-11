#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
from typing import Any

from attr import dataclass
from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import (
    BaseObserver,
    FrameProcessed,
    FramePushed,
    ProcessorSetUp,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager

_EVENT_HANDLERS = ("on_process_frame", "on_push_frame", "on_processor_setup")


def _implemented_handlers(observer: BaseObserver) -> frozenset[str]:
    """Return the high-volume event methods implemented by an observer."""
    return frozenset(
        name
        for name in _EVENT_HANDLERS
        if name in getattr(observer, "__dict__", ())
        or getattr(type(observer), name, None) is not getattr(BaseObserver, name)
    )


def _declared_push_types(observer: BaseObserver) -> tuple[type[Frame], ...] | None:
    """Validate an observer's optional pushed-frame filter."""
    declared = getattr(observer, "observed_frame_types", None)
    if declared is None:
        return None
    if isinstance(declared, type):
        declared = (declared,)
    try:
        types = tuple(declared)
    except TypeError:
        types = None
    if types is None or not all(isinstance(entry, type) for entry in types):
        logger.warning(
            f"{observer} declares observed_frame_types={declared!r}, which is not a "
            "tuple of frame classes; it will be sent every frame"
        )
        return None
    return types


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
    """

    queue: asyncio.Queue
    task: asyncio.Task
    observer: BaseObserver
    handlers: frozenset[str]
    push_types: tuple[type[Frame], ...] | None
    push_decisions: dict[type[Frame], bool]

    def observes(self, frame: Frame, frame_type: type[Frame]) -> bool:
        """Return whether this proxy accepts the concrete pushed-frame type."""
        if self.push_types is None:
            return True
        decided = self.push_decisions.get(frame_type)
        if decided is None:
            decided = isinstance(frame, self.push_types)
            self.push_decisions[frame_type] = decided
        return decided


class _PipelineStartedSignal:
    """Internal sentinel queued to observers when the pipeline has started."""

    pass


class WorkerObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline worker.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.
    """

    def __init__(
        self,
        *,
        observers: list[BaseObserver] | None = None,
        **kwargs,
    ):
        """Initialize the WorkerObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._proxies: dict[BaseObserver, Proxy] | None = (
            None  # Becomes a dict after start() is called
        )

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)

        # If we already started, create a new proxy for the observer.
        # Otherwise, it will be created in start().
        if self._proxies is not None:
            # The public registration API is synchronous. Queue delivery behind
            # the observer's asynchronous setup so a late observer still sees
            # the same lifecycle as one supplied at construction time.
            proxy = self._create_proxy(observer, setup_observer=True)
            self._proxies[observer] = proxy

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer and clean up its resources.

        Args:
            observer: The observer to remove.
        """
        # If the observer has a proxy, remove it.
        if self._proxies and observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            # Cancel the proxy worker right away.
            await self.cancel_task(proxy.task)

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)

    async def setup(self, task_manager: BaseTaskManager):
        """Set up a proxy for every managed observer.

        Processors report their own setup to observers, so the proxies are in
        place before any of them is set up.

        Args:
            task_manager: The task manager the proxies run their tasks on.
        """
        await super().setup(task_manager)
        self._proxies = {}
        for observer in list(self._observers):
            await observer.setup(task_manager)
            # An observer added concurrently after `_proxies` became a dict
            # already has its dynamically initialized proxy.
            if observer not in self._proxies:
                self._proxies[observer] = self._create_proxy(observer)

    async def wait_until_idle(self) -> None:
        """Wait until every observer has processed its currently queued frames."""
        if not self._proxies:
            return
        await asyncio.gather(*(proxy.queue.join() for proxy in self._proxies.values()))

    async def cleanup(self):
        """Cleanup all proxy observers."""
        await super().cleanup()

        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self.cancel_task(proxy.task)

        for observer in self._proxies:
            await observer.cleanup()

    async def on_pipeline_started(self):
        """Forward pipeline started signal to all managed observers."""
        await self._send_to_proxy(_PipelineStartedSignal())

    async def on_process_frame(self, data: FrameProcessed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data, "on_process_frame")

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data, "on_push_frame")

    async def on_processor_setup(self, data: ProcessorSetUp):
        """Queue processor setup timing for all managed observers.

        Args:
            data: The processor setup event data to distribute to observers.
        """
        await self._send_to_proxy(data, "on_processor_setup")

    def _create_proxy(self, observer: BaseObserver, *, setup_observer: bool = False) -> Proxy:
        """Create a proxy for a single observer."""
        queue = asyncio.Queue()

        async def run_proxy():
            if setup_observer:
                await observer.setup(self.task_manager)
            await self._proxy_task_handler(queue, observer)

        task = self.create_task(run_proxy())
        return Proxy(
            queue=queue,
            task=task,
            observer=observer,
            handlers=_implemented_handlers(observer),
            push_types=_declared_push_types(observer),
            push_decisions={},
        )

    def _create_proxies(self, observers: list[BaseObserver]) -> dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any, handler: str | None = None):
        if not self._proxies:
            return
        frame = data.frame if handler == "on_push_frame" else None
        frame_type = type(frame) if frame is not None else None
        for proxy in self._proxies.values():
            if handler is not None and handler not in proxy.handlers:
                continue
            if frame_type is not None and not proxy.observes(frame, frame_type):
                continue
            await proxy.queue.put(data)

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        """Handle frame processing for a single observer."""
        while True:
            data = await queue.get()

            if isinstance(data, _PipelineStartedSignal):
                await observer.on_pipeline_started()
            elif isinstance(data, FramePushed):
                await observer.on_push_frame(data)
            elif isinstance(data, FrameProcessed):
                await observer.on_process_frame(data)
            elif isinstance(data, ProcessorSetUp):
                await observer.on_processor_setup(data)

            queue.task_done()
