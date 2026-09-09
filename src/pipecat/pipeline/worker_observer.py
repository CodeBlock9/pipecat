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

from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed

#: The event methods a leaf observer may implement. An event is only queued
#: for an observer that overrides the one it would be delivered to.
_EVENT_HANDLERS = ("on_process_frame", "on_push_frame")


def _implemented_handlers(observer: BaseObserver) -> frozenset[str]:
    """Which of the frame event methods this observer actually overrides.

    Every processor emits a ``FrameProcessed`` for every frame it handles and a
    ``FramePushed`` for every frame it forwards, and both used to be queued for
    every observer. An observer that has not overridden the method an event is
    delivered to inherits ``BaseObserver``'s, whose body is ``pass`` -- so the
    queue put, the task wake-up and the dispatch all bought a no-op. Recorded
    once, when the proxy is created, rather than tested per frame.

    Comparing against ``BaseObserver``'s own functions rather than asking the
    observer anything keeps this right for observers defined outside this tree:
    one that overrides neither method is simply sent neither event.

    Args:
        observer: The observer a proxy is being created for.

    Returns:
        The names of the event methods it overrides.
    """
    return frozenset(
        name
        for name in _EVENT_HANDLERS
        if getattr(type(observer), name, None) is not getattr(BaseObserver, name)
    )


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
        handlers: The event methods this observer overrides. Events destined
            for a method it did not override are never queued.
    """

    queue: asyncio.Queue
    task: asyncio.Task
    observer: BaseObserver
    handlers: frozenset[str]


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
        if self._proxies:
            proxy = self._create_proxy(observer)
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

    async def start(self):
        """Start all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)

    async def stop(self):
        """Stop all proxy observer tasks."""
        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self.cancel_task(proxy.task)

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

        for observer in self._proxies:
            await observer.cleanup()

    async def on_pipeline_started(self):
        """Forward pipeline started signal to all managed observers."""
        await self._send_to_proxy(_PipelineStartedSignal())

    async def on_process_frame(self, data: FrameProcessed):
        """Queue frame data for the observers that handle it.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data, "on_process_frame")

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for the observers that handle it.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data, "on_push_frame")

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        """Create a proxy for a single observer."""
        queue = asyncio.Queue()
        task = self.create_task(self._proxy_task_handler(queue, observer))
        return Proxy(
            queue=queue,
            task=task,
            observer=observer,
            handlers=_implemented_handlers(observer),
        )

    def _create_proxies(self, observers: list[BaseObserver]) -> dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any, handler: str | None = None):
        """Queue an event for every observer that implements its handler.

        Args:
            data: The event to queue.
            handler: Name of the observer method this event will be delivered
                to, or None for an event -- the pipeline-started signal -- that
                every observer receives whatever it implements.
        """
        if not self._proxies:
            return
        for proxy in self._proxies.values():
            if handler is not None and handler not in proxy.handlers:
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

            queue.task_done()
