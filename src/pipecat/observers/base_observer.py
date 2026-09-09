#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base observer classes for monitoring frame flow in the Pipecat pipeline.

This module provides the foundation for observing frame transfers between
processors without modifying the pipeline structure. Observers can be used
for logging, debugging, analytics, and monitoring pipeline behavior.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class FrameProcessed:
    """Event data for frame processing in the pipeline.

    Represents an event where a frame is being processed by a processor. This
    data structure is typically used by observers to track the flow of frames
    through the pipeline for logging, debugging, or analytics purposes.

    Parameters:
        processor: The processor processing the frame.
        frame: The frame being processed.
        direction: The direction of the frame (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.

    """

    processor: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


@dataclass
class FramePushed:
    """Event data for frame transfers between processors in the pipeline.

    Represents an event where a frame is pushed from one processor to another
    within the pipeline. This data structure is typically used by observers
    to track the flow of frames through the pipeline for logging, debugging,
    or analytics purposes.

    Parameters:
        source: The processor sending the frame.
        destination: The processor receiving the frame.
        frame: The frame being transferred.
        direction: The direction of the transfer (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.
    """

    source: "FrameProcessor"
    destination: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.
    """

    #: Frame types this observer's ``on_push_frame`` acts on, or ``None`` for
    #: all of them. ``WorkerObserver`` reads it once per observer and skips the
    #: queue put for a frame the handler would have dropped on its first
    #: ``isinstance``. On a telephony call audio is about 98% of what an
    #: observer is handed, and most observers act on none of it.
    #:
    #: Set it on the class where the handler branches on fixed types, or assign
    #: it in ``__init__`` where the set depends on construction arguments. The
    #: default of ``None`` filters nothing, which is what an observer written
    #: against an older version of this class gets.
    #:
    #: It follows ``isinstance`` semantics, because every handler body does:
    #: declaring ``OutputAudioRawFrame`` also admits ``TTSAudioRawFrame``. A
    #: declaration that omits a type the handler branches on silently stops
    #: that branch from ever running, so ``tests/test_observer_declarations.py``
    #: walks each in-tree handler and holds the declaration against it.
    observed_frame_types: tuple[type[Frame], ...] | None = None

    async def on_process_frame(self, data: FrameProcessed):
        """Handle the event when a frame is being processed by a processor.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        being processed by a processor.

        Args:
            data: The event data containing details about the frame processing.
        """
        pass

    async def on_push_frame(self, data: FramePushed):
        """Handle the event when a frame is pushed from one processor to another.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        transferred through the pipeline.

        Args:
            data: The event data containing details about the frame transfer.
        """
        pass

    async def on_pipeline_started(self):
        """Called when the pipeline has fully started.

        Fired after the ``StartFrame`` has been processed by all processors
        in the pipeline, including nested ``ParallelPipeline`` branches.
        """
        pass
