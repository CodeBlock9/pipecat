#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The idle observer keeps no per-frame state for the length of the call."""

import asyncio

import pytest

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.worker import IdleFrameObserver
from pipecat.processors.frame_processor import FrameDirection


def _pushed(frame):
    return FramePushed(
        source=None,
        destination=None,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


@pytest.mark.asyncio
async def test_the_idle_observer_keeps_no_frame_ids():
    event = asyncio.Event()
    observer = IdleFrameObserver(idle_event=event, idle_timeout_frames=(InputAudioRawFrame,))
    for _ in range(500):
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await observer.on_push_frame(_pushed(frame))

    assert event.is_set()
    assert len(getattr(observer, "_processed_frames", ())) == 0


@pytest.mark.asyncio
async def test_control_a_start_frame_still_sets_the_event():
    event = asyncio.Event()
    observer = IdleFrameObserver(idle_event=event, idle_timeout_frames=(InputAudioRawFrame,))

    await observer.on_push_frame(_pushed(StartFrame()))

    assert event.is_set()
