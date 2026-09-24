#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime: the manual-mode pre-roll replays what went on the wire.

With turn detection disabled, ``_send_user_audio`` resamples PCM that is not at
24 kHz before it appends it, and ``_handle_interruption`` replays the pre-roll
after clearing the input buffer. The replay must therefore be the resampled
audio, and its window must be measured at the wire rate.
"""

import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class _Recorder:
    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def append_payloads(self) -> list[bytes]:
        return [
            base64.b64decode(e.audio)
            for e in self.events
            if isinstance(e, events.InputAudioBufferAppendEvent)
        ]


def _manual_service():
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            session_properties=SessionProperties(
                audio=AudioConfiguration(input=AudioInput(turn_detection=False))
            )
        ),
    )
    recorder = _Recorder()
    service.send_client_event = recorder  # type: ignore[method-assign]

    async def _noop(*args, **kwargs):
        pass

    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


def _frame(data: bytes, sample_rate: int) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=sample_rate, num_channels=1)


@pytest.mark.asyncio
async def test_the_replay_is_the_audio_that_went_on_the_wire():
    service, recorder = _manual_service()
    source = b"\xaa\xbb" * 160  # 10 ms at 16 kHz
    wire = b"\xcc\xdd" * 240  # the same 10 ms at 24 kHz
    service._input_resampler.resample = AsyncMock(return_value=wire)

    await service._send_user_audio(_frame(source, 16000))
    await service._handle_interruption()

    appended, replayed = recorder.append_payloads()
    assert appended == wire
    assert replayed == wire


@pytest.mark.asyncio
async def test_the_window_is_measured_at_the_wire_rate():
    """1 s of 16 kHz input through the real resampler; the default 0.5 s window is 24 kHz audio."""
    service, recorder = _manual_service()

    await service._send_user_audio(_frame(bytes(32000), 16000))

    # The streaming resampler's first output is a little short of 48000 bytes,
    # because it holds back its filter latency. Either way it is more than the
    # window, so the window is full.
    wire_bytes = len(recorder.append_payloads()[0])
    expected = int(OPENAI_SAMPLE_RATE * 1 * 2 * service._user_audio_preroll_secs)
    assert wire_bytes > expected
    assert len(service._user_audio_preroll_buffer) == expected


@pytest.mark.asyncio
async def test_control_24k_input_replays_what_was_sent():
    service, recorder = _manual_service()

    await service._send_user_audio(_frame(b"\x01\x02" * 240, OPENAI_SAMPLE_RATE))
    await service._handle_interruption()

    appended, replayed = recorder.append_payloads()
    assert replayed == appended
