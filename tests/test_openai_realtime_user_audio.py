#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio streaming in OpenAIRealtimeLLMService.

These tests drive the service's audio/turn handlers directly with a fake
``send_client_event`` and assert on the client events emitted to the service.
They cover the manual turn-detection (server-VAD-disabled) pre-roll behavior —
audio is appended to the input buffer and mirrored into a rolling pre-roll
buffer that is replayed after an interruption clears the input buffer — and the
server-VAD-enabled path, where no pre-roll is maintained.
"""

import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InputAudioRawFrame, SpeechControlParamsFrame
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import (
    AUTOSIZED_USER_AUDIO_PREROLL_MARGIN_SECS,
    OpenAIRealtimeLLMService,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records the client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    def append_payloads(self) -> list[str]:
        return [e.audio for e in self.events if isinstance(e, events.InputAudioBufferAppendEvent)]


def _make_service(*, manual_turn_detection: bool, preroll_secs: float | None = None):
    """Construct a service wired to a fake send_client_event. ``__init__`` does no I/O."""
    if manual_turn_detection:
        settings = OpenAIRealtimeLLMService.Settings(
            session_properties=SessionProperties(
                audio=AudioConfiguration(input=AudioInput(turn_detection=False))
            )
        )
    else:
        settings = None

    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=settings,
        user_audio_preroll_secs=preroll_secs,
    )

    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]

    async def _noop(*args, **kwargs):
        pass

    # _handle_interruption stops metrics, which needs a started pipeline; stub
    # it out so the handler can run in isolation.
    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


def _audio_frame(
    *, sample_rate: int = OPENAI_SAMPLE_RATE, data: bytes = b"\x01\x02" * 160
) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=sample_rate, num_channels=1)


# ---------------------------------------------------------------------------
# Manual turn detection: maintain and replay the pre-roll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pcm_audio_resampled_to_openai_sample_rate():
    """PCM input at another rate is resampled before it is sent to OpenAI."""
    service, recorder = _make_service(manual_turn_detection=False)
    source_audio = b"\xaa\xbb" * 160
    resampled_audio = b"\xcc\xdd" * 240
    service._input_resampler.resample = AsyncMock(return_value=resampled_audio)

    await service._send_user_audio(_audio_frame(sample_rate=16000, data=source_audio))

    service._input_resampler.resample.assert_awaited_once_with(
        source_audio, 16000, OPENAI_SAMPLE_RATE
    )
    assert recorder.append_payloads() == [base64.b64encode(resampled_audio).decode()]


@pytest.mark.asyncio
async def test_manual_mode_appends_and_buffers_audio():
    """Audio is appended to the input buffer and mirrored into the pre-roll buffer."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent", "InputAudioBufferAppendEvent"]
    assert recorder.append_payloads() == [
        base64.b64encode(b"\xaa\xbb").decode(),
        base64.b64encode(b"\xcc\xdd").decode(),
    ]
    assert bytes(service._user_audio_preroll_buffer) == b"\xaa\xbb\xcc\xdd"


@pytest.mark.asyncio
async def test_manual_mode_replays_preroll_after_interruption():
    """An interruption clears the input buffer, then replays the buffered onset."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    await service._handle_interruption()

    # Clear must come before the replay append, and the response cancel after.
    assert recorder.kinds() == [
        "InputAudioBufferAppendEvent",
        "InputAudioBufferAppendEvent",
        "InputAudioBufferClearEvent",
        "InputAudioBufferAppendEvent",
        "ResponseCancelEvent",
    ]
    # The replay re-appends the full buffered onset.
    assert recorder.append_payloads()[-1] == base64.b64encode(b"\xaa\xbb\xcc\xdd").decode()
    # The buffer is left intact (rolling window), so a later interruption can replay again.
    assert bytes(service._user_audio_preroll_buffer) == b"\xaa\xbb\xcc\xdd"


@pytest.mark.asyncio
async def test_manual_mode_interruption_with_empty_buffer_skips_replay():
    """With nothing buffered, the interruption clears and cancels but replays nothing."""
    service, recorder = _make_service(manual_turn_detection=True)

    await service._handle_interruption()

    assert recorder.kinds() == ["InputAudioBufferClearEvent", "ResponseCancelEvent"]


def _resample_to_a_distinct_pattern(service) -> None:
    """Stand in for the resampler: the 24 kHz length, in bytes unlike the input.

    The pre-roll holds what went on the wire, so each window test compares the
    buffer with the tail of the resampled payload, byte for byte.
    """

    async def resample(audio, in_rate, out_rate):
        return bytes(i % 251 for i in range(len(audio) * out_rate // in_rate))

    service._input_resampler.resample = resample  # type: ignore[method-assign]


def _wire_audio(recorder: _EventRecorder) -> bytes:
    return b"".join(base64.b64decode(payload) for payload in recorder.append_payloads())


@pytest.mark.asyncio
async def test_manual_mode_preroll_capped_to_default_window():
    """Before VAD params are known, the pre-roll keeps DEFAULT_USER_AUDIO_PREROLL_SECS."""
    service, recorder = _make_service(manual_turn_detection=True)
    _resample_to_a_distinct_pattern(service)

    # 1s at 16kHz goes on the wire as 1s at 24kHz mono / 16-bit = 48000 bytes;
    # the buffer keeps the most recent 0.5s of it = 24000 bytes.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(32000)))

    expected = int(OPENAI_SAMPLE_RATE * 1 * 2 * 0.5)
    assert bytes(service._user_audio_preroll_buffer) == _wire_audio(recorder)[-expected:]
    assert len(service._user_audio_preroll_buffer) == 24000


@pytest.mark.asyncio
async def test_manual_mode_preroll_sized_from_vad_start_secs():
    """A SpeechControlParamsFrame sizes the pre-roll to start_secs + margin."""
    service, recorder = _make_service(manual_turn_detection=True)
    _resample_to_a_distinct_pattern(service)

    start_secs = 0.5
    service._handle_speech_control_params(
        SpeechControlParamsFrame(vad_params=VADParams(start_secs=start_secs))
    )
    # Send 2s of audio — comfortably more than the window — so the buffer is
    # capped to (start_secs + margin), not limited by how much we sent.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(64000)))

    # The window is measured at the wire rate: 24kHz mono / 16-bit.
    expected = int(
        OPENAI_SAMPLE_RATE * 1 * 2 * (start_secs + AUTOSIZED_USER_AUDIO_PREROLL_MARGIN_SECS)
    )
    assert bytes(service._user_audio_preroll_buffer) == _wire_audio(recorder)[-expected:]
    assert len(service._user_audio_preroll_buffer) == expected


@pytest.mark.asyncio
async def test_manual_mode_preroll_override_pins_value_and_ignores_vad_params():
    """An explicit user_audio_preroll_secs pins the pre-roll; VAD params don't resize it."""
    service, recorder = _make_service(manual_turn_detection=True, preroll_secs=0.1)
    _resample_to_a_distinct_pattern(service)

    service._handle_speech_control_params(
        SpeechControlParamsFrame(vad_params=VADParams(start_secs=0.5))
    )
    # 0.1s at 24kHz mono / 16-bit = 4800 bytes of the wire audio.
    await service._send_user_audio(_audio_frame(sample_rate=16000, data=bytes(32000)))

    assert bytes(service._user_audio_preroll_buffer) == _wire_audio(recorder)[-4800:]
    assert len(service._user_audio_preroll_buffer) == 4800


# ---------------------------------------------------------------------------
# Server-side turn detection: no pre-roll, no clear/replay on interruption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_mode_appends_without_buffering():
    """With server-side turn detection, audio is appended but no pre-roll is kept."""
    service, recorder = _make_service(manual_turn_detection=False)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._send_user_audio(_audio_frame(data=b"\xcc\xdd"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent", "InputAudioBufferAppendEvent"]
    assert bytes(service._user_audio_preroll_buffer) == b""


@pytest.mark.asyncio
async def test_server_mode_interruption_does_not_clear_or_replay():
    """With server-side turn detection, an interruption doesn't touch the input buffer."""
    service, recorder = _make_service(manual_turn_detection=False)

    await service._send_user_audio(_audio_frame(data=b"\xaa\xbb"))
    await service._handle_interruption()

    assert "InputAudioBufferClearEvent" not in recorder.kinds()
    assert "ResponseCancelEvent" not in recorder.kinds()
