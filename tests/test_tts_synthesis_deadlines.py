#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a synthesis can be given a deadline, in both shapes it takes.

``run_tts`` has no deadline of its own. The 3s pause watchdog covers a
synthesiser that returns *nothing* — which is why a service that drops a
sentence recovers on the next one — and not one that never returns at all: the
generator stays parked and the call goes silent for as long as the caller is
prepared to wait.

Two shapes, because two service families. An HTTP-shaped service yields its
audio from the generator, so the deadline goes around the iteration. A
websocket-shaped service yields ``None`` and delivers audio through its own
receive loop, so the generator finishes at once and the same deadline has to be
expressed on the context instead.
"""

import asyncio

import pytest

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.tts_service import TTS_SYNTHESIS_TIMEOUT, TTSService


class _StubTTS(TTSService):
    """A service whose generator the test drives."""

    def __init__(self, chunks=None, **kwargs):
        super().__init__(**kwargs)
        self._chunks = chunks or []
        self.errors = []
        self.closed = False

    async def run_tts(self, text: str, context_id: str):
        for item in self._chunks:
            if isinstance(item, float):
                try:
                    await asyncio.sleep(item)
                except asyncio.CancelledError:
                    self.closed = True
                    raise
                continue
            yield item

    async def push_error(self, error_msg, exception=None, fatal=False):
        self.errors.append((error_msg, fatal))

    async def append_to_audio_context(self, context_id, frame):
        if isinstance(frame, TTSAudioRawFrame):
            await self._cancel_synthesis_watchdog(context_id)

    def create_task(self, coroutine, name=None):
        # The service is exercised outside a pipeline, so there is no task
        # manager to hand it to.
        return asyncio.create_task(coroutine)

    async def cancel_task(self, task, timeout=None):
        task.cancel()


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(b"\x00\x00", 16000, 1)


@pytest.mark.asyncio
async def test_an_unbounded_service_is_left_alone():
    """Both deadlines default to None, so upstream behaviour is unchanged."""
    service = _StubTTS(chunks=[_audio(), _audio()])

    await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))

    assert service.errors == []


@pytest.mark.asyncio
async def test_a_generator_that_never_yields_is_abandoned_at_the_deadline():
    """The measured failure: the generator parks and the call goes silent."""
    service = _StubTTS(
        chunks=[30.0, _audio()],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))
    elapsed = loop.time() - started

    assert elapsed < 1.0
    assert service.errors, "no error was reported for the abandoned synthesis"
    message, fatal = service.errors[0]
    assert message.startswith(TTS_SYNTHESIS_TIMEOUT)
    assert fatal is False, "one dropped sentence is not fatal"


@pytest.mark.asyncio
async def test_a_stopped_stream_is_bounded_by_the_gap_deadline():
    """First audio arrives, then nothing: the gap deadline is what ends it."""
    service = _StubTTS(
        chunks=[_audio(), 30.0, _audio()],
        synthesis_first_chunk_timeout_s=5.0,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))

    assert service.errors
    assert "further audio" in service.errors[0][0]


@pytest.mark.asyncio
async def test_a_burst_of_timeouts_becomes_fatal():
    """A synthesiser that is not coming back should end the call, not the sentence."""
    service = _StubTTS(
        chunks=[30.0],
        synthesis_first_chunk_timeout_s=0.02,
        synthesis_chunk_gap_timeout_s=0.02,
        synthesis_timeout_burst=3,
    )

    for _ in range(3):
        await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))

    assert [fatal for _msg, fatal in service.errors] == [False, False, True]


@pytest.mark.asyncio
async def test_timeouts_spread_beyond_the_window_never_escalate():
    service = _StubTTS(
        chunks=[30.0],
        synthesis_first_chunk_timeout_s=0.02,
        synthesis_chunk_gap_timeout_s=0.02,
        synthesis_timeout_burst=3,
        synthesis_timeout_window_s=0.0,
    )

    for _ in range(3):
        await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))

    assert all(not fatal for _msg, fatal in service.errors)


@pytest.mark.asyncio
async def test_the_context_watchdog_bounds_a_websocket_shaped_service():
    """It yields None and delivers audio elsewhere, so the generator is instant."""
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)
    await service._arm_synthesis_watchdog("ctx")

    await asyncio.sleep(0.2)

    assert service.errors
    assert service.errors[0][0].startswith(TTS_SYNTHESIS_TIMEOUT)


@pytest.mark.asyncio
async def test_the_first_audio_frame_disarms_the_context_watchdog():
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)
    await service._arm_synthesis_watchdog("ctx")
    await service.append_to_audio_context("ctx", _audio())

    await asyncio.sleep(0.2)

    assert service.errors == []


@pytest.mark.asyncio
async def test_no_watchdog_is_armed_when_no_deadline_is_configured():
    service = _StubTTS(chunks=[])
    await service._arm_synthesis_watchdog("ctx")

    assert service._synthesis_watchdogs == {}
