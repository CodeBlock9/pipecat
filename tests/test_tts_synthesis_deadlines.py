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

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTS_SYNTHESIS_TIMEOUT, TTSService
from pipecat.utils.text.base_text_aggregator import AggregationType


class _StubTTS(TTSService):
    """A service whose generator the test drives."""

    def __init__(self, chunks=None, **kwargs):
        super().__init__(**kwargs)
        self._chunks = chunks or []
        self.errors = []
        self.pushed = []
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

    async def push_error(
        self,
        error_msg,
        exception=None,
        fatal=False,
        force_treat_as_permanent=False,
        **_kwargs,
    ):
        self.errors.append((error_msg, fatal or force_treat_as_permanent))

    async def append_to_audio_context(self, context_id, frame):
        if isinstance(frame, TTSAudioRawFrame):
            await self._cancel_synthesis_watchdog(context_id)

    def create_task(self, coroutine, name=None):
        # The service is exercised outside a pipeline, so there is no task
        # manager to hand it to.
        return asyncio.create_task(coroutine)

    async def cancel_task(self, task, timeout=None):
        task.cancel()

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)

    async def start_processing_metrics(self):
        pass

    async def stop_processing_metrics(self):
        pass

    async def start_ttfb_metrics(self):
        pass

    async def stop_ttfb_metrics(self):
        pass


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
    service = _StubTTS(chunks=[None], synthesis_first_chunk_timeout_s=0.05)

    await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))
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


# ---------------------------------------------------------------------------
# One synthesis is bounded once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_generator_that_yielded_audio_arms_no_context_watchdog():
    """Otherwise an HTTP-shaped service is bounded twice.

    Its generator deadline has already had its say by the time the generator
    ends, so a watchdog also running on the context counts the same stalled
    sentence a second time against `synthesis_timeout_burst` -- and reaches the
    fatal threshold in two stalls rather than three.
    """
    service = _StubTTS(
        chunks=[_audio()],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    await service.tts_process_generator("ctx", service.run_tts("hi", "ctx"))
    await asyncio.sleep(0.2)

    assert service.errors == []
    assert service._synthesis_watchdogs == {}


@pytest.mark.asyncio
async def test_a_context_that_ends_with_no_audio_fires_no_timeout():
    """A filtered sentence, or a provider answering with an empty stream.

    The watchdog was cleared only by a `TTSAudioRawFrame`, so a synthesis that
    legitimately produced none fired a timeout seconds after it was over.
    """
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)
    await service._arm_synthesis_watchdog("ctx")

    await service._end_synthesis_watchdog("ctx")
    await asyncio.sleep(0.2)

    assert service.errors == []


@pytest.mark.asyncio
async def test_a_fired_watchdog_leaves_no_entry_behind():
    """One entry per synthesis, kept for the life of the call, is a leak."""
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.02)
    await service._arm_synthesis_watchdog("ctx")

    await asyncio.sleep(0.2)

    assert service.errors
    assert service._synthesis_watchdogs == {}


# ---------------------------------------------------------------------------
# ... including the synthesis that stalled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_stalled_generator_shaped_synthesis_reports_once():
    """The stall path was still bounded twice.

    The watchdog was armed for *every* synthesis, before the request went out,
    and disarmed only where the generator yielded something. A generator that
    hits its own deadline yields nothing, so the watchdog -- carrying the same
    deadline, started marginally earlier -- reported the same stalled sentence
    a second time.
    """
    service = _StubTTS(
        chunks=[30.0],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    await service._push_tts_frames(AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE))
    await asyncio.sleep(0.3)

    assert len(service.errors) == 1, service.errors
    assert service._synthesis_watchdogs == {}


@pytest.mark.asyncio
async def test_three_stalled_sentences_reach_the_burst_and_not_two():
    """The burst counts stalled sentences, not reports of them."""
    service = _StubTTS(
        chunks=[30.0],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
        synthesis_timeout_burst=3,
    )

    for _ in range(3):
        await service._push_tts_frames(AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE))
    await asyncio.sleep(0.3)

    assert [fatal for _msg, fatal in service.errors] == [False, False, True]


@pytest.mark.asyncio
async def test_a_control_frame_is_not_audio_and_does_not_disarm_the_watchdog():
    """The websocket services that lose the watchdog by yielding one frame.

    Dograh, ElevenLabs, Rime and NVIDIA all yield a `TTSStartedFrame` from
    `run_tts` when the context does not exist yet -- the first sentence of
    every turn -- and then return, with the audio arriving on their receive
    loop. "Yielded anything" reads that as an HTTP-shaped service that has
    bounded itself, so no watchdog is armed and a websocket that never sends
    the audio is not bounded at all. Which is the first sentence of the turn,
    on four of the estate's services.
    """
    service = _StubTTS(
        chunks=[TTSStartedFrame(context_id="ctx")],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    await service._push_tts_frames(
        AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE)
    )
    await asyncio.sleep(0.3)

    assert len(service.errors) == 1, service.errors
    assert service.errors[0][0].startswith(TTS_SYNTHESIS_TIMEOUT)


@pytest.mark.asyncio
async def test_audio_after_a_control_frame_still_disarms_it():
    """The same turn, when the receive loop does deliver."""
    service = _StubTTS(
        chunks=[TTSStartedFrame(context_id="ctx")],
        synthesis_first_chunk_timeout_s=0.05,
        synthesis_chunk_gap_timeout_s=0.05,
    )

    await service._push_tts_frames(
        AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE)
    )
    # The receive loop appends the audio a moment later, as it would.
    context_id = next(iter(service._synthesis_watchdogs))
    await service.append_to_audio_context(context_id, _audio())
    await asyncio.sleep(0.2)

    assert service.errors == []
    assert service._synthesis_watchdogs == {}


# ---------------------------------------------------------------------------
# ... and disarmed by every way a synthesis can end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_barge_in_during_the_arming_window_fires_no_timeout():
    """The window the round-4 widening opened, and the frame that lands in it.

    Arming after `run_tts` returns means a websocket-shaped service is watched
    from the moment its request is away until its first audio chunk -- and that
    is exactly the window a caller barges into on the first sentence of a turn,
    before the bot has made a sound. `_handle_interruption` tears down the
    aggregator, the sequencer, the serialization queue and the audio contexts
    and never touched `_synthesis_watchdogs`, so the watchdog outlived the
    synthesis it was watching and reported a timeout for audio nobody was
    waiting for any more. Three of those inside the window is fatal.
    """
    from pipecat.frames.frames import InterruptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)

    await service._push_tts_frames(
        AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE)
    )
    assert service._synthesis_watchdogs, "nothing was armed, so this proves nothing"

    await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.25)

    assert service._synthesis_watchdogs == {}
    assert service.errors == []


@pytest.mark.asyncio
async def test_a_generator_that_yielded_an_error_frame_arms_nothing():
    """Every HTTP TTS yields one on its exception path.

    `yield ErrorFrame(...)` is how a Google, OpenAI or Azure synthesis reports
    that it failed. That is not audio, so the widened rule armed a watchdog for
    it -- and then reported a synthesis timeout seconds later for a synthesis
    whose failure had already been pushed. One fault, two reports, and the
    second one counting towards the fatal burst.
    """
    from pipecat.frames.frames import ErrorFrame

    service = _StubTTS(
        chunks=[ErrorFrame("provider said no")],
        synthesis_first_chunk_timeout_s=0.05,
    )

    await service._push_tts_frames(
        AggregatedTextFrame(text="hi", aggregated_by=AggregationType.SENTENCE)
    )
    await asyncio.sleep(0.25)

    assert service._synthesis_watchdogs == {}
    assert service.errors == []


@pytest.mark.asyncio
async def test_a_context_torn_down_by_the_handler_takes_its_watchdog_with_it():
    """`del self._audio_contexts[...]` is not `remove_audio_context`.

    Only the latter calls `_end_synthesis_watchdog`, and the handler that
    finishes a context deletes it directly -- so a context that completed
    without ever appending a `TTSAudioRawFrame` left its watchdog armed and
    firing.
    """
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)

    await service.create_audio_context("ctx")
    await service._arm_synthesis_watchdog("ctx")
    # Mark it for deletion, then let the handler drain and tear it down. The
    # sentinel goes into the queue directly: this stub overrides
    # append_to_audio_context and would swallow it.
    await service._audio_contexts["ctx"].put(None)
    await service._serialization_queue.put("ctx")
    handler = asyncio.create_task(service._audio_context_task_handler())
    await asyncio.sleep(0.1)

    try:
        assert "ctx" not in service._audio_contexts
        assert service._synthesis_watchdogs == {}
        await asyncio.sleep(0.2)
        assert len(service.errors) == 1
        assert not service.errors[0][0].startswith(TTS_SYNTHESIS_TIMEOUT)
    finally:
        handler.cancel()


@pytest.mark.asyncio
async def test_timeout_and_empty_completion_count_one_context_once():
    """A timed-out context is not reported again when playback finds no audio."""
    service = _StubTTS(chunks=[], synthesis_first_chunk_timeout_s=0.05)

    await service._report_synthesis_timeout("ctx", f"{TTS_SYNTHESIS_TIMEOUT}: no audio")
    await service._record_context_audio_outcome("ctx", received_audio=False)

    assert len(service.errors) == 1
    assert service._consecutive_zero_audio_contexts == 0
