#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import wave
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    STTMuteFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# Distinct, non-zero 16-bit samples so a misread WAV header would be obvious.
PCM = bytes(range(0, 240)) * 4  # 960 bytes, even length

# 10 ms each, told apart by their sample value.
PREROLL = b"\x07\x00" * 160
BEFORE = b"\x01\x00" * 160
MUTED = b"\x02\x00" * 160
AFTER = b"\x03\x00" * 160


def _make_capturing_service(wants_wav: bool | None = None) -> SegmentedSTTService:
    """Build a SegmentedSTTService that captures the bytes handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py, which
    would try to construct it and fail on its (intentionally minimal) settings.

    Args:
        wants_wav: If None, inherit the base default; otherwise force the
            ``wants_wav_segments`` contract to this value.
    """

    class _CapturingSegmentedSTTService(SegmentedSTTService):
        def __init__(self, **kwargs):
            super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
            self.captured: list[bytes] = []

        def can_generate_metrics(self) -> bool:
            return True

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    if wants_wav is not None:
        _CapturingSegmentedSTTService.wants_wav_segments = property(lambda self: wants_wav)

    return _CapturingSegmentedSTTService()


async def _drive_one_segment(service: SegmentedSTTService):
    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
    )


@pytest.mark.asyncio
async def test_default_mode_wraps_segment_in_wav():
    service = _make_capturing_service()
    assert service.wants_wav_segments is True

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    audio = service.captured[0]

    # A valid WAV container with the right sample rate and the exact PCM payload.
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.asyncio
async def test_passthrough_mode_preserves_exact_pcm():
    service = _make_capturing_service(wants_wav=False)
    assert service.wants_wav_segments is False

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # Raw PCM, byte-for-byte: no WAV header prepended.
    assert service.captured[0] == PCM


@pytest.mark.asyncio
async def test_segment_emits_usage_for_raw_buffer_duration():
    # WAV mode: usage must measure the raw PCM buffer, not the WAV container.
    service = _make_capturing_service()

    received_down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    usage_data = [
        d
        for f in received_down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, STTUsageMetricsData)
    ]
    assert len(usage_data) == 1
    assert usage_data[0].value.audio_seconds == pytest.approx(len(PCM) / (SAMPLE_RATE * 2))


def _audio(pcm: bytes) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1)


async def _run_with_usage(frames_to_send: list[Frame]) -> tuple[list[bytes], list]:
    """Drive a raw-PCM capturing service; return what it transcribed and the usage it reported."""
    service = _make_capturing_service(wants_wav=False)
    received_down, _ = await run_test(
        service,
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )
    usage = [
        d
        for f in received_down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, STTUsageMetricsData)
    ]
    return service.captured, usage


@pytest.mark.asyncio
async def test_a_segment_spoken_while_muted_is_not_transcribed():
    captured, usage = await _run_with_usage(
        [
            STTMuteFrame(mute=True),
            VADUserStartedSpeakingFrame(),
            _audio(MUTED),
            VADUserStoppedSpeakingFrame(),
        ]
    )
    assert captured == []
    assert usage == []


@pytest.mark.asyncio
async def test_audio_that_arrived_while_muted_is_cut_from_the_segment():
    captured, _ = await _run_with_usage(
        [
            VADUserStartedSpeakingFrame(),
            _audio(BEFORE),
            STTMuteFrame(mute=True),
            _audio(MUTED),
            STTMuteFrame(mute=False),
            _audio(AFTER),
            VADUserStoppedSpeakingFrame(),
        ]
    )
    # Speech already in progress when the mute starts keeps its pre-mute part.
    assert captured == [BEFORE + AFTER]


@pytest.mark.asyncio
async def test_an_unmuted_segment_is_transcribed_whole():
    captured, _ = await _run_with_usage(
        [
            VADUserStartedSpeakingFrame(),
            _audio(BEFORE),
            _audio(AFTER),
            VADUserStoppedSpeakingFrame(),
        ]
    )
    assert captured == [BEFORE + AFTER]


@pytest.mark.asyncio
async def test_a_turn_inside_a_mute_that_starts_after_caller_audio_is_not_transcribed():
    # A mute that starts on the bot's first words, as an application's first-message
    # window does, finds the pre-roll already holding the caller's audio. A turn
    # spoken wholly inside the window must not flush that pre-roll as a segment.
    captured, usage = await _run_with_usage(
        [
            _audio(PREROLL),
            STTMuteFrame(mute=True),
            VADUserStartedSpeakingFrame(),
            _audio(MUTED),
            VADUserStoppedSpeakingFrame(),
            STTMuteFrame(mute=False),
        ]
    )
    assert captured == []
    assert usage == []
