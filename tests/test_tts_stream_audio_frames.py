#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TTSService._stream_audio_frames_from_iterator and the WAV header.

The real helper on a minimal TTSService (built by a factory so
test_service_init.py's discovery scan does not see it), fed WAV files written by
Python's wave module. A network delivers the header however it likes: split
anywhere, a byte at a time, or longer than 44 bytes because a chunk such as
``LIST`` comes before ``data``. None of it may be played, and every frame,
the tail frame included, carries the context id.
"""

import io
import wave

import pytest
from loguru import logger

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

PCM = bytes(range(1, 241)) * 2  # 480 bytes, distinct non-zero samples

# A LIST chunk as ffmpeg and sox write it, and a chunk with an odd size and its pad byte.
LIST = (
    b"LIST"
    + (26).to_bytes(4, "little")
    + b"INFOISFT"
    + (14).to_bytes(4, "little")
    + b"Lavf61.7.100\x00\x00"
)
ODD = b"junk" + (3).to_bytes(4, "little") + b"abc" + b"\x00"


def _wav(rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(PCM)
    return buffer.getvalue()


def _with_chunk_before_data(chunk: bytes, rate: int = 24000) -> bytes:
    """The same WAV with `chunk` between `fmt ` and `data`."""
    wav = _wav(rate)
    body = wav[12:36] + chunk + wav[36:]
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WAVE" + body


class _RecordingResampler:
    """Records the source rate of every resample and returns the audio unchanged."""

    def __init__(self):
        self.rates: list[int] = []

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        self.rates.append(in_rate)
        return audio


async def _service() -> TTSService:
    class _Minimal(TTSService):
        async def run_tts(self, text: str, context_id: str):
            yield None

    service = _Minimal(sample_rate=24000)
    await service.setup(frame_processor_setup(TaskManager()))
    return service


async def _stream(
    parts: list[bytes], service: TTSService | None = None, in_sample_rate: int | None = None
) -> bytes:
    service = service or await _service()

    async def chunks():
        for part in parts:
            yield part

    frames = [
        f
        async for f in service._stream_audio_frames_from_iterator(
            chunks(), strip_wav_header=True, in_sample_rate=in_sample_rate, context_id="ctx"
        )
    ]
    assert all(isinstance(f, TTSAudioRawFrame) for f in frames)
    assert all(f.context_id == "ctx" for f in frames), "a frame without the context id"
    return b"".join(f.audio for f in frames)


async def _stream_and_warnings(parts: list[bytes]) -> tuple[bytes, list[str]]:
    warnings: list[str] = []
    sink_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        audio = await _stream(parts)
    finally:
        logger.remove(sink_id)
    return audio, warnings


def _splits(data: bytes, cuts: list[int]) -> list[bytes]:
    cuts = [0, *cuts, len(data)]
    return [data[a:b] for a, b in zip(cuts, cuts[1:])]


def test_the_fixture_header_is_44_bytes():
    assert len(_wav()) == 44 + len(PCM)


def test_the_list_fixture_header_is_not_44_bytes():
    wav = _with_chunk_before_data(LIST)
    assert len(wav) - len(PCM) == 44 + 8 + 26
    assert wav[-len(PCM) :] == PCM


@pytest.mark.asyncio
async def test_a_whole_header_in_the_first_chunk_is_stripped():
    assert await _stream([_wav()]) == PCM


@pytest.mark.asyncio
async def test_a_header_split_after_20_bytes_is_stripped():
    wav = _wav()
    assert await _stream([wav[:20], wav[20:]]) == PCM


@pytest.mark.asyncio
async def test_a_header_arriving_byte_by_byte_is_stripped():
    wav = _wav()
    assert await _stream([wav[i : i + 1] for i in range(50)] + [wav[50:]]) == PCM


@pytest.mark.asyncio
async def test_a_header_longer_than_44_bytes_is_stripped():
    assert await _stream([_with_chunk_before_data(LIST)]) == PCM


@pytest.mark.asyncio
async def test_every_two_way_split_of_44_56_and_78_byte_headers():
    service = await _service()
    for wav in (_wav(), _with_chunk_before_data(ODD), _with_chunk_before_data(LIST)):
        header = len(wav) - len(PCM)
        for cut in range(1, header + 12):
            assert await _stream(_splits(wav, [cut]), service) == PCM, (header, cut)


@pytest.mark.asyncio
async def test_every_three_way_split_inside_the_44_byte_header():
    service = await _service()
    wav = _wav()
    for a in range(1, 44):
        for b in range(a + 1, 48):
            assert await _stream(_splits(wav, [a, b]), service) == PCM, (a, b)


@pytest.mark.asyncio
async def test_the_rate_is_read_from_a_split_header():
    for cut in range(1, 44):
        service = await _service()
        service._resampler = resampler = _RecordingResampler()
        assert await _stream(_splits(_wav(16000), [cut]), service) == PCM, cut
        assert set(resampler.rates) == {16000}, (cut, resampler.rates)


@pytest.mark.asyncio
async def test_a_given_in_sample_rate_wins_over_the_header_rate():
    service = await _service()
    service._resampler = resampler = _RecordingResampler()
    assert await _stream([_wav(16000)], service, in_sample_rate=22050) == PCM
    assert set(resampler.rates) == {22050}


@pytest.mark.asyncio
async def test_a_provider_that_sends_no_header():
    service = await _service()
    warnings: list[str] = []
    sink_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        for total in list(range(1, 30)) + [480]:
            raw = bytes((i % 250) + 1 for i in range(total))
            padded = raw + (b"\x00" if total % 2 else b"")
            assert await _stream([raw], service) == padded, total
            assert await _stream([raw[i : i + 1] for i in range(total)], service) == padded, total
    finally:
        logger.remove(sink_id)
    assert warnings == []


@pytest.mark.asyncio
async def test_a_raw_stream_whose_later_chunk_begins_with_riff_is_played_whole():
    raw = b"\x10\x00" * 20
    later = b"RIFF" + b"\x20\x00" * 30
    assert await _stream([raw, later]) == raw + later


@pytest.mark.asyncio
async def test_a_streaming_header_with_unknown_sizes():
    wav = bytearray(_wav())
    wav[4:8] = (0xFFFFFFFF).to_bytes(4, "little")  # RIFF size
    wav[40:44] = (0xFFFFFFFF).to_bytes(4, "little")  # data size
    assert await _stream(_splits(bytes(wav), [7, 30])) == PCM


@pytest.mark.asyncio
async def test_a_stream_that_ends_inside_a_riff_header_plays_nothing_and_warns():
    audio, warnings = await _stream_and_warnings([_wav()[:30]])
    assert audio == b""
    assert len(warnings) == 1 and "WAV header" in warnings[0], warnings


@pytest.mark.asyncio
async def test_a_riff_header_whose_data_never_arrives_plays_nothing_and_warns():
    # A malformed chunk size: the walk jumps past the end and waits for more.
    bad = bytearray(_with_chunk_before_data(LIST))
    bad[40:44] = (0x7FFFFFF0).to_bytes(4, "little")  # LIST size
    audio, warnings = await _stream_and_warnings([bytes(bad)] + [PCM] * 50)
    assert audio == b""
    assert len(warnings) == 1 and "WAV header" in warnings[0], warnings
