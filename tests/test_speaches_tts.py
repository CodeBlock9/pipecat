#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SpeachesTTSService."""

import asyncio

import httpx
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.speaches.tts import SpeachesTTSService, SpeachesTTSSettings
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

CHUNK = bytes(range(0, 240)) * 4  # 960 bytes, 20 ms at 24 kHz, distinct samples
CHUNKS = 25


@pytest.mark.asyncio
async def test_run_speaches_tts_allows_custom_voice(aiohttp_client):
    """Speaches should pass custom voice IDs through unchanged."""

    request_bodies = []

    async def handler(request):
        request_bodies.append(await request.json())

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/pcm"},
        )
        await response.prepare(request)
        await response.write(b"\x00\x01\x02\x03" * 1024)
        await asyncio.sleep(0.01)
        await response.write(b"\x04\x05\x06\x07" * 1024)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1"))

    tts_service = SpeachesTTSService(
        api_key="none",
        base_url=base_url,
        sample_rate=24000,
        settings=SpeachesTTSSettings(
            model="speaches-ai/piper-tr_TR-fettah-medium",
            voice="fettah",
        ),
    )

    down_frames, _ = await run_test(
        tts_service,
        frames_to_send=[TTSSpeakFrame(text="Merhaba dunya.")],
    )

    frame_types = [type(frame) for frame in down_frames]
    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)

    assert len(request_bodies) == 1
    assert request_bodies[0] == {
        "input": "Merhaba dunya.",
        "model": "speaches-ai/piper-tr_TR-fettah-medium",
        "voice": "fettah",
        "response_format": "pcm",
    }


def _streaming_client(sent: list[int]) -> httpx.AsyncClient:
    """A client whose server streams CHUNKS network chunks, recording each one sent."""

    async def body():
        for i in range(CHUNKS):
            sent.append(i)
            yield CHUNK

    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    )


async def _streaming_service(sent: list[int]) -> SpeachesTTSService:
    service = SpeachesTTSService(
        base_url="http://speaches.invalid/v1",
        sample_rate=24000,
        http_client=_streaming_client(sent),
        settings=SpeachesTTSSettings(model="speaches-model", voice="custom-voice"),
    )
    await service.setup(frame_processor_setup(TaskManager()))  # sets the output sample rate
    return service


@pytest.mark.asyncio
async def test_the_first_frame_follows_the_first_network_chunk():
    sent: list[int] = []
    service = await _streaming_service(sent)
    frames = service.run_tts("Hello there.", "ctx")

    first = await anext(frames)
    chunks_sent_before_first_frame = len(sent)
    await frames.aclose()

    assert isinstance(first, TTSAudioRawFrame)
    assert chunks_sent_before_first_frame <= 2, (
        f"{chunks_sent_before_first_frame} x 20 ms chunks arrived before the first frame"
    )


@pytest.mark.asyncio
async def test_every_byte_is_yielded_in_order():
    sent: list[int] = []
    service = await _streaming_service(sent)

    frames = [f async for f in service.run_tts("Hello there.", "ctx")]

    assert b"".join(f.audio for f in frames) == CHUNK * CHUNKS
    assert all(f.context_id == "ctx" for f in frames)
