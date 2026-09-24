#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how OpenAITTSService streams a synthesis.

The real service, with the real OpenAI SDK, reads a response that an
``httpx.MockTransport`` streams as 25 network chunks of 20 ms of 24 kHz PCM. The
tests record how many chunks the server had sent when ``run_tts`` yielded its
first frame, so a service that waits for a fixed amount of audio before it yields
anything is visible.
"""

import httpx
import pytest

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

CHUNK = bytes(range(0, 240)) * 4  # 960 bytes, 20 ms at 24 kHz, distinct samples
CHUNKS = 25


def _client(sent: list[int]) -> httpx.AsyncClient:
    async def body():
        for i in range(CHUNKS):
            sent.append(i)
            yield CHUNK

    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    )


async def _service(sent: list[int]) -> OpenAITTSService:
    service = OpenAITTSService(api_key="test-key", sample_rate=24000, http_client=_client(sent))
    await service.setup(frame_processor_setup(TaskManager()))  # sets the output sample rate
    return service


@pytest.mark.asyncio
async def test_the_first_frame_follows_the_first_network_chunk():
    sent: list[int] = []
    service = await _service(sent)
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
    service = await _service(sent)

    frames = [f async for f in service.run_tts("Hello there.", "ctx")]

    assert b"".join(f.audio for f in frames) == CHUNK * CHUNKS
    assert all(f.context_id == "ctx" for f in frames)
