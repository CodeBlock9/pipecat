#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Deepgram TTS error handling and usage."""

import json
from unittest.mock import AsyncMock

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

TEXT = "Your table for four is booked for seven."


def _websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


@pytest.mark.asyncio
async def test_deepgram_rejected_api_key_makes_the_service_unusable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = DeepgramTTSService(api_key="wrong-key", sample_rate=24000)

    await service._connect_websocket()

    assert not service.is_usable


@pytest.mark.asyncio
async def test_deepgram_server_error_leaves_the_service_usable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(503)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = DeepgramTTSService(api_key="test-key", sample_rate=24000)

    await service._connect_websocket()

    assert service.is_usable


class _Socket:
    """An open socket that records what is sent, or fails every send."""

    def __init__(self, fail: bool = False):
        self.state = State.OPEN
        self.sent: list[str] = []
        self._fail = fail

    async def send(self, message: str):
        if self._fail:
            raise ConnectionError("socket closed under us")
        self.sent.append(message)


async def _service_with_socket(socket: _Socket) -> tuple[DeepgramTTSService, list]:
    # An Aura voice, usage metrics on, as an application's pipeline sets them.
    service = DeepgramTTSService(
        api_key="dg-key",
        sample_rate=24000,
        settings=DeepgramTTSSettings(voice="aura-2-helena-en", extra={}),
    )
    await service.setup(frame_processor_setup(TaskManager(), enable_usage_metrics=True))
    service._websocket = socket
    pushed: list = []
    service.push_frame = AsyncMock(side_effect=lambda f, *a, **k: pushed.append(f))
    return service, pushed


def _usage(pushed) -> list[TTSUsageMetricsData]:
    return [
        d
        for f in pushed
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TTSUsageMetricsData)
    ]


@pytest.mark.asyncio
async def test_a_spoken_sentence_reports_its_characters():
    socket = _Socket()
    service, pushed = await _service_with_socket(socket)

    [item async for item in service.run_tts(TEXT, "ctx")]

    assert [json.loads(m) for m in socket.sent] == [{"type": "Speak", "text": TEXT}]
    assert [u.value for u in _usage(pushed)] == [len(TEXT)]
    assert _usage(pushed)[0].processor.startswith("DeepgramTTSService#")


@pytest.mark.asyncio
async def test_a_failed_send_reports_no_usage():
    service, pushed = await _service_with_socket(_Socket(fail=True))

    [item async for item in service.run_tts(TEXT, "ctx")]

    assert _usage(pushed) == []
