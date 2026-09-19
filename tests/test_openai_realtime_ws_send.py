#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for realtime websocket send failures.

A send-side failure retires the socket and reports the failure as fatal, so a
session whose socket has gone reports once rather than once per frame the
transport keeps feeding it. The retired socket is closed by ``_disconnect``.
"""

import pytest

from pipecat.frames.frames import ErrorFrame
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class _FakeWebsocket:
    """A websocket whose ``send`` fails the way a dropped connection does."""

    def __init__(self, *, send_error: Exception | None = None):
        self._send_error = send_error
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str):
        if self._send_error:
            raise self._send_error
        self.sent.append(message)

    async def close(self):
        self.closed = True


class _ErrorRecorder:
    def __init__(self):
        self.frames: list[ErrorFrame] = []
        self.permanent: list[bool] = []

    async def __call__(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        self.frames.append(error)
        self.permanent.append(force_treat_as_permanent)


def _service(**kwargs) -> OpenAIRealtimeLLMService:
    return OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-2",
            system_instruction="be helpful",
        ),
        **kwargs,
    )


def _attach_recorder(service) -> _ErrorRecorder:
    recorder = _ErrorRecorder()
    service.push_error_frame = recorder
    return recorder


@pytest.mark.asyncio
async def test_send_failure_is_reported_once_as_fatal():
    service = _service()
    websocket = _FakeWebsocket(send_error=ConnectionError("keepalive ping timeout"))
    service._websocket = websocket
    errors = _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))

    assert len(errors.frames) == 1
    assert errors.permanent == [True]
    assert "keepalive ping timeout" in errors.frames[0].error
    assert service._websocket is None
    assert service._dead_websocket is websocket
    # Parked, not closed: closing goes through _disconnect, which cancels the
    # receive task — and the receive task can be the caller.
    assert websocket.closed is False


@pytest.mark.asyncio
async def test_later_sends_on_a_retired_socket_report_nothing():
    service = _service()
    websocket = _FakeWebsocket(send_error=ConnectionError("keepalive ping timeout"))
    service._websocket = websocket
    errors = _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))
    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="BBBB"))
    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="CCCC"))

    assert len(errors.frames) == 1


@pytest.mark.asyncio
async def test_disconnect_closes_the_retired_socket():
    service = _service()
    websocket = _FakeWebsocket(send_error=ConnectionError("keepalive ping timeout"))
    service._websocket = websocket
    _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))
    await service._disconnect()

    assert websocket.closed is True
    assert service._dead_websocket is None


@pytest.mark.asyncio
async def test_disconnect_survives_a_retired_socket_that_will_not_close():
    class _UncloseableWebsocket(_FakeWebsocket):
        async def close(self):
            raise ConnectionError("socket already gone")

    service = _service()
    service._websocket = _UncloseableWebsocket(send_error=ConnectionError("gone"))
    errors = _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))
    await service._disconnect()

    # Only the send failure is reported; a close failure on an already-dead
    # socket says nothing new.
    assert len(errors.frames) == 1
    assert service._dead_websocket is None
    assert service._disconnecting is False


@pytest.mark.asyncio
async def test_a_healthy_socket_is_left_alone():
    service = _service()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    errors = _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))

    assert errors.frames == []
    assert service._websocket is websocket
    assert service._dead_websocket is None
    assert len(websocket.sent) == 1


@pytest.mark.asyncio
async def test_send_failure_while_disconnecting_reports_nothing():
    service = _service()
    service._websocket = _FakeWebsocket(send_error=ConnectionError("gone"))
    service._disconnecting = True
    errors = _attach_recorder(service)

    await service.send_client_event(events.InputAudioBufferAppendEvent(audio="AAAA"))

    assert errors.frames == []


@pytest.mark.asyncio
async def test_connect_failure_is_fatal():
    service = _service()
    errors = _attach_recorder(service)

    async def _fail(*args, **kwargs):
        raise ConnectionError("handshake refused")

    import pipecat.services.openai.realtime.llm as realtime_llm

    original = realtime_llm.websocket_connect
    realtime_llm.websocket_connect = _fail
    try:
        await service._connect()
    finally:
        realtime_llm.websocket_connect = original

    assert len(errors.frames) == 1
    assert errors.permanent == [True]
    assert service._websocket is None


@pytest.mark.asyncio
async def test_azure_connect_failure_is_fatal():
    """The Azure subclass connects with its own headers but the same verdict."""
    service = AzureRealtimeLLMService(
        api_key="test-key",
        base_url="wss://example.openai.azure.com/openai/realtime",
        settings=AzureRealtimeLLMService.Settings(
            model="gpt-realtime",
            system_instruction="be helpful",
        ),
    )
    errors = _attach_recorder(service)

    async def _fail(*args, **kwargs):
        raise ConnectionError("handshake refused")

    import pipecat.services.azure.realtime.llm as azure_llm

    original = azure_llm.websocket_connect
    azure_llm.websocket_connect = _fail
    try:
        await service._connect()
    finally:
        azure_llm.websocket_connect = original

    assert len(errors.frames) == 1
    assert errors.permanent == [True]
    assert service._websocket is None
