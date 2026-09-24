#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Contract tests for Dograh-owned telephony serializers."""

from types import SimpleNamespace

import aiohttp
import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import OutputTransportMessageFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.asterisk import AsteriskFrameSerializer
from pipecat.serializers.call_strategies import CARRIER_REQUEST_TIMEOUT_SECS
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.utils.asyncio.task_manager import TaskManager


def _setup(*, sample_rate: int = 16000) -> FrameProcessorSetup:
    return FrameProcessorSetup(
        audio_in_sample_rate=sample_rate,
        clock=SystemClock(),
        task_manager=TaskManager(),
        pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_vobiz_uses_v18_setup_and_filters_rtvi_messages():
    serializer = VobizFrameSerializer(
        stream_id="stream",
        params=VobizFrameSerializer.InputParams(auto_hang_up=False),
    )

    await serializer.setup(_setup())

    assert serializer._sample_rate == 16000
    assert (
        await serializer.serialize(
            OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "test"})
        )
        is None
    )


@pytest.mark.asyncio
async def test_asterisk_uses_v18_setup_contract():
    serializer = AsteriskFrameSerializer(
        channel_id="channel",
        ari_endpoint="http://asterisk.invalid",
        app_name="app",
        app_password="secret",
        params=AsteriskFrameSerializer.InputParams(sample_rate=24000),
    )

    await serializer.setup(_setup(sample_rate=16000))

    assert serializer._sample_rate == 24000


class _NoContent:
    status = 204

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingSession:
    """An aiohttp session that records how it was built and answers with 204."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.deleted = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def delete(self, url, **kwargs):
        self.deleted.append(url)
        return _NoContent()


@pytest.mark.asyncio
async def test_vobiz_hangup_is_bounded(monkeypatch):
    """The hangup runs inside the EndFrame's traversal, so a carrier that stops
    answering would otherwise hold the call's end for aiohttp's 300 s."""
    sessions = []

    def client_session(**kwargs):
        sessions.append(_RecordingSession(**kwargs))
        return sessions[-1]

    monkeypatch.setattr(aiohttp, "ClientSession", client_session)
    serializer = VobizFrameSerializer(
        stream_id="stream", call_id="call", auth_id="MA0123456789", auth_token="token"
    )

    await serializer._hang_up_call()

    assert [session.deleted for session in sessions] == [
        ["https://api.vobiz.ai/api/v1/Account/MA0123456789/Call/call/"]
    ]
    timeout = sessions[0].kwargs.get("timeout")
    assert timeout is not None and timeout.total == CARRIER_REQUEST_TIMEOUT_SECS, (
        f"the hangup session is built with timeout={timeout!r}"
    )
