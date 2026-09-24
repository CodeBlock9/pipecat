#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pytest

from pipecat.serializers.call_strategies import CARRIER_REQUEST_TIMEOUT_SECS
from pipecat.serializers.vonage import VonageFrameSerializer


class _FakeResponse:
    status = 204

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def text(self):
        raise AssertionError("204 Vonage hangup responses should not be treated as errors")


class _FakeClientSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.put_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    def put(self, endpoint, *, headers, json):
        self.put_calls.append((endpoint, headers, json))
        return _FakeResponse()


async def _hang_up() -> _FakeClientSession:
    """Hang up through a stand-in aiohttp, returning the one session it built."""
    sessions = []

    def client_session(**kwargs):
        sessions.append(_FakeClientSession(**kwargs))
        return sessions[-1]

    fake_aiohttp = SimpleNamespace(
        ClientSession=client_session, ClientTimeout=aiohttp.ClientTimeout
    )
    fake_jwt = SimpleNamespace(encode=lambda claims, private_key, algorithm: "token")

    serializer = VonageFrameSerializer(
        call_uuid="call-123",
        application_id="app-123",
        private_key="private-key",
    )

    with patch.dict(sys.modules, {"aiohttp": fake_aiohttp, "jwt": fake_jwt}):
        await serializer._hang_up_call()

    assert len(sessions) == 1
    return sessions[0]


@pytest.mark.asyncio
async def test_vonage_hangup_treats_204_as_success():
    session = await _hang_up()

    assert session.put_calls == [
        (
            "https://api.nexmo.com/v1/calls/call-123",
            {"Authorization": "Bearer token", "Content-Type": "application/json"},
            {"action": "hangup"},
        )
    ]


@pytest.mark.asyncio
async def test_vonage_hangup_is_bounded():
    """The hangup runs inside the EndFrame's traversal, so a carrier that stops
    answering would otherwise hold the call's end for aiohttp's 300 s."""
    session = await _hang_up()

    timeout = session.kwargs.get("timeout")
    assert timeout is not None and timeout.total == CARRIER_REQUEST_TIMEOUT_SECS, (
        f"the hangup session is built with timeout={timeout!r}"
    )
