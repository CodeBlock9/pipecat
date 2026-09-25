#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A carrier hangup cut by a cancel is retried by the CancelFrame that follows.

The worker's post-drain preempt can cancel ``serialize(EndFrame())`` while the
hangup request is in flight. The serializer must then forget that it tried, or
the CancelFrame that follows skips the hangup and the caller stays on the line.
"""

import asyncio

import pytest

from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.serializers.vonage import VonageFrameSerializer


class _Strategy:
    """Counts hangup requests; the first one can be made to hang until cancelled."""

    def __init__(self, *, block_first: bool):
        self.calls = 0
        self._block_first = block_first
        self._release = asyncio.Event()

    async def execute_hangup(self, context):
        self.calls += 1
        if self.calls == 1 and self._block_first:
            await self._release.wait()
        return True


def _twilio(strategy):
    return TwilioFrameSerializer(
        "MZ1", call_sid="CA1", account_sid="AC1", auth_token="t", hangup_strategy=strategy
    )


def _telnyx(strategy):
    return TelnyxFrameSerializer(
        "st", "PCMU", "PCMU", call_control_id="cc", api_key="k", hangup_strategy=strategy
    )


def _plivo(strategy):
    return PlivoFrameSerializer(
        "st", call_id="c1", auth_id="a", auth_token="t", hangup_strategy=strategy
    )


def _vobiz(strategy):
    serializer = VobizFrameSerializer("st", call_id="c1", auth_id="a", auth_token="t")
    serializer._hang_up_call = lambda: strategy.execute_hangup({})
    return serializer


def _vonage(strategy):
    serializer = VonageFrameSerializer("uuid", application_id="app", private_key="pk")
    serializer._hang_up_call = lambda: strategy.execute_hangup({})
    return serializer


BUILDERS = [_twilio, _telnyx, _plivo, _vobiz, _vonage]


@pytest.mark.parametrize("build", BUILDERS, ids=lambda b: b.__name__[1:])
@pytest.mark.asyncio
async def test_a_cancelled_hangup_is_retried_by_the_cancel_frame(build):
    strategy = _Strategy(block_first=True)
    serializer = build(strategy)
    ending = asyncio.create_task(serializer.serialize(EndFrame()))
    await asyncio.sleep(0.05)
    assert strategy.calls == 1

    ending.cancel()  # the post-drain preempt cuts the request mid-flight
    with pytest.raises(asyncio.CancelledError):
        await ending

    await serializer.serialize(CancelFrame())
    assert strategy.calls == 2, "the CancelFrame after a cut hangup must hang up again"


@pytest.mark.parametrize("build", BUILDERS, ids=lambda b: b.__name__[1:])
@pytest.mark.asyncio
async def test_control_a_finished_hangup_is_not_repeated(build):
    strategy = _Strategy(block_first=False)
    serializer = build(strategy)

    await serializer.serialize(EndFrame())
    await serializer.serialize(CancelFrame())

    assert strategy.calls == 1
