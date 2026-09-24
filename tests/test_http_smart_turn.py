#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HttpSmartTurnAnalyzer sends its request from the model thread.

BaseSmartTurn runs ``_predict_endpoint`` on its model thread, which has no
running event loop, so the analyzer must submit the request to the loop that
owns its aiohttp session. Only the session is a stand-in here.
"""

import numpy as np
import pytest

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer

RATE = 16_000


class _Response:
    def __init__(self, status: int, payload: dict | None = None):
        self.status = status
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def text(self):
        return "server error"

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status}")


class _Post:
    def __init__(self, outcome):
        self._outcome = outcome

    async def __aenter__(self):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, outcome):
        self._outcome = outcome
        self.posts = 0

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts += 1
        return _Post(self._outcome)


def _analyzer(session: _Session) -> HttpSmartTurnAnalyzer:
    analyzer = HttpSmartTurnAnalyzer(
        url="http://smart-turn.invalid/predict",
        aiohttp_session=session,
        sample_rate=RATE,
        params=SmartTurnParams(stop_secs=1.0),
    )
    analyzer.set_sample_rate(RATE)
    for _ in range(10):
        analyzer.append_audio(np.full(320, 1000, dtype=np.int16).tobytes(), True)
    return analyzer


@pytest.mark.asyncio
async def test_complete_prediction_is_requested_and_used():
    session = _Session(_Response(200, {"prediction": 1, "probability": 0.93}))
    analyzer = _analyzer(session)

    state, metrics = await analyzer.analyze_end_of_turn()

    assert session.posts == 1, "the prediction request was never sent"
    assert state == EndOfTurnState.COMPLETE
    assert metrics is not None and metrics.probability == pytest.approx(0.93)
    await analyzer.cleanup()


@pytest.mark.asyncio
async def test_timed_out_request_ends_the_turn():
    """BaseSmartTurn's own contract: a SmartTurnTimeoutException completes the turn."""
    session = _Session(TimeoutError())
    analyzer = _analyzer(session)

    state, _ = await analyzer.analyze_end_of_turn()

    assert session.posts == 1
    assert state == EndOfTurnState.COMPLETE
    await analyzer.cleanup()


@pytest.mark.asyncio
async def test_server_error_is_incomplete_not_a_crash():
    session = _Session(_Response(500))
    analyzer = _analyzer(session)

    state, _ = await analyzer.analyze_end_of_turn()

    assert state == EndOfTurnState.INCOMPLETE
    await analyzer.cleanup()
