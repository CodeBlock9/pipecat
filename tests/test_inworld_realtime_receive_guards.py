#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld Realtime: a server event that fails to parse is logged by type and first line only.

The fixture is wire JSON, read by the receive loop from a fake websocket and
parsed with the fork's own ``events.parse_server_event``, as it is in a call.
"""

import io
import json
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from pipecat.services.inworld.realtime import events
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService

# A recognised event that fails validation (no item_id), carrying what the
# caller said. The parse error repeats the whole event after its first line.
PRIVATE_WORDS = "my card number is 4111 1111 1111 1111"
MALFORMED_TRANSCRIPT = json.dumps(
    {
        "event_id": "event_rec_4",
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": PRIVATE_WORDS,
    }
)

SPEECH_STARTED = json.dumps(
    {
        "event_id": "event_rec_5",
        "type": "input_audio_buffer.speech_started",
        "item_id": "item_rec_5",
    }
)


class _FakeWebsocket:
    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for message in self._messages:
            yield message


@pytest.mark.asyncio
async def test_a_skipped_event_is_logged_by_type_and_first_line_only():
    with pytest.raises(Exception, match="validation error") as raised:
        events.parse_server_event(MALFORMED_TRANSCRIPT)
    assert PRIVATE_WORDS in str(raised.value)

    service = InworldRealtimeLLMService(api_key="test-key")
    service._handle_evt_speech_started = AsyncMock()
    service._websocket = _FakeWebsocket([MALFORMED_TRANSCRIPT, SPEECH_STARTED])
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._receive_task_handler()
    finally:
        logger.remove(handler_id)
    warnings = sink.getvalue().splitlines()

    service._handle_evt_speech_started.assert_awaited_once()
    assert len(warnings) == 1
    assert (
        "Failed to parse server event of type "
        "conversation.item.input_audio_transcription.completed: 1 validation error"
    ) in warnings[0]
    assert "4111" not in warnings[0]
