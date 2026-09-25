#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime: null usage and unparseable server events end neither a turn nor the session.

Every fixture is wire JSON. The receive loop reads it from a fake websocket and
parses it with the fork's own ``events.parse_server_event``, as it does in a
call; the first test pins that decoding.
"""

import io
import json
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from pipecat.frames.frames import LLMFullResponseEndFrame, TTSStoppedFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

CANCELLED_DONE_NULL_USAGE = json.dumps(
    {
        "event_id": "event_rec_1",
        "type": "response.done",
        "response": {
            "id": "resp_rec_1",
            "object": "realtime.response",
            "status": "cancelled",
            "status_details": {"type": "cancelled", "reason": "turn_detected"},
            "output": [],
            "usage": None,
        },
    }
)

SPEECH_STARTED = json.dumps(
    {
        "event_id": "event_rec_2",
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 1000,
        "item_id": "item_rec_2",
    }
)

# A GA server event the fork has no model for.
UNMODELLED_EVENT = json.dumps(
    {
        "event_id": "event_rec_3",
        "type": "input_audio_buffer.timeout_triggered",
        "audio_start_ms": 1000,
        "audio_end_ms": 6000,
        "item_id": "item_rec_3",
    }
)

# A modelled event that fails validation (no item_id), carrying what the
# caller said. The parse error repeats the whole event after its first line.
PRIVATE_WORDS = "my card number is 4111 1111 1111 1111"
MALFORMED_TRANSCRIPT = json.dumps(
    {
        "event_id": "event_rec_4",
        "type": "conversation.item.input_audio_transcription.completed",
        "content_index": 0,
        "transcript": PRIVATE_WORDS,
    }
)

FATAL_ERROR = json.dumps(
    {
        "event_id": "event_rec_5",
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "code": "session_expired",
            "message": "Your session hit the maximum duration of 60 minutes.",
            "param": None,
            "event_id": None,
        },
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


def _service(messages) -> OpenAIRealtimeLLMService:
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(model="gpt-realtime"),
    )
    service.start_llm_usage_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service._call_event_handler = AsyncMock()
    service._handle_evt_speech_started = AsyncMock()
    # An audio reply is playing, as it is when a barge-in cancels it.
    service._current_audio_response = object()
    service._websocket = _FakeWebsocket(messages)
    return service


async def _receive_with_warnings(service) -> list[str]:
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._receive_task_handler()
    finally:
        logger.remove(handler_id)
    return sink.getvalue().splitlines()


def test_the_fixtures_are_the_forks_own_decoding():
    done = events.parse_server_event(CANCELLED_DONE_NULL_USAGE)
    assert isinstance(done, events.ResponseDone)
    assert done.response.status == "cancelled"
    assert done.response.usage is None
    assert isinstance(
        events.parse_server_event(SPEECH_STARTED), events.InputAudioBufferSpeechStarted
    )
    with pytest.raises(Exception, match="Unimplemented server event type"):
        events.parse_server_event(UNMODELLED_EVENT)
    with pytest.raises(Exception, match="validation error") as raised:
        events.parse_server_event(MALFORMED_TRANSCRIPT)
    assert PRIVATE_WORDS in str(raised.value)


@pytest.mark.asyncio
async def test_the_receive_loop_survives_a_null_usage_response_done():
    service = _service([CANCELLED_DONE_NULL_USAGE, SPEECH_STARTED])

    await service._receive_task_handler()

    assert [type(call.args[0]) for call in service.push_frame.await_args_list] == [
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    service.start_llm_usage_metrics.assert_not_called()
    service._handle_evt_speech_started.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_receive_loop_skips_an_unmodelled_server_event():
    service = _service([UNMODELLED_EVENT, SPEECH_STARTED])

    warnings = await _receive_with_warnings(service)

    service._handle_evt_speech_started.assert_awaited_once()
    assert len(warnings) == 1
    assert (
        "Failed to parse server event of type input_audio_buffer.timeout_triggered" in (warnings[0])
    )


@pytest.mark.asyncio
async def test_a_skipped_event_is_logged_by_type_and_first_line_only():
    service = _service([MALFORMED_TRANSCRIPT, SPEECH_STARTED])

    warnings = await _receive_with_warnings(service)

    service._handle_evt_speech_started.assert_awaited_once()
    assert len(warnings) == 1
    assert (
        "Failed to parse server event of type "
        "conversation.item.input_audio_transcription.completed: 1 validation error"
    ) in warnings[0]
    assert "4111" not in warnings[0]


def _completed_done(usage) -> str:
    return json.dumps(
        {
            "event_id": "event_rec_6",
            "type": "response.done",
            "response": {
                "id": "resp_rec_6",
                "object": "realtime.response",
                "status": "completed",
                "status_details": None,
                "output": [],
                "usage": usage,
            },
        }
    )


# Usage the model cannot read: an empty object, and one without its detail objects.
UNREADABLE_USAGE = {
    "empty": {},
    "no_details": {"total_tokens": 3, "input_tokens": 2, "output_tokens": 1},
}


@pytest.mark.parametrize("usage", UNREADABLE_USAGE.values(), ids=UNREADABLE_USAGE.keys())
def test_unreadable_usage_is_dropped_and_the_response_done_parses(usage):
    done = events.parse_server_event(_completed_done(usage))

    assert isinstance(done, events.ResponseDone)
    assert done.response.status == "completed"
    assert done.response.usage is None


@pytest.mark.parametrize("usage", UNREADABLE_USAGE.values(), ids=UNREADABLE_USAGE.keys())
@pytest.mark.asyncio
async def test_a_response_done_with_unreadable_usage_still_closes_the_turn(usage):
    service = _service([_completed_done(usage), SPEECH_STARTED])

    warnings = await _receive_with_warnings(service)

    assert warnings[0].startswith("Realtime response.done usage unreadable, dropped: "), warnings
    assert "validation error" in warnings[0]
    assert not any("Failed to parse server event" in w for w in warnings)
    service.start_llm_usage_metrics.assert_not_called()
    service.stop_processing_metrics.assert_awaited_once()
    assert [type(call.args[0]) for call in service.push_frame.await_args_list] == [
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    service._handle_evt_speech_started.assert_awaited_once()


@pytest.mark.asyncio
async def test_control_a_fatal_error_event_still_ends_the_loop():
    service = _service([FATAL_ERROR, SPEECH_STARTED])

    await service._receive_task_handler()

    service.push_error.assert_awaited_once()
    service._handle_evt_speech_started.assert_not_called()
