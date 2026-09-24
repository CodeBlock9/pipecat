#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The Responses services' terminal policy for function calls.

Both transports keep only the calls whose arguments finished streaming when a
response ends in ``response.failed``, ``response.incomplete`` or ``error``;
the WebSocket reports the Responses API's own failure fields and coalesces a
null usage count; and neither runs a call whose arguments do not parse.

The WebSocket fixtures are Responses API wire JSON fed through the service's
own ``_ws_recv``. ``test_the_recorded_events_decode_as_sdk_stream_events``
checks that each one decodes as the SDK's ``ResponseStreamEvent`` union.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai._models import construct_type
from openai.types.responses import ResponseStreamEvent

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService


def _make_service():
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(api_key="test-key")
    service._client = AsyncMock()
    service._push_llm_text = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_ttfat_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    service.push_error = AsyncMock()
    service.run_function_calls = AsyncMock()
    service._store_previous_response_state = MagicMock()
    return service


def _ws(*events):
    ws = AsyncMock()
    ws.recv = AsyncMock(side_effect=[json.dumps(e) for e in events])
    ws.close_code = None
    return ws


def _response(status, **fields):
    response = {
        "id": "resp_rec",
        "object": "response",
        "created_at": 1758600000,
        "model": "gpt-4.1",
        "status": status,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }
    response.update(fields)
    return response


def _added(item_id, name, call_id, seq):
    return {
        "type": "response.output_item.added",
        "sequence_number": seq,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": item_id,
            "name": name,
            "call_id": call_id,
            "arguments": "",
            "status": "in_progress",
        },
    }


def _args_delta(item_id, delta, seq):
    return {
        "type": "response.function_call_arguments.delta",
        "sequence_number": seq,
        "output_index": 0,
        "item_id": item_id,
        "delta": delta,
    }


def _args_done(item_id, name, arguments, seq):
    return {
        "type": "response.function_call_arguments.done",
        "sequence_number": seq,
        "output_index": 0,
        "item_id": item_id,
        "name": name,
        "arguments": arguments,
    }


def _item_done(item_id, name, call_id, arguments, seq):
    return {
        "type": "response.output_item.done",
        "sequence_number": seq,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": item_id,
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
            "status": "completed",
        },
    }


def _failed(seq):
    return {
        "type": "response.failed",
        "sequence_number": seq,
        "response": _response(
            "failed",
            error={"code": "server_error", "message": "The model failed to generate a response."},
        ),
    }


def _incomplete(seq):
    return {
        "type": "response.incomplete",
        "sequence_number": seq,
        "response": _response("incomplete", incomplete_details={"reason": "max_output_tokens"}),
    }


def _error(seq):
    return {
        "type": "error",
        "sequence_number": seq,
        "code": "server_error",
        "message": "upstream provider error",
        "param": None,
        "error": {"code": "server_error", "message": "upstream provider error"},
    }


def _completed(usage, seq):
    return {
        "type": "response.completed",
        "sequence_number": seq,
        "response": _response("completed", usage=usage),
    }


FULL_USAGE = {
    "input_tokens": 40,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 52,
}

NULL_COUNTS_USAGE = {
    "input_tokens": None,
    "input_tokens_details": {"cached_tokens": None},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": None},
    "total_tokens": None,
}


def _finished_call(item_id, name, call_id, arguments, seq):
    return [
        _added(item_id, name, call_id, seq),
        _args_done(item_id, name, arguments, seq + 1),
        _item_done(item_id, name, call_id, arguments, seq + 2),
    ]


async def _receive(service, *events):
    service._websocket = _ws(*events)
    await service._receive_response_events(MagicMock(spec=LLMContext), [])


def _dispatched(service):
    return [(fc.function_name, fc.arguments) for fc in service.run_function_calls.call_args.args[0]]


@pytest.mark.parametrize(
    "terminal",
    [
        pytest.param(_failed, id="failed"),
        pytest.param(_incomplete, id="incomplete"),
        pytest.param(_error, id="error"),
    ],
)
@pytest.mark.asyncio
async def test_websocket_terminal_event_does_not_run_an_announced_call(terminal):
    service = _make_service()

    await _receive(
        service,
        _added("fc_1", "transfer_call", "call_1", 1),
        _args_delta("fc_1", '{"destination": "+6', 2),
        terminal(3),
    )

    service.push_error.assert_called_once()
    service.run_function_calls.assert_not_called()


@pytest.mark.asyncio
async def test_websocket_incomplete_still_runs_the_call_that_finished():
    service = _make_service()

    await _receive(
        service,
        *_finished_call("fc_1", "get_weather", "call_1", '{"city": "SF"}', 1),
        _added("fc_2", "transfer_call", "call_2", 4),
        _args_delta("fc_2", '{"destination": "+6', 5),
        _incomplete(6),
    )

    assert _dispatched(service) == [("get_weather", {"city": "SF"})]


@pytest.mark.asyncio
async def test_websocket_failure_reports_the_servers_message():
    service = _make_service()

    await _receive(service, _failed(1))

    assert service.push_error.call_args.kwargs["error_msg"] == (
        "LLM response error: The model failed to generate a response."
    )


@pytest.mark.asyncio
async def test_websocket_incomplete_reports_the_servers_reason():
    service = _make_service()

    await _receive(service, _incomplete(1))

    assert service.push_error.call_args.kwargs["error_msg"] == (
        "LLM response error: max_output_tokens"
    )


@pytest.mark.asyncio
async def test_websocket_null_usage_counts_keep_the_finished_call():
    """A Responses-compatible server may send null counts."""
    service = _make_service()

    await _receive(
        service,
        *_finished_call("fc_1", "get_weather", "call_1", '{"city": "SF"}', 1),
        _completed(NULL_COUNTS_USAGE, 4),
    )

    assert _dispatched(service) == [("get_weather", {"city": "SF"})]
    service._store_previous_response_state.assert_called_once()
    usage = service.start_llm_usage_metrics.call_args.args[0]
    assert (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        usage.cache_read_input_tokens,
        usage.reasoning_tokens,
    ) == (0, 12, 0, 0, 0)


@pytest.mark.asyncio
async def test_control_websocket_finished_call_with_full_usage_runs():
    service = _make_service()

    await _receive(
        service,
        *_finished_call("fc_1", "get_weather", "call_1", '{"city": "SF"}', 1),
        _completed(FULL_USAGE, 4),
    )

    assert _dispatched(service) == [("get_weather", {"city": "SF"})]
    assert service.start_llm_usage_metrics.call_args.args[0].total_tokens == 52


def test_the_recorded_events_decode_as_sdk_stream_events():
    recorded = [
        _added("fc_1", "get_weather", "call_1", 1),
        _args_delta("fc_1", "{", 2),
        _args_done("fc_1", "get_weather", "{}", 3),
        _item_done("fc_1", "get_weather", "call_1", "{}", 4),
        _failed(5),
        _incomplete(6),
        _error(7),
        _completed(FULL_USAGE, 8),
        _completed(NULL_COUNTS_USAGE, 9),
    ]

    decoded = [construct_type(type_=ResponseStreamEvent, value=e) for e in recorded]

    assert [type(d).__name__ for d in decoded] == [
        "ResponseOutputItemAddedEvent",
        "ResponseFunctionCallArgumentsDeltaEvent",
        "ResponseFunctionCallArgumentsDoneEvent",
        "ResponseOutputItemDoneEvent",
        "ResponseFailedEvent",
        "ResponseIncompleteEvent",
        "ResponseErrorEvent",
        "ResponseCompletedEvent",
        "ResponseCompletedEvent",
    ]
    assert decoded[4].response.error.message == "The model failed to generate a response."
    assert decoded[5].response.incomplete_details.reason == "max_output_tokens"
    assert decoded[8].response.usage.input_tokens is None


def _process(function_calls):
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(api_key="test-key")
    return [
        (c.function_name, c.arguments)
        for c in service._process_function_calls(LLMContext(), function_calls)
    ]


def test_a_call_whose_arguments_do_not_parse_is_not_dispatched():
    calls = _process(
        {
            "fc_1": {"name": "transfer_call", "call_id": "call_1", "arguments": '{"to": "+614'},
            "fc_2": {"name": "get_weather", "call_id": "call_2", "arguments": '{"city": "SF"}'},
        }
    )

    assert calls == [("get_weather", {"city": "SF"})]


def test_control_calls_without_parameters_run_with_empty_arguments():
    calls = _process(
        {
            "fc_1": {"name": "check_hours", "call_id": "call_1", "arguments": "{}"},
            "fc_2": {"name": "end_call", "call_id": "call_2", "arguments": ""},
        }
    )

    assert calls == [("check_hours", {}), ("end_call", {})]
