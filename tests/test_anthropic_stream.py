#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that the Anthropic stream runs every tool_use block and reads usage as cumulative.

Every event is an Anthropic wire-format stream event decoded by the SDK's own
lenient decoder (``anthropic._models.construct_type`` into
``BetaRawMessageStreamEvent``), which is what the stream yields. A message can
carry several ``tool_use`` blocks, each streaming its own JSON by content-block
index. The usage counters are cumulative: ``message_start`` reports them, and
``message_delta`` reports running totals, with ``input_tokens`` optional.
"""

from unittest.mock import AsyncMock

import pytest
from anthropic._models import construct_type
from anthropic.types.beta import BetaRawMessageStreamEvent

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService

CHECK = ("check_availability", "toolu_a", {"party": 2})
SMS = ("send_sms", "toolu_b", {"to": "+61400000000"})


def _ev(raw):
    return construct_type(type_=BetaRawMessageStreamEvent, value=raw)


def _message_start(input_tokens=400, output_tokens=1, cache_creation=0, cache_read=0):
    return _ev(
        {
            "type": "message_start",
            "message": {
                "id": "msg_rec",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                },
            },
        }
    )


def _text_block(index, text):
    return [
        _ev(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _ev(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
        ),
        _ev({"type": "content_block_stop", "index": index}),
    ]


def _tool_block(index, tool_id, name, json_parts):
    events = [
        _ev(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            }
        )
    ]
    for part in json_parts:
        events.append(
            _ev(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": part},
                }
            )
        )
    events.append(_ev({"type": "content_block_stop", "index": index}))
    return events


def _message_delta(usage):
    return _ev(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": usage,
        }
    )


MESSAGE_STOP = _ev({"type": "message_stop"})

FULL_DELTA_USAGE = {
    "input_tokens": 400,
    "output_tokens": 42,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def _service(events):
    service = AnthropicLLMService(api_key="test-key")

    async def generator():
        for event in events:
            yield event

    async def fake_stream(api_call, params):
        return generator()

    service._create_message_stream = fake_stream
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_ttfat_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()
    service._report_usage_metrics = AsyncMock()
    return service


async def _run(events):
    """Run one turn; return the service and (name, tool_call_id, arguments) of each call run."""
    service = _service(events)
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "Book a table and text me"}])
    )
    dispatched = [
        (fc.function_name, fc.tool_call_id, fc.arguments)
        for call in service.run_function_calls.await_args_list
        for fc in call.args[0]
    ]
    return service, dispatched


@pytest.mark.asyncio
async def test_two_tool_use_blocks_both_run():
    service, dispatched = await _run(
        [
            _message_start(),
            *_text_block(0, "Let me check that."),
            *_tool_block(1, "toolu_a", "check_availability", ['{"par', 'ty": 2}']),
            *_tool_block(2, "toolu_b", "send_sms", ['{"to": ', '"+61400000000"}']),
            _message_delta(FULL_DELTA_USAGE),
            MESSAGE_STOP,
        ]
    )

    service.push_error.assert_not_called()
    assert dispatched == [CHECK, SMS]


@pytest.mark.asyncio
async def test_an_unparseable_block_is_skipped_and_the_other_runs():
    service, dispatched = await _run(
        [
            _message_start(),
            *_tool_block(0, "toolu_a", "check_availability", ['{"party": 2}']),
            *_tool_block(1, "toolu_b", "send_sms", ['{"to": ']),
            _message_delta(FULL_DELTA_USAGE),
            MESSAGE_STOP,
        ]
    )

    service.push_error.assert_not_called()
    assert dispatched == [CHECK]


@pytest.mark.asyncio
async def test_message_delta_without_input_tokens_still_runs_the_call():
    """``input_tokens`` is optional on message_delta; message_start's count stands."""
    service, dispatched = await _run(
        [
            _message_start(input_tokens=400, output_tokens=1),
            *_tool_block(0, "toolu_a", "check_availability", ['{"party": 2}']),
            _message_delta({"output_tokens": 42}),
            MESSAGE_STOP,
        ]
    )

    service.push_error.assert_not_called()
    assert dispatched == [CHECK]
    reported = service._report_usage_metrics.await_args.kwargs
    assert (reported["prompt_tokens"], reported["completion_tokens"]) == (400, 42)


@pytest.mark.asyncio
async def test_cumulative_usage_is_reported_as_the_final_totals_not_summed():
    """message_delta repeats the running totals; they replace message_start's, never add to them."""
    service, _ = await _run(
        [
            _message_start(input_tokens=400, output_tokens=1, cache_creation=10, cache_read=20),
            *_tool_block(0, "toolu_a", "check_availability", ['{"party": 2}']),
            _message_delta(
                {
                    "input_tokens": 400,
                    "output_tokens": 42,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                }
            ),
            MESSAGE_STOP,
        ]
    )

    reported = service._report_usage_metrics.await_args.kwargs
    assert reported == {
        "prompt_tokens": 400,
        "completion_tokens": 42,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
    }


@pytest.mark.asyncio
async def test_control_one_tool_use_block_runs():
    service, dispatched = await _run(
        [
            _message_start(),
            *_tool_block(0, "toolu_a", "check_availability", ['{"party": 2}']),
            _message_delta(FULL_DELTA_USAGE),
            MESSAGE_STOP,
        ]
    )

    service.push_error.assert_not_called()
    assert dispatched == [CHECK]


def test_the_fixture_is_the_sdk_decoding():
    delta = _message_delta({"output_tokens": 42})
    assert type(delta).__name__ == "BetaRawMessageDeltaEvent"
    assert delta.usage.input_tokens is None
    start = _tool_block(1, "toolu_b", "send_sms", [])[0]
    assert type(start).__name__ == "BetaRawContentBlockStartEvent"
    assert start.index == 1
