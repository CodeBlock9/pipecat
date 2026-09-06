#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a language-model request can be given a deadline, and is classified.

Nothing bounded a generation. The SDK client's default is ten minutes, and
``retry_timeout_secs`` is applied only when ``retry_on_timeout=True``, which
nothing sets — so a provider that stopped responding mid-turn silenced the call
until the caller gave up. These tests pin the three parts of the fix: the
client-level timeout, the per-request override for the conversational turn, and
the classification of a timeout that the SDK raises as its own type.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APITimeoutError

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.openai.llm import OpenAILLMService


def test_no_deadline_by_default_keeps_the_sdk_behaviour():
    """Every new parameter defaults to None, so upstream is unchanged."""
    service = OpenAILLMService(api_key="test-key")

    assert service._request_timeout is None
    assert service._turn_request_timeout is None
    assert service._max_client_retries is None
    assert service._client.max_retries == 2


def test_the_client_carries_the_timeout_and_the_retry_bound():
    service = OpenAILLMService(
        api_key="test-key",
        request_timeout=httpx.Timeout(45.0, connect=5.0),
        max_client_retries=1,
    )

    assert service._client.max_retries == 1
    timeout = service._client._client.timeout
    assert timeout.read == 45.0
    assert timeout.connect == 5.0


def test_azure_forwards_the_same_settings():
    """Not a corner case: on a deployment that uses Azure it is every request."""
    service = AzureLLMService(
        api_key="test-key",
        endpoint="https://example.openai.azure.com",
        request_timeout=httpx.Timeout(45.0, connect=5.0),
        max_client_retries=1,
    )

    assert service._client.max_retries == 1
    assert service._client._client.timeout.read == 45.0


def test_anthropic_forwards_the_same_settings():
    service = AnthropicLLMService(api_key="test-key", request_timeout=45.0, max_client_retries=1)

    assert service._client.max_retries == 1
    assert service._client.timeout == 45.0


def test_a_supplied_anthropic_client_is_left_alone():
    """Its owner configured its timeouts; overriding them here is not ours."""

    class _Given:
        max_retries = 7

    given = _Given()
    service = AnthropicLLMService(api_key="test-key", client=given, request_timeout=45.0)

    assert service._client is given
    assert service._client.max_retries == 7


@pytest.mark.asyncio
async def test_the_streaming_turn_overrides_the_client_timeout_per_request():
    """The turn and the out-of-band inference share one client on purpose.

    On every non-realtime call the conversational service *is* the extraction
    and summarization client, and those are non-streaming calls that
    legitimately take much longer than a turn is allowed to. So the client
    carries the offline deadline and the streaming turn passes its own.
    """
    service = OpenAILLMService(
        api_key="test-key",
        request_timeout=httpx.Timeout(45.0, connect=5.0),
        turn_request_timeout=12.0,
    )

    captured = {}

    async def _create(**params):
        captured.update(params)
        return object()

    service._client.chat.completions.create = _create
    from pipecat.processors.aggregators.llm_context import LLMContext

    await service.get_chat_completions(LLMContext(messages=[]))

    assert captured["timeout"] == 12.0


@pytest.mark.asyncio
async def test_an_out_of_band_inference_does_not_take_the_turn_deadline():
    """`run_inference` is the final extraction; 12s would cut it in half."""
    service = OpenAILLMService(
        api_key="test-key",
        request_timeout=httpx.Timeout(45.0, connect=5.0),
        turn_request_timeout=12.0,
    )

    captured = {}

    class _Response:
        usage = None
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    async def _create(**params):
        captured.update(params)
        return _Response()

    service._client.chat.completions.create = _create
    from pipecat.processors.aggregators.llm_context import LLMContext

    assert await service.run_inference(LLMContext(messages=[])) == "ok"
    assert "timeout" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("stream stalled"),
        APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
        TimeoutError(),
    ],
    ids=["stream-iteration", "sdk-client-timeout", "asyncio"],
)
async def test_every_timeout_shape_fires_on_completion_timeout(exc):
    """Only `httpx.TimeoutException` was caught, and it is the one that never arrives.

    The SDK maps a client-level timeout on `create()` to
    `openai.APITimeoutError`, so the handler fired for stream-iteration
    timeouts alone: a generation that never started at all was reported as a
    generic completion error, with no `on_completion_timeout` event for an
    application to act on.
    """
    service = OpenAILLMService(api_key="test-key")

    # Asserted on the dispatch rather than on a registered handler: dispatching
    # an event needs a running task manager, which a service outside a pipeline
    # does not have.
    with (
        patch.object(service, "_process_context", side_effect=exc),
        patch.object(service, "push_frame", new=AsyncMock()),
        patch.object(service, "push_error", new=AsyncMock()) as push_error,
        patch.object(service, "start_processing_metrics", new=AsyncMock()),
        patch.object(service, "stop_processing_metrics", new=AsyncMock()),
        patch.object(service, "_call_event_handler", new=AsyncMock()) as call_event,
    ):
        from pipecat.processors.aggregators.llm_context import LLMContext

        await service.process_frame(
            LLMContextFrame(LLMContext(messages=[])), FrameDirection.DOWNSTREAM
        )

    call_event.assert_awaited_once_with("on_completion_timeout")
    assert push_error.await_args.kwargs["error_msg"] == "LLM completion timeout"
