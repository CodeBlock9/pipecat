#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for routing unknown provider options into ``extra_body``.

A provider option the SDK method has no parameter for is sent as a raw body
field, so options an OpenAI-compatible endpoint understands keep reaching it
and the rest come back as a provider response rather than a ``TypeError``
raised before the request leaves the process.
"""

import functools
import io
from types import SimpleNamespace

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_warnings():
    """Attach a fresh loguru sink that captures WARNING-and-above messages."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    return sink, handler_id


def _build(service, options: dict) -> tuple[dict, str]:
    """Apply provider options and build the request parameters."""
    service.apply_provider_options(options)
    sink, handler_id = _capture_warnings()
    try:
        params = service.build_chat_completion_params({"messages": []})
    finally:
        logger.remove(handler_id)
    return params, sink.getvalue()


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def test_unknown_option_moves_to_extra_body_and_warns_once():
    """``reasoning`` is a Responses-API field; chat completions has no parameter for it."""
    service = OpenAILLMService(api_key="test-key")

    params, warnings = _build(service, {"reasoning": {"effort": "low"}})

    assert "reasoning" not in params
    assert params["extra_body"] == {"reasoning": {"effort": "low"}}
    assert warnings.count("reasoning") == 1

    # A second turn is silent — the warning is per service instance, not per request.
    sink, handler_id = _capture_warnings()
    try:
        again = service.build_chat_completion_params({"messages": []})
    finally:
        logger.remove(handler_id)
    assert again["extra_body"] == {"reasoning": {"effort": "low"}}
    assert sink.getvalue() == ""


def test_known_option_stays_at_the_top_level():
    """``reasoning_effort`` and ``verbosity`` are real chat-completion parameters."""
    service = OpenAILLMService(api_key="test-key")

    params, warnings = _build(service, {"reasoning_effort": "low", "verbosity": "high"})

    assert params["reasoning_effort"] == "low"
    assert params["verbosity"] == "high"
    assert "extra_body" not in params
    assert warnings == ""


def test_openai_compatible_option_still_reaches_the_request_body():
    """DeepSeek's ``thinking`` rides ``extra_body`` rather than being dropped."""
    service = OpenAILLMService(api_key="test-key")

    params, _ = _build(service, {"thinking": {"type": "enabled"}})

    assert params["extra_body"] == {"thinking": {"type": "enabled"}}


def test_an_explicit_extra_body_is_merged_not_replaced():
    service = OpenAILLMService(api_key="test-key")

    params, _ = _build(
        service,
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "reasoning": {"e": 1}},
    )

    assert params["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"e": 1},
    }


def test_the_wire_signature_is_read_once_per_service():
    service = OpenAILLMService(api_key="test-key")
    service.apply_provider_options({"reasoning": {"effort": "low"}})

    calls = 0
    original = service._read_wire_parameter_names

    def counting():
        nonlocal calls
        calls += 1
        return original()

    service._read_wire_parameter_names = counting

    for _ in range(5):
        service.build_chat_completion_params({"messages": []})

    assert calls == 1


@pytest.mark.asyncio
async def test_run_inference_reaches_the_sdk_with_the_option_in_the_body():
    """The out-of-band path builds its parameters through the same routing."""
    service = OpenAILLMService(api_key="test-key")
    service.apply_provider_options({"reasoning": {"effort": "low"}})

    captured = {}

    # `functools.wraps` keeps the SDK method's signature, which is what the
    # routing reads, so the stand-in accepts exactly what the real one does.
    @functools.wraps(service._client.chat.completions.create)
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage=None,
        )

    service._client.chat.completions.create = create

    answer = await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]))

    assert answer == "hi"
    assert "reasoning" not in captured
    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}


def test_no_options_leaves_the_parameters_alone():
    service = OpenAILLMService(api_key="test-key")

    params, warnings = _build(service, {})

    assert "extra_body" not in params
    assert warnings == ""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_routes_against_its_own_wire_method():
    """``thinking`` is an Anthropic parameter; ``reasoning`` is not."""
    service = AnthropicLLMService(api_key="test-key")

    supported = service._resolve_wire_parameter_names()

    assert supported is not None
    assert "thinking" in supported
    assert "reasoning" not in supported

    params = {"model": "claude", "thinking": {"type": "enabled"}, "reasoning": {"effort": "low"}}
    routed = service._route_unsupported_options_to_extra_body(params)

    assert routed["thinking"] == {"type": "enabled"}
    assert routed["extra_body"] == {"reasoning": {"effort": "low"}}


# ---------------------------------------------------------------------------
# Services with no single wire method
# ---------------------------------------------------------------------------


def test_a_service_without_a_wire_callable_routes_nothing():
    service = OpenAILLMService(api_key="test-key")
    service._wire_callable = lambda: None

    params = {"anything": 1}

    assert service._route_unsupported_options_to_extra_body(params) == {"anything": 1}
