#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a generation producing nothing says so.

A model that answers with nothing pushes no error frame, produces no
classified failure and leaves a run row that reads `completed`. The transcript
shows a bot that stopped answering; nothing else does. This warning is the only
signal such a turn has.
"""

import pytest
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    EMPTY_COMPLETION_LOG,
    LLMAssistantAggregator,
)
from pipecat.services.llm_service import FunctionCallFromLLM


def _aggregator() -> LLMAssistantAggregator:
    return LLMAssistantAggregator(context=LLMContext(messages=[]))


async def _generation(aggregator, *frames):
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING")
    try:
        await aggregator._handle_llm_start(LLMFullResponseStartFrame())
        for frame in frames:
            if isinstance(frame, TextFrame):
                await aggregator._handle_text(frame)
            elif isinstance(frame, FunctionCallsStartedFrame):
                await aggregator._handle_function_calls_started(frame)
        await aggregator._handle_llm_end(LLMFullResponseEndFrame())
    finally:
        logger.remove(sink_id)
    return [line for line in lines if EMPTY_COMPLETION_LOG in line]


@pytest.mark.asyncio
async def test_a_generation_with_no_text_and_no_tool_call_is_reported():
    warnings = await _generation(_aggregator())

    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_a_generation_that_produced_text_is_not_reported():
    warnings = await _generation(_aggregator(), TextFrame("Hello there"))

    assert warnings == []


@pytest.mark.asyncio
async def test_a_tool_only_turn_is_not_reported():
    """The function-call guard is what keeps this from crying wolf.

    A turn whose whole output is a tool call legitimately produces no text, and
    those are common enough that warning on them would bury the real case.
    """
    call = FunctionCallFromLLM(
        function_name="lookup",
        tool_call_id="call-1",
        arguments={},
        context=LLMContext(messages=[]),
    )
    warnings = await _generation(
        _aggregator(),
        FunctionCallsStartedFrame(function_calls=[call]),
    )

    assert warnings == []


@pytest.mark.asyncio
async def test_the_flag_resets_between_generations():
    """One good turn must not vouch for the next."""
    aggregator = _aggregator()

    assert await _generation(aggregator, TextFrame("Hello")) == []
    assert len(await _generation(aggregator)) == 1


def test_the_literal_is_a_module_constant():
    """An alarm matches this string; a reworded f-string would blind it."""
    assert EMPTY_COMPLETION_LOG == "LLM generation produced no text"
    assert FunctionSchema  # imported to keep the tool-call shape explicit
