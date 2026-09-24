#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that the OpenAI chat streaming loop runs every tool call a turn asks for.

Every chunk is a recorded ``chat.completion.chunk`` in wire format, decoded by
the SDK's own lenient decoder (``openai._models.construct_type``, which the
client uses for every streamed chunk), so the objects are exactly what the loop
sees. The loop reads every entry of a delta and keys the calls by index, then by
id, so parallel calls survive whichever of the stream shapes below a provider
uses. SambaNova, which once kept its own copy of the loop for index-less
streams, runs through the same cases.
"""

from unittest.mock import AsyncMock, patch

import pytest
from openai._models import construct_type
from openai.types.chat import ChatCompletionChunk

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.baseten.llm import BasetenLLMService
from pipecat.services.novita.llm import NovitaLLMService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.perplexity.llm import PerplexityLLMService
from pipecat.services.sambanova.llm import SambaNovaLLMService
from pipecat.services.xai.llm import GrokLLMService

SERVICES = [
    pytest.param(OpenAILLMService, id="openai"),
    pytest.param(SambaNovaLLMService, id="sambanova"),
]

# The services test_openai_compatible_token_usage.py covers, for the usage cases.
USAGE_SERVICES = [
    pytest.param(OpenAILLMService, id="openai"),
    pytest.param(BasetenLLMService, id="baseten"),
    pytest.param(GrokLLMService, id="grok"),
    pytest.param(NovitaLLMService, id="novita"),
    pytest.param(PerplexityLLMService, id="perplexity"),
    pytest.param(NvidiaLLMService, id="nvidia"),
    pytest.param(SambaNovaLLMService, id="sambanova"),
]

CHECK = ("check_availability", "call_a", {"party": 2})
SMS = ("send_sms", "call_b", {"to": "+61400000000"})


def _chunk(delta=None, usage=None, finish_reason=None, with_choice=True):
    """One recorded chat.completion.chunk, decoded as the SDK decodes it."""
    raw = {
        "id": "chatcmpl-rec",
        "object": "chat.completion.chunk",
        "created": 1758600000,
        "model": "test-model",
        "choices": (
            [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
            if with_choice
            else []
        ),
    }
    if usage is not None:
        raw["usage"] = usage
    return construct_type(type_=ChatCompletionChunk, value=raw)


def _tc(index=None, *, id=None, name=None, arguments=None):
    """One entry of delta.tool_calls in wire format; None leaves a field out."""
    entry = {"type": "function", "function": {}}
    if index is not None:
        entry["index"] = index
    if id is not None:
        entry["id"] = id
    if name is not None:
        entry["function"]["name"] = name
    if arguments is not None:
        entry["function"]["arguments"] = arguments
    return entry


def _calls(*entries):
    """A chunk whose delta carries the given tool-call entries."""
    return _chunk(delta={"tool_calls": list(entries)})


FINISH = _chunk(delta={}, finish_reason="tool_calls")


class _FakeStream:
    """Stands in for the SDK's AsyncStream, which the loop iterates and closes."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _service(service_class, chunks):
    with patch.object(service_class, "create_client"):
        service = service_class(
            api_key="test-key", settings=service_class.Settings(model="test-model")
        )
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_ttfat_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


async def _dispatched(service_class, chunks):
    """(name, tool_call_id, arguments) of every call the turn handed to run_function_calls."""
    service = _service(service_class, chunks)
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "Book a table and text me"}])
    )
    return [
        (fc.function_name, fc.tool_call_id, fc.arguments)
        for call in service.run_function_calls.await_args_list
        for fc in call.args[0]
    ]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_two_calls_in_one_delta_both_run(service_class):
    """A provider that sends both calls whole in one delta."""
    chunks = [
        _chunk(delta={"role": "assistant", "content": None}),
        _calls(
            _tc(0, id="call_a", name="check_availability", arguments='{"party": 2}'),
            _tc(1, id="call_b", name="send_sms", arguments='{"to": "+61400000000"}'),
        ),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [CHECK, SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_interleaved_fragments_accumulate_by_index(service_class):
    """Fragments of two calls arrive interleaved; each index keeps its own."""
    chunks = [
        _calls(_tc(0, id="call_a", name="check_availability", arguments="")),
        _calls(_tc(1, id="call_b", name="send_sms", arguments="")),
        _calls(_tc(0, arguments='{"par')),
        _calls(_tc(1, arguments='{"to": ')),
        _calls(_tc(0, arguments='ty": 2}')),
        _calls(_tc(1, arguments='"+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [CHECK, SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_calls_without_an_index_key_by_id(service_class):
    """SambaNova's shape: one whole call per delta, with an id and no index.

    Before, the loop compared the missing index with its counter and ran a
    phantom call with no name ahead of the two real ones.
    """
    chunks = [
        _chunk(delta={"role": "assistant", "content": None}),
        _calls(_tc(id="call_a", name="check_availability", arguments='{"party": 2}')),
        _calls(_tc(id="call_b", name="send_sms", arguments='{"to": "+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [CHECK, SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_fragments_with_neither_index_nor_id_continue_the_last_call(service_class):
    """An index-less stream whose argument fragments carry no id join the call they follow."""
    chunks = [
        _calls(_tc(id="call_a", name="check_availability", arguments='{"par')),
        _calls(_tc(arguments='ty": 2}')),
        _calls(_tc(id="call_b", name="send_sms", arguments='{"to": ')),
        _calls(_tc(arguments='"+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [CHECK, SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_an_unparseable_call_is_skipped_and_the_other_runs(service_class):
    chunks = [
        _calls(_tc(0, id="call_a", name="check_availability", arguments='{"party": ')),
        _calls(_tc(1, id="call_b", name="send_sms", arguments='{"to": "+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_a_nameless_call_is_skipped_and_the_other_runs(service_class):
    chunks = [
        _calls(_tc(0, id="call_a", arguments='{"party": 2}')),
        _calls(_tc(1, id="call_b", name="send_sms", arguments='{"to": "+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_control_sequential_calls_run(service_class):
    """Control: OpenAI's own shape, one call finished before the next begins."""
    chunks = [
        _calls(_tc(0, id="call_a", name="check_availability", arguments="")),
        _calls(_tc(0, arguments='{"party": 2}')),
        _calls(_tc(1, id="call_b", name="send_sms", arguments="")),
        _calls(_tc(1, arguments='{"to": "+61400000000"}')),
        FINISH,
        _chunk(
            with_choice=False,
            usage={"prompt_tokens": 40, "completion_tokens": 17, "total_tokens": 57},
        ),
    ]
    assert await _dispatched(service_class, chunks) == [CHECK, SMS]


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_limit_whole_calls_with_neither_index_nor_id_merge_and_are_skipped(service_class):
    """A documented limit: with neither an index nor an id there is nothing to key on.

    The second call continues the first, the merged arguments do not parse, and
    both are skipped. A call with no id cannot complete a tool round trip anyway.
    No provider is known to stream this shape; change the rule deliberately.
    """
    chunks = [
        _calls(_tc(name="check_availability", arguments='{"party": 2}')),
        _calls(_tc(name="send_sms", arguments='{"to": "+61400000000"}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == []


@pytest.mark.parametrize("service_class", SERVICES)
@pytest.mark.asyncio
async def test_limit_an_id_only_start_then_index_only_fragments_split(service_class):
    """A documented limit: a call that starts keyed by id cannot be continued by index.

    The named start runs with ``{}``, and the index-keyed arguments are skipped
    as a call with no name. No provider is known to stream this shape; change
    the rule deliberately.
    """
    chunks = [
        _calls(_tc(id="call_a", name="check_availability", arguments="")),
        _calls(_tc(0, arguments='{"party": 2}')),
        FINISH,
    ]
    assert await _dispatched(service_class, chunks) == [("check_availability", "call_a", {})]


@pytest.mark.parametrize("service_class", USAGE_SERVICES)
@pytest.mark.asyncio
async def test_partial_usage_keeps_the_turn(service_class):
    """A usage block without prompt and total counts reports 0 for them, and the call runs.

    Before, the missing counts failed ``LLMTokenUsage``'s validation, and the
    turn was lost with its tool call.
    """
    chunks = [
        _calls(_tc(0, id="call_a", name="check_availability", arguments="")),
        _calls(_tc(0, arguments='{"party": 2}')),
        FINISH,
        _chunk(with_choice=False, usage={"completion_tokens": 17}),
    ]
    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        assert await _dispatched(service_class, chunks) == [CHECK]

    reported.assert_called_once()
    usage = reported.call_args.args[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 17, 0)


@pytest.mark.parametrize("service_class", USAGE_SERVICES)
@pytest.mark.asyncio
async def test_usage_survives_a_later_chunk_without_usage(service_class):
    """A provider that attaches usage to the finishing chunk and sends one more after it."""
    chunks = [
        _chunk(delta={"content": "Your table is booked."}),
        _chunk(
            delta={},
            finish_reason="stop",
            usage={"prompt_tokens": 40, "completion_tokens": 17, "total_tokens": 57},
        ),
        _chunk(delta={}),
    ]
    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        await _dispatched(service_class, chunks)

    reported.assert_called_once()
    usage = reported.call_args.args[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (40, 17, 57)


def test_the_fixture_is_the_sdk_decoding():
    """The chunks are SDK objects: a partial usage block and a missing index decode to None."""
    usage_chunk = _chunk(with_choice=False, usage={"completion_tokens": 17})
    assert isinstance(usage_chunk, ChatCompletionChunk)
    assert usage_chunk.usage.prompt_tokens is None
    assert usage_chunk.usage.total_tokens is None
    two = _calls(_tc(0, id="a", name="x"), _tc(1, id="b", name="y"))
    assert [t.index for t in two.choices[0].delta.tool_calls] == [0, 1]
    no_index = _calls(_tc(id="a", name="x"))
    assert no_index.choices[0].delta.tool_calls[0].index is None
