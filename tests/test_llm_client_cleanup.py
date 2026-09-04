#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that language-model services close their provider SDK client.

The OpenAI and Anthropic clients pool connections with no keepalive expiry, so
a service whose client is never closed keeps its sockets for the life of the
process — once per call, in a process that runs calls back to back.
"""

import pytest

from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService


class _RecordingClient:
    def __init__(self):
        self.closes = 0

    async def close(self):
        self.closes += 1


class _UncloseableClient(_RecordingClient):
    async def close(self):
        await super().close()
        raise RuntimeError("the event loop is already gone")


class _StubClient:
    """A stand-in with no ``close()``, as the stub services use."""


@pytest.mark.asyncio
async def test_openai_cleanup_closes_the_client():
    service = OpenAILLMService(api_key="test-key")
    client = _RecordingClient()
    service._client = client

    await service.cleanup()

    assert client.closes == 1


@pytest.mark.asyncio
async def test_cleanup_twice_is_safe():
    service = OpenAILLMService(api_key="test-key")
    client = _RecordingClient()
    service._client = client

    await service.cleanup()
    await service.cleanup()

    assert client.closes == 2


@pytest.mark.asyncio
async def test_anthropic_cleanup_closes_the_client():
    service = AnthropicLLMService(api_key="test-key")
    client = _RecordingClient()
    service._client = client

    await service.cleanup()

    assert client.closes == 1


@pytest.mark.asyncio
async def test_responses_cleanup_closes_the_client():
    service = OpenAIResponsesHttpLLMService(api_key="test-key")
    client = _RecordingClient()
    service._client = client

    await service.cleanup()

    assert client.closes == 1


@pytest.mark.asyncio
async def test_a_client_that_will_not_close_does_not_strand_teardown():
    """A pipeline cleans processors up in sequence; one raising would skip the rest."""
    service = OpenAILLMService(api_key="test-key")
    client = _UncloseableClient()
    service._client = client

    await service.cleanup()

    assert client.closes == 1


@pytest.mark.asyncio
async def test_a_client_without_close_is_left_alone():
    service = OpenAILLMService(api_key="test-key")
    service._client = _StubClient()

    await service.cleanup()


def test_openai_compatible_services_inherit_the_close():
    for service_class in (AzureLLMService, DograhLLMService, OpenAILLMService):
        assert service_class.cleanup is BaseOpenAILLMService.cleanup
