#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Novita LLM service.

Novita runs the base OpenAI streaming loop, which
``test_openai_compatible_token_usage.py`` covers for it. What is Novita's own is
the client it builds.
"""

from unittest.mock import patch

from pipecat.services.novita.llm import NovitaLLMService


def test_novita_llm_client_uses_its_endpoint_and_default_model():
    with patch.object(NovitaLLMService, "create_client", return_value=object()) as create_client:
        service = NovitaLLMService(api_key="test-key")

    assert service._settings.model == "moonshotai/kimi-k2.5"
    kwargs = create_client.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://api.novita.ai/openai"
