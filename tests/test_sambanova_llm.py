#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for SambaNova LLM service.

SambaNova runs the base OpenAI streaming loop, which
``test_openai_compatible_token_usage.py`` and ``test_openai_tool_call_stream.py``
cover for it. What is SambaNova's own is the client it builds.
"""

from unittest.mock import patch

from pipecat.services.sambanova.llm import SambaNovaLLMService


def test_sambanova_llm_client_uses_its_endpoint_and_default_model():
    with patch.object(SambaNovaLLMService, "create_client", return_value=object()) as create_client:
        service = SambaNovaLLMService(api_key="test-key")

    assert service.supports_developer_role is False
    assert service._settings.model == "Meta-Llama-3.3-70B-Instruct"
    kwargs = create_client.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://api.sambanova.ai/v1"
