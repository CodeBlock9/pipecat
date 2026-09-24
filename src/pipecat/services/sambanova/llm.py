#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SambaNova LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class SambaNovaLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for SambaNovaLLMService."""

    pass


class SambaNovaLLMService(OpenAILLMService):
    """A service for interacting with SambaNova using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to SambaNova's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # SambaNova doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = SambaNovaLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.sambanova.ai/v1",
        settings: Settings | None = None,
        **kwargs,
    ) -> None:
        """Initialize SambaNova LLM service.

        Args:
            api_key: The API key for accessing SambaNova API.
            model: The model identifier to use. Defaults to "Meta-Llama-3.3-70B-Instruct".

                .. deprecated:: 0.0.105
                    Use ``settings=SambaNovaLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            base_url: The base URL for SambaNova API. Defaults to "https://api.sambanova.ai/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="Meta-Llama-3.3-70B-Instruct")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> Any:
        """Create OpenAI-compatible client for SambaNova API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API endpoint. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            Configured OpenAI-compatible client instance.
        """
        logger.debug(f"Creating SambaNova client with API {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for SambaNova chat completion request.

        SambaNova doesn't support some OpenAI parameters like frequency_penalty,
        presence_penalty, and seed.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self._settings.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
            "max_completion_tokens": self._settings.max_completion_tokens,
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings.extra)

        return params
