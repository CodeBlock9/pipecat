#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base LLM service implementation for services that use the AsyncOpenAI client."""

import asyncio
import json
import ssl
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN as OPENAI_NOT_GIVEN,
)
from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
)
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsFromLLMInfoFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

#: One TLS context for every provider client this process builds.
#:
#: httpx builds a context per ``AsyncClient`` and loads the system trust store
#: into each one. A call constructs one to three of these clients -- the
#: dialogue LLM, and optionally a variable-extraction and a voicemail one --
#: and they live for the length of the call, so the trust store was being
#: parsed and held once per client per call. Sharing one is safe: an
#: ``ssl.SSLContext`` is designed to back many connections and httpx never
#: mutates the one it is handed.
_SSL_CONTEXT: ssl.SSLContext | None = None
_SSL_CONTEXT_LOCK = threading.Lock()


def shared_ssl_context() -> ssl.SSLContext:
    """The process-wide TLS context provider clients verify against.

    Built on first use rather than at import, because loading the trust store
    is the cost being avoided and a process that never talks to a provider
    should not pay it either.

    Returns:
        The shared context, with httpx's own defaults.
    """
    global _SSL_CONTEXT
    with _SSL_CONTEXT_LOCK:
        if _SSL_CONTEXT is None:
            _SSL_CONTEXT = httpx.create_ssl_context()
        return _SSL_CONTEXT


@dataclass
class OpenAILLMSettings(LLMSettings):
    """Settings for BaseOpenAILLMService.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
    """

    # Override inherited LLMSettings fields to also accept the OpenAI SDK's
    # sentinel, which the service stores here so these fields can be passed
    # through unchanged to the AsyncOpenAI client.
    frequency_penalty: float | None | NotGiven | OpenAINotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    presence_penalty: float | None | NotGiven | OpenAINotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    seed: int | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_tokens: int | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_completion_tokens: int | None | NotGiven | OpenAINotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class BaseOpenAILLMService(LLMService[OpenAILLMAdapter]):
    """Base class for all services that use the AsyncOpenAI client.

    This service consumes LLMContextFrame frames, which contain a reference to
    an LLMContext object. The context defines what is sent to the LLM for
    completion, including user, assistant, and system messages, as well as tool
    choices and function call configurations.
    """

    Settings = OpenAILLMSettings
    _settings: Settings

    supports_developer_role: bool = True
    """Whether this service's API supports the "developer" message role.

    OpenAI's native API supports it, but some OpenAI-compatible services
    (e.g. Cerebras) do not. Subclasses that don't support it should set
    this to ``False``, which causes the adapter to convert "developer"
    messages to "user" messages before sending them to the API.
    """

    @deprecated(
        "`BaseOpenAILLMService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `BaseOpenAILLMService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for OpenAI model configuration.

        .. deprecated:: 0.0.105
            Use ``settings=BaseOpenAILLMService.Settings(...)`` instead of
            ``params=InputParams(...)``.
            Will be removed in 2.0.0.

        Parameters:
            frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0).
            presence_penalty: Penalty for new tokens (-2.0 to 2.0).
            seed: Random seed for deterministic outputs.
            temperature: Sampling temperature (0.0 to 2.0).
            top_k: Top-k sampling parameter (currently ignored by OpenAI).
            top_p: Top-p (nucleus) sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens in response (deprecated, use max_completion_tokens).
            max_completion_tokens: Maximum completion tokens to generate.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            extra: Additional model-specific parameters.
        """

        frequency_penalty: float | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=-2.0, le=2.0
        )
        presence_penalty: float | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=-2.0, le=2.0
        )
        seed: int | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=0
        )
        temperature: float | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=0.0, le=2.0
        )
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignored right now.
        top_k: int | None = Field(default=None, ge=0)
        top_p: float | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=0.0, le=1.0
        )
        max_tokens: int | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=1
        )
        max_completion_tokens: int | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN, ge=1
        )
        service_tier: str | None = Field(  # pyright: ignore[reportAssignmentType]
            default_factory=lambda: OPENAI_NOT_GIVEN
        )
        extra: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Mapping[str, str] | None = None,
        service_tier: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        request_timeout: float | httpx.Timeout | None = None,
        turn_request_timeout: float | httpx.Timeout | None = None,
        max_client_retries: int | None = None,
        **kwargs,
    ):
        """Initialize the BaseOpenAILLMService.

        Args:
            model: The OpenAI model name to use (e.g., "gpt-4.1", "gpt-4o").

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            params: Input parameters for model configuration and behavior.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            retry_timeout_secs: How long an inference may go without producing
                output before it is abandoned and re-issued when retrying is enabled.
            retry_on_timeout: Whether to re-issue once when the first attempt
                produces no output within ``retry_timeout_secs``.
            request_timeout: Timeout applied to the HTTP client, and therefore to
                every request this service makes that does not override it. A
                float is httpx's whole-operation timeout; an ``httpx.Timeout``
                lets connect and read be set separately, which is usually what
                is wanted -- ``read`` bounds time-to-first-token and then each
                inter-chunk gap, so a trickling stream is tolerated and a
                stopped one is not. ``None`` keeps the SDK's own default, which
                is ten minutes.
            turn_request_timeout: Timeout passed per request on the *streaming*
                completion only. A conversational turn and an out-of-band
                inference have very different deadlines and often share one
                client, so the client carries the looser one and the turn
                overrides it. ``None`` means the client's timeout applies to
                both.
            max_client_retries: Retries the SDK performs inside one call.
                ``None`` keeps the SDK's default of two. Set it to bound the
                worst case: with a request timeout, the deadline a caller
                experiences is the timeout multiplied by the attempts.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=OPENAI_NOT_GIVEN,
            presence_penalty=OPENAI_NOT_GIVEN,
            seed=OPENAI_NOT_GIVEN,
            temperature=OPENAI_NOT_GIVEN,
            top_p=OPENAI_NOT_GIVEN,
            top_k=None,
            max_tokens=OPENAI_NOT_GIVEN,
            max_completion_tokens=OPENAI_NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )

        # 2. Apply direct init arg overrides (no warnings in base class)
        if model is not None:
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None and not settings:
            default_settings.frequency_penalty = params.frequency_penalty
            default_settings.presence_penalty = params.presence_penalty
            default_settings.seed = params.seed
            default_settings.temperature = params.temperature
            default_settings.top_p = params.top_p
            default_settings.max_tokens = params.max_tokens
            default_settings.max_completion_tokens = params.max_completion_tokens
            if isinstance(params.extra, dict):
                default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )
        self._service_tier = service_tier
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        # Bound before create_client runs: subclasses override that method and
        # read these there.
        self._request_timeout = request_timeout
        self._turn_request_timeout = turn_request_timeout
        self._max_client_retries = max_client_retries
        self._full_model_name: str = ""
        self._client = self.create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            **kwargs,
        )

        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

        # Store node-transition calls that need to be executed after TTS.
        self._pending_node_transition_function_calls: list[FunctionCallFromLLM] = []

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create an AsyncOpenAI client instance.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for the API.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.
            **kwargs: Additional client configuration arguments.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        client_kwargs = {}
        if self._max_client_retries is not None:
            client_kwargs["max_retries"] = self._max_client_retries
        http_kwargs = {}
        if self._request_timeout is not None:
            http_kwargs["timeout"] = self._request_timeout
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                ),
                verify=shared_ssl_context(),
                **http_kwargs,
            ),
            default_headers=default_headers,
            **client_kwargs,
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI service supports metrics generation.
        """
        return True

    def set_full_model_name(self, full_model_name: str):
        """Set the full AI model name.

        Args:
            full_model_name: The full name of the AI model to use.
        """
        self._full_model_name = full_model_name

    def get_full_model_name(self):
        """Get the current full model name.

        Returns:
            The full name of the AI model being used.
        """
        return self._full_model_name

    async def get_chat_completions(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API with optional timeout and retry.

        Args:
            context: Context to use for the chat completion.
                Contains messages, tools, and tool choice.

        Returns:
            Async stream of chat completion chunks.
        """
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating chat from context {adapter.get_messages_for_logging(context)}"
        )

        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )

        params = self.build_chat_completion_params(params_from_context)

        # A per-request override rather than a client setting: one service
        # instance often serves both the conversational turn and out-of-band
        # inference, and a non-streaming inference legitimately takes much
        # longer than a turn is allowed to.
        if self._turn_request_timeout is not None:
            params["timeout"] = self._turn_request_timeout

        if self._retry_on_timeout:
            try:
                chunks = await asyncio.wait_for(
                    self._client.chat.completions.create(**params), timeout=self._retry_timeout_secs
                )
                return chunks
            except (TimeoutError, APITimeoutError):
                # Retry without the wait_for so we get a response. A configured
                # `request_timeout` still applies at the client, so the retry
                # carries the same deadline.
                logger.debug(f"{self}: Retrying chat completion due to timeout")
                chunks = await self._client.chat.completions.create(**params)
                return chunks
        else:
            chunks = await self._client.chat.completions.create(**params)
            return chunks

    async def cleanup(self):
        """Clean up the service and close the provider client.

        The client pools connections with no keepalive expiry, so its sockets
        outlive the service unless it is closed.
        """
        await super().cleanup()
        await self._close_provider_client(self._client)

    def _wire_callable(self):
        """The SDK method the built parameters are passed to."""
        return self._client.chat.completions.create

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for chat completion request.

        Subclasses can override this to customize parameters for different providers.

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
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "seed": self._settings.seed,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
            "max_completion_tokens": self._settings.max_completion_tokens,
            "service_tier": self._service_tier
            if self._service_tier is not None
            else OPENAI_NOT_GIVEN,
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params = self.merge_provider_options(params)
        params = self._route_unsupported_options_to_extra_body(params)

        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens/max_completion_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        effective_instruction = system_instruction or assert_given(
            self._settings.system_instruction
        )
        adapter = self.get_llm_adapter()
        invocation_params = adapter.get_llm_invocation_params(
            context,
            system_instruction=effective_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )

        # Build params using the same method as streaming completions
        params = self.build_chat_completion_params(invocation_params)

        # Override for non-streaming
        params["stream"] = False
        params.pop("stream_options", None)

        # Override max_tokens if provided
        if max_tokens is not None:
            # Use max_completion_tokens for newer models, fallback to max_tokens
            if "max_completion_tokens" in params:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

        # LLM completion
        response = await self._client.chat.completions.create(**params)

        self.record_inference_usage(self._token_usage(response))

        return response.choices[0].message.content

    def _token_usage(self, completion) -> LLMTokenUsage | None:
        """Build token usage from a chat completion or one streamed chunk.

        A streamed ``ChatCompletionChunk`` carries the same ``usage`` attribute
        as a whole ``ChatCompletion``, so the streaming loop and
        ``run_inference`` meter on one basis. A count the provider leaves out
        is reported as 0: ``LLMTokenUsage`` requires integers, and a usage
        block that failed to build would lose the whole turn, tool calls
        included.

        Args:
            completion: The completion, or the streamed chunk, to read.

        Returns:
            The usage, or None when the completion carries none.
        """
        usage = getattr(completion, "usage", None)
        if not usage:
            return None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return LLMTokenUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            cache_read_input_tokens=(
                getattr(prompt_details, "cached_tokens", None) if prompt_details else None
            ),
            reasoning_tokens=(
                getattr(completion_details, "reasoning_tokens", None)
                if completion_details
                else None
            ),
        )

    @staticmethod
    def _accumulate_tool_call_deltas(calls: dict[int | str, dict[str, str]], tool_calls) -> None:
        """Fold every tool-call entry of one streamed delta into the turn's calls.

        A provider may send several calls in one delta, or interleave the
        fragments of several calls, so every entry is read, not only the first.
        An entry belongs to the call its ``index`` names. A provider that sends
        no index (SambaNova) is keyed by the call's ``id`` instead, and an entry
        with neither continues the call started last. Name and argument
        fragments are appended, and the id is recorded when the entry has one.

        Args:
            calls: The turn's accumulator, in the order the calls started. Each
                value holds the ``name``, ``id`` and ``arguments`` gathered so far.
            tool_calls: The ``tool_calls`` of one streamed delta.
        """
        for tool_call in tool_calls:
            if tool_call.index is not None:
                key = tool_call.index
            elif tool_call.id:
                key = tool_call.id
            else:
                key = next(reversed(calls), 0)
            call = calls.setdefault(key, {"name": "", "id": "", "arguments": ""})
            if tool_call.id:
                call["id"] = tool_call.id
            if tool_call.function and tool_call.function.name:
                call["name"] += tool_call.function.name
            if tool_call.function and tool_call.function.arguments:
                call["arguments"] += tool_call.function.arguments

    def _function_calls_from_stream(
        self, context: LLMContext, calls: dict[int | str, dict[str, str]]
    ) -> list[FunctionCallFromLLM]:
        """Build the function calls a streamed turn asked for.

        Calls are built in the order they started. A call that never received
        a name cannot be dispatched, so it is skipped with a warning. Arguments
        default to ``{}``, and a call whose arguments do not parse is skipped
        with a warning. Either way, the other calls of the turn still run.

        Args:
            context: The context the turn ran on.
            calls: The accumulator ``_accumulate_tool_call_deltas`` filled.

        Returns:
            The function calls to run, possibly none.
        """
        function_calls = []
        for call in calls.values():
            if not call["name"]:
                logger.warning(f"{self}: Skipping a streamed tool call with no name: {call}")
                continue
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                logger.warning(
                    f"{self}: Failed to parse function call arguments: {call['arguments']}"
                )
                continue
            function_calls.append(
                FunctionCallFromLLM(
                    context=context,
                    tool_call_id=call["id"],
                    function_name=call["name"],
                    arguments=arguments,
                )
            )
        return function_calls

    @traced_llm
    async def _process_context(self, context: LLMContext):
        # The turn's tool calls, gathered from every delta of the stream and
        # dispatched together once it ends.
        calls: dict[int | str, dict[str, str]] = {}

        # Reset pending node-transition calls when processing a new context.
        self._pending_node_transition_function_calls = []

        # Flag to store whether some text was generated in the current generation
        text_generated_signal = False

        await self.start_ttfb_metrics()

        # Generate chat completions from LLMContext
        chunk_stream = await self.get_chat_completions(context)

        # Ensure stream and its async iterator are closed on cancellation/exception
        # to prevent socket leaks and uvloop crashes. Closing the iterator first
        # cascades cleanup through nested async generators (httpx/httpcore internals),
        # preventing uvloop's broken asyncgen finalizer from firing on Python 3.12+
        # (MagicStack/uvloop#699).
        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                # Close the iterator first to cascade cleanup through
                # nested async generators (httpx/httpcore internals).
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                # Then close the stream to release HTTP resources.
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        # Providers differ in how often they send usage: some once at the end,
        # others a cumulative snapshot on every chunk. Holding the latest and
        # reporting it after the stream keeps that to one report per completion.
        token_usage: LLMTokenUsage | None = None

        try:
            async with _closing(chunk_stream) as chunk_iter:
                async for chunk in chunk_iter:
                    # Guarded, so a chunk without usage after the one that
                    # carried it does not clear the turn's usage.
                    if chunk.usage:
                        token_usage = self._token_usage(chunk)

                    if chunk.model and self.get_full_model_name() != chunk.model:
                        self.set_full_model_name(chunk.model)

                    if chunk.choices is None or len(chunk.choices) == 0:
                        continue

                    await self.stop_ttfb_metrics()

                    if not chunk.choices[0].delta:
                        continue

                    if chunk.choices[0].delta.tool_calls:
                        # A turn that only calls tools produces no answer text, so
                        # the call itself is what the caller gets and TTFAT ends
                        # here rather than going unmeasured.
                        await self.stop_ttfat_metrics()

                        # Text is pushed chunk by chunk, but a tool call is
                        # only usable whole: its fragments are gathered here and
                        # the calls are built once the stream ends.
                        self._accumulate_tool_call_deltas(calls, chunk.choices[0].delta.tool_calls)
                    elif chunk.choices[0].delta.content:
                        text_generated_signal = True
                        await self._push_llm_text(chunk.choices[0].delta.content)

                    # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
                    # we need to get LLMTextFrame for the transcript
                    elif (
                        hasattr(chunk.choices[0].delta, "audio")
                        and chunk.choices[0].delta.audio
                        and chunk.choices[0].delta.audio.get("transcript")
                    ):
                        await self.push_frame(
                            LLMTextFrame(chunk.choices[0].delta.audio["transcript"])
                        )
        finally:
            # Report even if the response is interrupted or cancelled mid-stream.
            if token_usage:
                await self.start_llm_usage_metrics(token_usage)

        # Run every call the turn asked for. A call with no registered handler
        # gets the missing-function result from run_function_calls.
        if calls:
            function_calls = self._function_calls_from_stream(context, calls)

            # Send the info frame with function calls so that it can be traced by service_decorators
            await self.push_frame(
                FunctionCallsFromLLMInfoFrame(function_calls=function_calls),
                direction=FrameDirection.DOWNSTREAM,
            )

            await self._run_or_defer_function_calls(
                function_calls,
                text_generated=text_generated_signal,
            )

    async def _run_or_defer_function_calls(
        self,
        function_calls: list[FunctionCallFromLLM],
        *,
        text_generated: bool,
    ) -> None:
        """Defer node-transition batches until their preceding TTS completes."""
        contains_node_transition = any(
            self._function_is_node_transition(fc.function_name) for fc in function_calls
        )
        if text_generated and contains_node_transition:
            # Keep a provider tool-call batch together. Splitting a mixed batch
            # would discard Pipecat's shared function-call group and could run
            # the LLM before every result from the original batch has arrived.
            self._pending_node_transition_function_calls = function_calls
            logger.debug(
                f"{self}: Deferring {len(function_calls)} node-transition "
                "function calls until after TTS"
            )
            return

        logger.debug(f"{self}: Executing {len(function_calls)} function calls")
        await self.run_function_calls(function_calls)

    async def _run_pending_node_transition_function_calls(self) -> None:
        if not self._pending_node_transition_function_calls:
            return

        function_calls = self._pending_node_transition_function_calls
        self._pending_node_transition_function_calls = []
        logger.debug(
            f"{self}: Executing {len(function_calls)} deferred node-transition "
            "function calls after TTS"
        )
        await self.run_function_calls(function_calls)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # Handle BotStoppedSpeakingFrame to execute pending node-transition calls.
        if isinstance(frame, BotStoppedSpeakingFrame):
            await self._run_pending_node_transition_function_calls()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except (TimeoutError, httpx.TimeoutException, APITimeoutError) as e:
                # All three shapes a request deadline can take: the SDK raises
                # `APITimeoutError` for a client-level timeout on `create()`
                # and lets httpx's own exception through while the stream is
                # being iterated.
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
