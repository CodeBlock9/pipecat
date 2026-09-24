import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic import Field as PydanticField

from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.utils.types import NOT_GIVEN, NotGiven


class NestedOptions(BaseModel):
    enabled: bool = False
    metadata: dict = PydanticField(default_factory=dict)


@dataclass
class NestedSettings(ServiceSettings):
    nested: NestedOptions | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def test_provider_options_apply_declared_settings_and_deep_merge_overflow():
    service = AIService(settings=ServiceSettings(model="base-model"))

    service.apply_provider_options(
        {
            "model": "advanced-model",
            "metadata": {
                "source": "advanced",
                "nested": {"enabled": True},
            },
        }
    )

    assert service._settings.model == "advanced-model"
    assert service.provider_options["model"] == "advanced-model"
    assert service._settings.extra == {
        "metadata": {
            "source": "advanced",
            "nested": {"enabled": True},
        }
    }
    assert service.merge_provider_options(
        {
            "model": "base-model",
            "metadata": {
                "base": True,
                "nested": {"count": 1},
            },
        }
    ) == {
        "model": "advanced-model",
        "metadata": {
            "base": True,
            "source": "advanced",
            "nested": {"count": 1, "enabled": True},
        },
    }


def test_provider_options_coerce_and_merge_nested_pydantic_settings():
    service = AIService(
        settings=NestedSettings(
            model="base-model",
            nested=NestedOptions(metadata={"base": True}),
        )
    )

    service.apply_provider_options(
        {
            "nested": {
                "enabled": True,
                "metadata": {"advanced": True},
            }
        }
    )

    assert isinstance(service._settings.nested, NestedOptions)
    assert service._settings.nested == NestedOptions(
        enabled=True,
        metadata={"base": True, "advanced": True},
    )


def test_provider_options_coerce_nested_pydantic_setting_from_none():
    service = AIService(
        settings=NestedSettings(
            model="base-model",
            nested=None,
        )
    )

    service.apply_provider_options({"nested": {"enabled": True}})

    assert service._settings.nested == NestedOptions(enabled=True)


def test_provider_options_preserve_unknown_nested_fields_on_wire_payload():
    service = AIService(
        settings=NestedSettings(
            model="base-model",
            nested=NestedOptions(),
        )
    )
    service.apply_provider_options(
        {
            "nested": {
                "enabled": True,
                "future_option": {"mode": "advanced"},
            }
        }
    )

    payload = service.merge_provider_options(
        {"nested": service._settings.nested.model_dump(mode="python")}
    )

    assert payload["nested"]["future_option"] == {"mode": "advanced"}


def test_cartesia_preserves_unknown_generation_options_in_wire_payload():
    from pipecat.services.cartesia.tts import CartesiaTTSService

    service = CartesiaTTSService(api_key="test-key")
    service._output_sample_rate = 24000
    service.apply_provider_options(
        {
            "generation_config": {
                "speed": 1.2,
                "future_option": {"mode": "advanced"},
            }
        }
    )

    payload = json.loads(service._build_msg(text="hello"))

    assert payload["generation_config"] == {
        "speed": 1.2,
        "future_option": {"mode": "advanced"},
    }


@pytest.mark.asyncio
async def test_deepgram_declared_model_option_reaches_websocket_url(monkeypatch):
    from pipecat.services.deepgram.tts import DeepgramTTSService

    websocket = SimpleNamespace(response=SimpleNamespace(headers={}))
    connect = AsyncMock(return_value=websocket)
    service = DeepgramTTSService(api_key="test-key")
    # 1.7.0 moved the module-level `websocket_connect` import behind
    # WebsocketService._websocket_connect, so patch the bound method.
    monkeypatch.setattr(service, "_websocket_connect", connect)
    service.apply_provider_options({"model": "future-deepgram-model"})

    await service._connect_websocket()

    url = connect.await_args.args[0]
    assert "model=future-deepgram-model" in url


def test_dograh_billing_metadata_replaces_non_object_provider_value():
    from pipecat.services.dograh.llm import DograhLLMService

    service = DograhLLMService(
        api_key="test-key",
        correlation_id="server-correlation",
    )
    service.apply_provider_options({"metadata": "not-an-object"})

    params = service.build_chat_completion_params({"messages": []})

    assert params["metadata"] == {
        "correlation_id": "server-correlation",
        "mps_billing_version": "2",
    }


def test_sarvam_explicit_advanced_options_survive_compatibility_cleanup():
    from pipecat.services.sarvam.llm import SarvamLLMService

    service = SarvamLLMService(api_key="test-key")
    service.apply_provider_options(
        {
            "stream_options": {"include_usage": False},
            "max_completion_tokens": 321,
            "service_tier": "priority",
        }
    )

    params = service.build_chat_completion_params({"messages": []})

    assert params["stream_options"] == {"include_usage": False}
    assert params["max_completion_tokens"] == 321
    assert params["service_tier"] == "priority"


# ---------------------------------------------------------------------------
# settings.extra reaches the wire: the chat completion, Responses and Gemini
# builders deep-merge it first, and the applied provider options merge over it.
# Each service is built the way an application configures one, through
# ``Settings(extra=...)``, never through ``apply_provider_options`` alone.
# ---------------------------------------------------------------------------


def _openai_service(extra, model="gpt-5-mini"):
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key="test-key",
        settings=OpenAILLMService.Settings(model=model, extra=extra),
    )


def test_settings_extra_reaches_the_chat_completion_request():
    service = _openai_service({"reasoning_effort": "minimal", "verbosity": "low"})

    params = service.build_chat_completion_params({"messages": []})

    assert params.get("reasoning_effort") == "minimal"
    assert params.get("verbosity") == "low"


def test_settings_extra_the_sdk_cannot_take_rides_extra_body():
    """DeepSeek's ``thinking`` is no parameter of the SDK method, so it is routed."""
    from pipecat.services.openai.llm import OpenAILLMService

    service = OpenAILLMService(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        settings=OpenAILLMService.Settings(
            model="deepseek-chat", temperature=0.1, extra={"thinking": {"type": "disabled"}}
        ),
    )

    params = service.build_chat_completion_params({"messages": []})

    assert "thinking" not in params
    assert params.get("extra_body", {}).get("thinking") == {"type": "disabled"}


def test_applied_option_wins_over_settings_extra():
    service = _openai_service({"reasoning_effort": "minimal", "verbosity": "low"})
    service.apply_provider_options({"verbosity": "medium"})

    params = service.build_chat_completion_params({"messages": []})

    assert (params.get("verbosity"), params.get("reasoning_effort")) == ("medium", "minimal")


def test_applied_nested_option_keeps_the_builders_sibling_keys():
    """``apply_provider_options`` also mirrors an undeclared option into
    ``settings.extra``. Applied there wholesale, a partial nested option would
    replace the builder's own value: here ``include_usage``, and with it the
    usage chunk the call's token count comes from."""
    service = _openai_service({})
    service.apply_provider_options({"stream_options": {"include_obfuscation": False}})

    params = service.build_chat_completion_params({"messages": []})

    assert params["stream_options"] == {"include_usage": True, "include_obfuscation": False}


def test_a_nested_settings_extra_value_is_not_shared_with_the_request():
    """A later edit of the request, as Sarvam's and Dograh's builders make in
    place, must not rewrite the stored settings."""
    service = _openai_service({"metadata": {"source": "profile"}})

    params = service.build_chat_completion_params({"messages": []})
    assert params.get("metadata") == {"source": "profile"}
    params["metadata"]["source"] = "mutated"

    assert service._settings.extra["metadata"] == {"source": "profile"}


def test_deepseek_service_sends_settings_extra():
    from pipecat.services.deepseek.llm import DeepSeekLLMService

    service = DeepSeekLLMService(
        api_key="test-key",
        settings=DeepSeekLLMService.Settings(model="deepseek-chat", extra={"logprobs": True}),
    )

    params = service.build_chat_completion_params({"messages": []})

    assert params.get("logprobs") is True


def test_deepseek_service_routes_applied_options_through_the_base_builder():
    from pipecat.services.deepseek.llm import DeepSeekLLMService

    service = DeepSeekLLMService(api_key="test-key")
    service.apply_provider_options({"thinking": {"type": "enabled"}})

    params = service.build_chat_completion_params({"messages": []})

    assert "thinking" not in params
    assert params["extra_body"] == {"thinking": {"type": "enabled"}}


def test_responses_builder_sends_settings_extra():
    from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService

    service = OpenAIResponsesHttpLLMService(
        api_key="test-key",
        settings=OpenAIResponsesHttpLLMService.Settings(
            model="gpt-4.1", extra={"truncation": "auto"}
        ),
    )

    params = service._build_response_params({"input": []})

    assert params.get("truncation") == "auto"


def _gemini_service(extra):
    from pipecat.services.google.llm import GoogleLLMService

    return GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(model="gemini-2.5-flash", extra=extra),
    )


def test_gemini_builder_sends_settings_extra():
    service = _gemini_service({"response_mime_type": "text/plain"})

    params = service._build_generation_params()

    assert params.get("response_mime_type") == "text/plain"


def test_gemini_thinking_config_in_settings_extra_beats_the_low_latency_default():
    service = _gemini_service({"thinking_config": {"thinking_budget": 1024}})

    params = service._build_generation_params()

    assert params["thinking_config"] == {"thinking_budget": 1024}
