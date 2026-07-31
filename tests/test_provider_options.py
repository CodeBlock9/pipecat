import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic import Field as PydanticField

from pipecat.services.ai_service import AIService
from pipecat.services.settings import NOT_GIVEN, ServiceSettings, _NotGiven


class NestedOptions(BaseModel):
    enabled: bool = False
    metadata: dict = PydanticField(default_factory=dict)


@dataclass
class NestedSettings(ServiceSettings):
    nested: NestedOptions | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


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
    monkeypatch.setattr("pipecat.services.deepgram.tts.websocket_connect", connect)
    service = DeepgramTTSService(api_key="test-key")
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
