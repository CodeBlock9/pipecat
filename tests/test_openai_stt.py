#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace

import httpx
import pytest
from openai.types.audio import Transcription

from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.openai.stt import (
    OpenAIRealtimeSTTService,
    OpenAISTTService,
    OpenAISTTSettings,
)
from pipecat.tests.utils import run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

SAMPLE_RATE = 16000


def _capture_transcription_request(service):
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return Transcription(text="hello world")

    service._client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    return captured


@pytest.mark.asyncio
async def test_gpt_transcribe_sends_plural_context_through_extra_body():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(
            model="gpt-transcribe",
            prompt="Dograh is a product name.",
            keywords=["Dograh", "Pipecat"],
            languages=["en", "fr"],
        ),
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert captured["model"] == "gpt-transcribe"
    assert captured["prompt"] == "Dograh is a product name."
    assert "language" not in captured
    assert "languages" not in captured
    assert "keywords" not in captured
    assert captured["extra_body"] == {
        "languages": ["en", "fr"],
        "keywords": ["Dograh", "Pipecat"],
    }


@pytest.mark.asyncio
async def test_gpt_transcribe_context_survives_sdk_multipart_encoding():
    captured = {}

    async def handler(request):
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = await request.aread()
        return httpx.Response(200, json={"text": "hello world"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = OpenAISTTService(
            api_key="test-key",
            http_client=http_client,
            settings=OpenAISTTSettings(
                model="gpt-transcribe",
                languages=["en", "fr"],
                keywords=["Dograh"],
            ),
        )
        await service._transcribe(b"wav-bytes")

    assert captured["content_type"].startswith("multipart/form-data")
    body = captured["body"]
    assert body.count(b'name="languages[]"') == 2
    assert b'name="keywords[]"' in body
    assert all(value in body for value in (b"en", b"fr", b"Dograh"))


@pytest.mark.asyncio
async def test_gpt_transcribe_omits_empty_context_and_probability_options():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(
            model="gpt-transcribe",
            keywords=[],
            languages=[],
        ),
        include_prob_metrics=True,
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert "language" not in captured
    assert "extra_body" not in captured
    assert "include" not in captured
    assert "response_format" not in captured


@pytest.mark.asyncio
async def test_gpt_transcribe_sanitizes_language_after_provider_options_merge():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(
            model="gpt-transcribe",
            keywords=["Dograh"],
            languages=["en"],
        ),
    )
    service.apply_provider_options(
        {
            "language": "de",
            "languages": ["fr"],
            "extra_body": {
                "language": "es",
                "custom": {"enabled": True},
            },
        }
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert "language" not in captured
    assert captured["extra_body"] == {
        "languages": ["fr"],
        "keywords": ["Dograh"],
        "custom": {"enabled": True},
    }


@pytest.mark.asyncio
async def test_gpt_transcribe_uses_effective_model_after_provider_options():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(model="gpt-4o-transcribe"),
    )
    service.apply_provider_options(
        {
            "model": "gpt-transcribe-2026-09-01",
            "languages": ["en"],
        }
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert captured["model"] == "gpt-transcribe-2026-09-01"
    assert "language" not in captured
    assert captured["extra_body"] == {"languages": ["en"]}


@pytest.mark.asyncio
async def test_gpt_transcribe_context_can_be_updated_at_runtime():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(model="gpt-4o-transcribe"),
    )
    await service._update_settings(
        OpenAISTTSettings(
            model="gpt-transcribe",
            keywords=["Dograh"],
            languages=["en"],
        )
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert captured["model"] == "gpt-transcribe"
    assert "language" not in captured
    assert captured["extra_body"] == {
        "languages": ["en"],
        "keywords": ["Dograh"],
    }


@pytest.mark.asyncio
async def test_gpt_4o_transcribe_preserves_singular_language_and_logprobs():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTSettings(model="gpt-4o-transcribe", language="fr"),
        include_prob_metrics=True,
    )
    captured = _capture_transcription_request(service)

    await service._transcribe(b"wav-bytes")

    assert captured["language"] == "fr"
    assert captured["response_format"] == "json"
    assert captured["include"] == ["logprobs"]
    assert "extra_body" not in captured


@pytest.mark.asyncio
async def test_segment_emits_usage_and_transcription(monkeypatch):
    service = OpenAISTTService(api_key="test-key")

    async def fake_transcribe(audio: bytes) -> Transcription:
        return Transcription(text="hello world")

    monkeypatch.setattr(service, "_transcribe", fake_transcribe)

    pcm = b"\x01\x02" * SAMPLE_RATE  # 1s of 16-bit mono audio
    received_down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    transcripts = [f for f in received_down if isinstance(f, TranscriptionFrame)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "hello world"
    assert transcripts[0].finalized is True

    usage_indexes = [
        i
        for i, f in enumerate(received_down)
        if isinstance(f, MetricsFrame) and any(isinstance(d, STTUsageMetricsData) for d in f.data)
    ]
    assert len(usage_indexes) == 1
    usage_frame = received_down[usage_indexes[0]]
    usage = next(d for d in usage_frame.data if isinstance(d, STTUsageMetricsData))
    assert usage.value.audio_seconds == pytest.approx(len(pcm) / (SAMPLE_RATE * 2))

    # Usage precedes the transcript so tracing attaches it to the span the
    # finalized TranscriptionFrame closes.
    assert usage_indexes[0] < received_down.index(transcripts[0])


def test_openai_realtime_should_interrupt_rides_on_recommended_strategies():
    # should_interrupt configures the strategies the service recommends via its
    # metadata frame; the service never broadcasts the interruption itself.
    for should_interrupt in (True, False):
        service = OpenAIRealtimeSTTService(
            api_key="test-key",
            turn_detection={"type": "server_vad"},
            should_interrupt=should_interrupt,
        )
        strategies = service.service_metadata_frame().user_turn_strategies
        assert isinstance(strategies, ExternalUserTurnStrategies)
        assert strategies.enable_interruptions is should_interrupt


def test_openai_realtime_server_defaults_recommend_strategies():
    """``turn_detection=None`` omits the field, so the session's own default stands.

    That default detects turns, so the recommendation applies just as it does
    for an explicit configuration.
    """
    service = OpenAIRealtimeSTTService(api_key="test-key", turn_detection=None)
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)


def test_openai_realtime_local_vad_mode_recommends_no_strategies():
    """With turn detection off the server reports no boundaries to propose."""
    service = OpenAIRealtimeSTTService(api_key="test-key")
    assert service.service_metadata_frame().user_turn_strategies is None
