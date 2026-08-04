#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for xAI realtime audio session configuration."""

import json

import pytest
from pydantic import ValidationError

from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


def test_audio_output_speed_round_trips_through_session_properties():
    session = events.SessionProperties.model_validate({"audio": {"output": {"speed": 1.2}}})

    assert session.model_dump(exclude_none=True)["audio"]["output"]["speed"] == 1.2


def test_audio_output_speed_preserves_transport_sample_rates():
    service = GrokRealtimeLLMService(
        api_key="test-key",
        settings=GrokRealtimeLLMService.Settings(
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(
                    output=events.AudioOutput(speed=1.2),
                )
            )
        ),
    )

    service._ensure_audio_config(input_sample_rate=8000, output_sample_rate=16000)

    audio = service._settings.session_properties.audio
    assert audio.input.format.rate == 8000
    assert audio.output.format.rate == 16000
    assert audio.output.speed == 1.2


@pytest.mark.parametrize("speed", [0.69, 1.51])
def test_audio_output_speed_rejects_values_outside_xai_range(speed: float):
    with pytest.raises(ValidationError):
        events.SessionProperties.model_validate({"audio": {"output": {"speed": speed}}})


def test_session_created_event_is_parsed_as_known_lifecycle_event():
    event = events.parse_server_event(
        json.dumps(
            {
                "event_id": "event-1",
                "type": "session.created",
                "session": {"id": "session-1"},
            }
        )
    )

    assert isinstance(event, events.SessionCreatedEvent)
