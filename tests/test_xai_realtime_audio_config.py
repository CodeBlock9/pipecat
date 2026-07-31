#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for xAI realtime audio session configuration."""

import pytest
from pydantic import ValidationError

from pipecat.services.xai.realtime import events


def test_audio_output_speed_round_trips_through_session_properties():
    session = events.SessionProperties.model_validate({"audio": {"output": {"speed": 1.2}}})

    assert session.model_dump(exclude_none=True)["audio"]["output"]["speed"] == 1.2


@pytest.mark.parametrize("speed", [0.69, 1.51])
def test_audio_output_speed_rejects_values_outside_xai_range(speed: float):
    with pytest.raises(ValidationError):
        events.SessionProperties.model_validate({"audio": {"output": {"speed": speed}}})
