#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for the short-window volume gate used only by VAD."""

import audioop
import pathlib
import wave

import numpy as np
import pyloudnorm as pyln

from pipecat.audio.utils import normalize_value
from pipecat.audio.vad.vad_analyzer import _calculate_vad_audio_volume

SPEECH_WAV = (
    pathlib.Path(__file__).parents[1]
    / "src"
    / "pipecat"
    / "services"
    / "aws"
    / "nova_sonic"
    / "ready.wav"
)
MIN_VOLUME = 0.6
BOUNDARY_MARGIN = 0.05


def _ebu_r128_volume(audio: bytes, sample_rate: int) -> float:
    samples = np.frombuffer(audio, dtype=np.int16)
    meter = pyln.Meter(sample_rate, block_size=samples.size / sample_rate)
    return normalize_value(meter.integrated_loudness(samples.astype(np.float64)), -20, 80)


def _corpus():
    with wave.open(str(SPEECH_WAV)) as handle:
        source_pcm = handle.readframes(handle.getnframes())
        source_rate = handle.getframerate()
    rng = np.random.default_rng(7)
    for sample_rate, window in ((8000, 256), (16000, 512)):
        pcm = source_pcm
        if source_rate != sample_rate:
            pcm, _ = audioop.ratecv(pcm, 2, 1, source_rate, sample_rate, None)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        for gain_db in (-40, -30, -24, -20, -18, -16, -14, -12, -6, 0, 6):
            scaled = np.clip(samples * 10 ** (gain_db / 20), -32768, 32767).astype(np.int16)
            raw = scaled.tobytes()
            for start in range(0, len(raw) - window * 2, window * 2):
                yield sample_rate, raw[start : start + window * 2]
        for amplitude in (0, 1, 10, 50, 100, 300, 1000, 3000, 10000):
            raw = (
                (rng.standard_normal(window * 40) * amplitude)
                .clip(-32768, 32767)
                .astype(np.int16)
                .tobytes()
            )
            for start in range(0, len(raw) - window * 2, window * 2):
                yield sample_rate, raw[start : start + window * 2]


def test_short_window_gate_matches_previous_measurement_away_from_boundary():
    decided = 0
    for sample_rate, chunk in _corpus():
        old = _ebu_r128_volume(chunk, sample_rate)
        if abs(old - MIN_VOLUME) <= BOUNDARY_MARGIN:
            continue
        decided += 1
        assert (_calculate_vad_audio_volume(chunk) >= MIN_VOLUME) == (old >= MIN_VOLUME)
    assert decided > 500


def test_short_window_volume_handles_silence_empty_and_loud_audio():
    assert _calculate_vad_audio_volume(b"") == 0.0
    assert _calculate_vad_audio_volume(b"\x00" * 512) == 0.0
    loud = np.full(256, 32000, dtype=np.int16).tobytes()
    assert _calculate_vad_audio_volume(loud) > MIN_VOLUME
