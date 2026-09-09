#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The volume the VAD gate reads still decides the same way.

``calculate_audio_volume`` exists for one consumer: ``VADAnalyzer`` smooths it
and compares it against ``min_volume``, default 0.6. It used to build a fresh
``pyln.Meter`` per call and run a single-block EBU R128 integrated-loudness
measurement over un-normalised int16 — a filter bank, a gating pass and a
scipy import, 50 times a second per call, to produce one number that is then
compared against one threshold.

The replacement is the same formula without the K-weighting: BS.1770's
``-0.691 + 20*log10(rms)``, normalised over the same ``-20..80`` range, so the
scale and therefore the 0.6 threshold are unchanged. What K-weighting was
contributing is a few dB of spectral tilt, and this holds the two against each
other over recorded speech and noise to say how few.
"""

import audioop
import math
import pathlib
import wave

import numpy as np
import pyloudnorm as pyln

from pipecat.audio.utils import calculate_audio_volume, normalize_value

#: Half a second of recorded speech that ships with the fork.
SPEECH_WAV = (
    pathlib.Path(__file__).parents[1]
    / "src"
    / "pipecat"
    / "services"
    / "aws"
    / "nova_sonic"
    / "ready.wav"
)

#: The VAD's default ``min_volume``: the only threshold this function feeds.
MIN_VOLUME = 0.6

#: How close to the threshold the two measurements are allowed to disagree.
#: The normalised scale spans 100 dB, so 0.05 is 5 dB. Measured worst case is
#: 3.2 dB, all of it white noise sitting exactly on the gate.
BOUNDARY_MARGIN = 0.05


def ebu_r128_volume(audio: bytes, sample_rate: int) -> float:
    """The measurement this replaced, kept as the reference to compare against.

    Deliberately a copy rather than an import: the point of the change is that
    nothing in ``src/`` reaches pyloudnorm, and therefore scipy, any more.
    """
    samples = np.frombuffer(audio, dtype=np.int16)
    meter = pyln.Meter(sample_rate, block_size=samples.size / sample_rate)
    loudness = meter.integrated_loudness(samples.astype(np.float64))
    return normalize_value(loudness, -20, 80)


def _speech_chunks(sample_rate: int, window: int) -> list[tuple[str, bytes]]:
    """Recorded speech at a range of gains, cut into VAD-sized windows."""
    with wave.open(str(SPEECH_WAV)) as handle:
        pcm = handle.readframes(handle.getnframes())
        source_rate = handle.getframerate()
    if sample_rate != source_rate:
        pcm, _ = audioop.ratecv(pcm, 2, 1, source_rate, sample_rate, None)

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    chunks = []
    for gain_db in (-40, -30, -24, -20, -18, -16, -14, -12, -6, 0, 6):
        scaled = np.clip(samples * 10 ** (gain_db / 20.0), -32768, 32767)
        raw = scaled.astype(np.int16).tobytes()
        for start in range(0, len(raw) - window * 2, window * 2):
            chunks.append((f"speech {gain_db:+d} dB", raw[start : start + window * 2]))
    return chunks


def _noise_chunks(window: int) -> list[tuple[str, bytes]]:
    """White noise from silence up to full scale."""
    rng = np.random.default_rng(7)
    chunks = []
    for amplitude in (0, 1, 10, 50, 100, 300, 1000, 3000, 10000):
        raw = (
            (rng.standard_normal(window * 40) * amplitude)
            .clip(-32768, 32767)
            .astype(np.int16)
            .tobytes()
        )
        for start in range(0, len(raw) - window * 2, window * 2):
            chunks.append((f"noise {amplitude}", raw[start : start + window * 2]))
    return chunks


def _corpus() -> list[tuple[int, str, bytes]]:
    """Both sample rates a call runs at, at their Silero window sizes."""
    corpus = []
    for sample_rate, window in ((16000, 512), (8000, 256)):
        for label, chunk in _speech_chunks(sample_rate, window) + _noise_chunks(window):
            corpus.append((sample_rate, label, chunk))
    return corpus


def test_the_gate_decides_the_same_way_away_from_its_own_threshold():
    """Every chunk the old measurement placed clear of the gate lands the same side."""
    decided = 0
    for sample_rate, label, chunk in _corpus():
        old = ebu_r128_volume(chunk, sample_rate)
        if abs(old - MIN_VOLUME) <= BOUNDARY_MARGIN:
            continue
        decided += 1
        new = calculate_audio_volume(chunk, sample_rate)
        assert (new >= MIN_VOLUME) == (old >= MIN_VOLUME), (
            f"{label} at {sample_rate} Hz: EBU R128 {old:.4f}, RMS {new:.4f} — "
            f"the two disagree about the {MIN_VOLUME} gate {abs(old - MIN_VOLUME) * 100:.1f} dB away from it"
        )
    assert decided > 500, decided


def test_the_two_measurements_stay_within_the_margin_everywhere():
    """A disagreement is only tolerable because the gap itself is small."""
    worst = 0.0
    for sample_rate, label, chunk in _corpus():
        old = ebu_r128_volume(chunk, sample_rate)
        new = calculate_audio_volume(chunk, sample_rate)
        worst = max(worst, abs(new - old))
    assert worst <= 0.07, f"largest gap {worst:.4f} on the normalised scale"


def test_silence_and_full_scale_sit_at_the_ends_of_the_scale():
    assert calculate_audio_volume(b"\x00" * 512, 8000) == 0.0
    loud = (np.full(256, 32000, dtype=np.int16)).tobytes()
    assert calculate_audio_volume(loud, 8000) > MIN_VOLUME


def test_an_empty_buffer_is_silence_rather_than_an_error():
    """A zero-length chunk reaches here from a transport that sent nothing."""
    assert calculate_audio_volume(b"", 8000) == 0.0


def test_the_scale_matches_the_formula_it_documents():
    samples = np.full(1000, 1000, dtype=np.int16)
    expected = normalize_value(20.0 * math.log10(1000.0) - 0.691, -20, 80)
    assert calculate_audio_volume(samples.tobytes(), 8000) == expected
