#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How SoundfileMixer fills an output chunk from a clip that does not divide into chunks.

The real mixer loads a real WAV written with soundfile. Mixing into silence at
volume 1.0 makes the output exactly the clip samples the mixer chose.
"""

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer  # noqa: E402

RATE = 16_000


async def _mixer(tmp_path, samples, *, loop=True) -> SoundfileMixer:
    path = tmp_path / "clip.wav"
    sf.write(str(path), np.array(samples, dtype=np.int16), RATE, subtype="PCM_16")
    mixer = SoundfileMixer(
        sound_files={"clip": str(path)}, default_sound="clip", volume=1.0, loop=loop
    )
    await mixer.start(RATE)
    return mixer


def _silence(n: int) -> bytes:
    return np.zeros(n, dtype=np.int16).tobytes()


def _samples(audio: bytes) -> list[int]:
    return np.frombuffer(audio, dtype=np.int16).tolist()


@pytest.mark.asyncio
async def test_a_clip_shorter_than_a_chunk_is_looped_into_the_chunk(tmp_path):
    mixer = await _mixer(tmp_path, list(range(1, 9)))  # 8 samples

    mixed = await mixer.mix(_silence(320))  # one 20 ms chunk

    assert _samples(mixed) == [(i % 8) + 1 for i in range(320)]


@pytest.mark.asyncio
async def test_a_looping_clip_keeps_its_tail(tmp_path):
    mixer = await _mixer(tmp_path, [1, 2, 3, 4, 5])

    chunks = [_samples(await mixer.mix(_silence(3))) for _ in range(3)]

    assert chunks == [[1, 2, 3], [4, 5, 1], [2, 3, 4]]


@pytest.mark.asyncio
async def test_a_long_clip_is_mixed_in_order(tmp_path):
    mixer = await _mixer(tmp_path, list(range(1, 11)))

    assert _samples(await mixer.mix(_silence(4))) == [1, 2, 3, 4]
    assert _samples(await mixer.mix(_silence(4))) == [5, 6, 7, 8]


@pytest.mark.asyncio
async def test_a_clip_that_does_not_loop_plays_its_tail_then_stops(tmp_path):
    mixer = await _mixer(tmp_path, [1, 2, 3, 4, 5], loop=False)

    assert _samples(await mixer.mix(_silence(3))) == [1, 2, 3]
    assert _samples(await mixer.mix(_silence(3))) == [4, 5, 0]
    assert _samples(await mixer.mix(_silence(3))) == [0, 0, 0]


@pytest.mark.asyncio
async def test_an_empty_clip_leaves_the_audio_unchanged(tmp_path):
    mixer = await _mixer(tmp_path, [])
    audio = np.array([7, -7, 1000], dtype=np.int16).tobytes()

    assert await mixer.mix(audio) == audio
