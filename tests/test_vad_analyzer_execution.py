#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a VAD analyzer reaches its inference thread, and gives it back.

One analyzer is built per call. What it does with a thread therefore multiplies
by concurrency, and what it fails to release multiplies by every call the
process has ever taken.
"""

import asyncio
import unittest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState


class StubVADAnalyzer(VADAnalyzer):
    """A VAD analyzer with a counter where the model would be.

    The real Silero analyzer loads an ONNX session; nothing here is about the
    model, only about the thread the model would run on and how often it is
    handed work.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.confidence_calls = 0
        self.confidence = 0.0

    def num_frames_required(self) -> int:
        return 256

    def voice_confidence(self, buffer: bytes) -> float:
        self.confidence_calls += 1
        return self.confidence


def _analyzer(**kwargs) -> StubVADAnalyzer:
    analyzer = StubVADAnalyzer(params=VADParams(min_volume=0.0), **kwargs)
    analyzer.set_sample_rate(8000)
    return analyzer


class TestExecutorLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_releases_the_inference_thread(self):
        analyzer = _analyzer()
        self.assertFalse(analyzer._executor._shutdown)

        await analyzer.cleanup()

        # Without this the thread stays parked on its work queue for the life
        # of the process: one leaked thread per call.
        self.assertTrue(analyzer._executor._shutdown)

    async def test_cleanup_drops_the_buffered_audio(self):
        analyzer = _analyzer()
        await analyzer.analyze_audio(b"\x01\x00" * 100)

        await analyzer.cleanup()

        self.assertEqual(analyzer._vad_buffer, b"")

    async def test_each_analyzer_owns_its_own_thread(self):
        """One analyzer per call, one audio stream per analyzer."""
        first = _analyzer()
        second = _analyzer()
        try:
            self.assertIsNot(first._executor, second._executor)
        finally:
            await first.cleanup()
            await second.cleanup()


class TestAnalysisStillWorks(unittest.IsolatedAsyncioTestCase):
    async def test_a_full_window_of_loud_speech_starts_a_turn(self):
        analyzer = _analyzer()
        analyzer.confidence = 1.0
        try:
            state = VADState.QUIET
            # start_secs 0.2 at 256 frames of 8 kHz is about six windows.
            for _ in range(10):
                state = await analyzer.analyze_audio(bytes(512))
            self.assertEqual(state, VADState.SPEAKING)
        finally:
            await analyzer.cleanup()

    async def test_silence_keeps_the_analyzer_quiet(self):
        analyzer = _analyzer()
        analyzer.confidence = 0.0
        try:
            state = await analyzer.analyze_audio(bytes(512 * 10))
            self.assertEqual(state, VADState.QUIET)
        finally:
            await analyzer.cleanup()


if __name__ == "__main__":
    unittest.main()
