#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a VAD analyzer reaches the model, and how often.

One analyzer is built per call, so whatever it holds multiplies by concurrency
and whatever work it does per chunk multiplies by 50 a second per call. Two
things follow: inference runs on a pool shared by the whole process, and audio
is accumulated into whole windows on the event loop before any of it is sent
there.
"""

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
    async def test_analyzers_share_one_inference_pool(self):
        """A pool each meant one operating-system thread per concurrent call."""
        first = _analyzer()
        second = _analyzer()

        self.assertIs(first._executor, second._executor)

    async def test_cleanup_drops_the_buffered_audio(self):
        analyzer = _analyzer()
        await analyzer.analyze_audio(b"\x01\x00" * 100)

        await analyzer.cleanup()

        self.assertEqual(analyzer._vad_buffer, b"")

    async def test_cleanup_leaves_the_pool_usable_for_other_calls(self):
        """Shutting the shared pool down would stop VAD for every other call."""
        first = _analyzer()
        second = _analyzer()
        second.confidence = 1.0

        await first.cleanup()

        self.assertEqual(await second.analyze_audio(bytes(512)), VADState.STARTING)


class TestWindowBuffering(unittest.IsolatedAsyncioTestCase):
    async def test_a_short_chunk_does_not_reach_the_model(self):
        """20 ms chunks against a 32 ms window: a third used to be round trips."""
        analyzer = _analyzer()

        state = await analyzer.analyze_audio(bytes(320))

        self.assertEqual(analyzer.confidence_calls, 0)
        self.assertEqual(state, VADState.QUIET)
        self.assertEqual(len(analyzer._vad_buffer), 320)

    async def test_a_whole_window_is_analyzed_and_the_remainder_kept(self):
        analyzer = _analyzer()

        await analyzer.analyze_audio(bytes(320))
        await analyzer.analyze_audio(bytes(320))

        self.assertEqual(analyzer.confidence_calls, 1)
        self.assertEqual(len(analyzer._vad_buffer), 128)

    async def test_several_windows_at_once_are_analyzed_in_order(self):
        analyzer = _analyzer()

        await analyzer.analyze_audio(bytes(512 * 3 + 64))

        self.assertEqual(analyzer.confidence_calls, 3)
        self.assertEqual(len(analyzer._vad_buffer), 64)


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
