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
there. Sending several windows at once must not hide a speech start or stop
from the controller, whatever the packet size.
"""

import random
import unittest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


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


WINDOW = 512 * 2  # bytes in one 32 ms model window at 16 kHz
VOICED = b"\x01" + b"\x00" * (WINDOW - 1)
QUIET = b"\x00" * WINDOW


class ScriptedVADAnalyzer(VADAnalyzer):
    """The real analyzer with a scripted model: a window is speech when its first byte is not 0.

    start_secs and stop_secs are 0.2 s, six 32 ms windows each; min_volume 0
    lets the script alone decide speech.
    """

    def __init__(self):
        super().__init__(sample_rate=16000, params=VADParams(stop_secs=0.2, min_volume=0.0))

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0 if buffer[0] else 0.0


def _seeded_stream(seed: int = 7, windows: int = 3000) -> bytes:
    """Alternating quiet and voiced runs of 1 to 20 windows each, from a fixed seed."""
    rng = random.Random(seed)
    runs: list[bytes] = []
    voiced = False
    while len(runs) < windows:
        runs.extend([VOICED if voiced else QUIET] * rng.randint(1, 20))
        voiced = not voiced
    return b"".join(runs[:windows])


def _packets(stream: bytes, size: int) -> list[bytes]:
    return [stream[pos : pos + size] for pos in range(0, len(stream), size)]


async def _edges(packets: list[bytes]) -> list[tuple[str, int]]:
    """Each speech start and stop a real VADController reports, with the packet it came in."""
    controller = VADController(ScriptedVADAnalyzer(), audio_idle_timeout=0)
    edges: list[tuple[str, int]] = []
    index = 0

    @controller.event_handler("on_speech_started")
    async def on_speech_started(_controller):
        edges.append(("started", index))

    @controller.event_handler("on_speech_stopped")
    async def on_speech_stopped(_controller):
        edges.append(("stopped", index))

    await controller.setup(frame_processor_setup(TaskManager()))
    try:
        for index, packet in enumerate(packets):
            await controller.process_frame(
                InputAudioRawFrame(audio=packet, sample_rate=16000, num_channels=1)
            )
    finally:
        await controller.cleanup()
    return edges


class TestEveryEdgeReachesTheController(unittest.IsolatedAsyncioTestCase):
    """A packet that completes several windows still reports each start and stop."""

    async def test_speech_start_one_window_per_packet(self):
        edges = await _edges([VOICED] * 7 + [QUIET])
        self.assertEqual([kind for kind, _ in edges], ["started"])

    async def test_speech_start_in_one_packet_with_a_trailing_quiet_window(self):
        edges = await _edges([VOICED * 7 + QUIET])
        self.assertEqual(
            [kind for kind, _ in edges], ["started"], "seven voiced windows then one quiet"
        )

    async def test_speech_stop_in_one_packet_with_a_trailing_voiced_window(self):
        edges = await _edges([VOICED] * 7 + [QUIET * 6 + VOICED])
        self.assertEqual(
            [kind for kind, _ in edges], ["started", "stopped"], "six quiet windows then one voiced"
        )

    async def test_packet_size_does_not_change_the_edges(self):
        """Any packet size reports the edges of one window per packet, each in the packet completing it.

        A seeded stream of 3,000 windows, in 20 ms (640 B), 100 ms (3,200 B)
        and 0.5 s (16,000 B) packets, reports the starts and stops the same
        stream reports one window per packet, each in the packet that holds the
        last byte of the window that completes it.
        """
        stream = _seeded_stream()
        reference = await _edges(_packets(stream, WINDOW))
        self.assertGreater(len(reference), 100)
        for size in (640, 3200, 16000):
            expected = [(kind, ((window + 1) * WINDOW - 1) // size) for kind, window in reference]
            self.assertEqual(await _edges(_packets(stream, size)), expected, f"{size} B packets")


if __name__ == "__main__":
    unittest.main()
