#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The two ONNX models are loaded once per process, not once per analyzer.

A server that builds one analyzer per call was paying a fresh ~9 MiB Silero
session and a fresh ~28 MiB smart-turn session for every call, and ONNX
Runtime does not return an arena to the operating system when the session is
destroyed -- so the process kept its worst-ever concurrency resident for good.
Both sessions are read-only graphs whose ``run()`` is thread-safe; everything
that varies per audio stream lives on the analyzer.
"""

import asyncio
import unittest


class TestSileroSessionSharing(unittest.TestCase):
    def test_analyzers_share_one_session_but_not_their_state(self):
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        first = SileroVADAnalyzer()
        second = SileroVADAnalyzer()

        self.assertIs(first._model.session, second._model.session)
        # Per-stream state must stay separate or two calls would corrupt each
        # other's speech detection.
        self.assertIsNot(first._model._state, second._model._state)
        self.assertIsNot(first._model._context, second._model._context)

    def test_a_second_analyzer_adds_no_session(self):
        from pipecat.audio.vad.silero import SileroOnnxModel, SileroVADAnalyzer

        SileroVADAnalyzer()
        before = len(SileroOnnxModel._session_cache)
        SileroVADAnalyzer()
        self.assertEqual(len(SileroOnnxModel._session_cache), before)


class TestSmartTurnSessionSharing(unittest.TestCase):
    def test_analyzers_share_one_session_but_not_their_executor(self):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        first = LocalSmartTurnAnalyzerV3()
        second = LocalSmartTurnAnalyzerV3()

        self.assertIs(first._session, second._session)
        # One inference thread per audio stream: sharing it would serialise
        # end-of-turn detection across concurrent calls.
        self.assertIsNot(first._executor, second._executor)

    def test_cleanup_releases_the_inference_thread(self):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        analyzer = LocalSmartTurnAnalyzerV3()
        self.assertFalse(analyzer._executor._shutdown)

        asyncio.run(analyzer.cleanup())

        # Without this the thread stays parked on its work queue for the life
        # of the process, one leaked thread per call.
        self.assertTrue(analyzer._executor._shutdown)
        self.assertEqual(analyzer._audio_buffer, [])

    def test_the_arena_is_shrunk_after_every_run(self):
        """The pool is sized by peak overlap and never handed back on its own.

        Smart turn runs about once every eight seconds per call and takes a
        quarter of a second, so simultaneous inference is rare -- and whatever
        a rare overlap grows the arena to is resident for the life of the
        process unless each run gives its blocks back.
        """
        import time

        import numpy as np

        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        analyzer = LocalSmartTurnAnalyzerV3()
        options = analyzer._shrinking_run_options()
        # Shared, and built once: the entry is per run, but the object is not.
        self.assertIs(options, LocalSmartTurnAnalyzerV3()._shrinking_run_options())

        captured = {}
        real_run = analyzer._session.run

        def recording(output_names, input_feed, run_options=None):
            captured["run_options"] = run_options
            return real_run(output_names, input_feed, run_options)

        analyzer._session.run = recording
        try:
            buffer = [(time.time(), np.zeros(1600, dtype=np.float32)) for _ in range(10)]
            analyzer._process_speech_segment(buffer)
        finally:
            del analyzer._session.run

        self.assertIs(captured["run_options"], options)

    def test_cleanup_leaves_the_shared_session_usable(self):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        first = LocalSmartTurnAnalyzerV3()
        second = LocalSmartTurnAnalyzerV3()
        asyncio.run(first.cleanup())

        # One call ending must not disturb the calls still running.
        self.assertIsNotNone(second._session)
        self.assertFalse(second._executor._shutdown)


if __name__ == "__main__":
    unittest.main()
