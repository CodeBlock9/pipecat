#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVIObserver's frame dedup and its bot-output delivery.

The real RTVIObserver, driven through ``on_push_frame`` with ``FramePushed``
events as the worker delivers them, and ``send_rtvi_message`` captured. A
spec'd mock stands in for the output transport, because the observer only acts
on aggregated text pushed by one.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    BotStartedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import BotOutputTransformResult
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.text.base_text_aggregator import AggregationType

OUTPUT = MagicMock(spec=BaseOutputTransport)
OTHER = MagicMock(spec=FrameProcessor)


def _observer(**params) -> tuple[RTVIObserver, list]:
    observer = RTVIObserver(params=RTVIObserverParams(**params))
    sent: list = []
    observer.send_rtvi_message = AsyncMock(side_effect=lambda m, *a, **k: sent.append(m))
    return observer, sent


async def _push(observer, frame, source=OUTPUT):
    await observer.on_push_frame(
        FramePushed(
            source=source,
            destination=OTHER,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


def _aggregate(text: str, *, spoken: bool) -> AggregatedTextFrame:
    frame = AggregatedTextFrame(text=text, aggregated_by=AggregationType.SENTENCE)
    frame.will_be_spoken = spoken
    return frame


def _bot_output(sent) -> list[str]:
    return [m.data.text for m in sent if isinstance(m, RTVI.BotOutputMessage)]


class TestFramesSeen(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_audio_levels_leave_no_trace(self):
        """15,000 audio frames with the level messages off: no message, and no id kept."""
        observer, sent = _observer()  # user and bot audio levels are off by default
        for _ in range(15_000):
            await _push(
                observer,
                InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1),
                source=OTHER,
            )
        self.assertEqual(sent, [])
        self.assertEqual(len(observer._frames_seen), 0)

    async def test_the_dedup_is_bounded(self):
        observer, sent = _observer(user_audio_level_enabled=True, audio_level_period_secs=3600)
        for _ in range(15_000):
            await _push(
                observer,
                InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1),
                source=OTHER,
            )
        self.assertLessEqual(len(observer._frames_seen), 10_000)

    async def test_a_frame_seen_twice_is_handled_once(self):
        observer, sent = _observer()
        frame = _aggregate("Hello there.", spoken=True)
        await _push(observer, BotStartedSpeakingFrame())
        await _push(observer, frame)
        await _push(observer, frame)
        self.assertEqual(_bot_output(sent), ["Hello there."])


class TestBotOutputDelivery(unittest.IsolatedAsyncioTestCase):
    async def test_nonspoken_output_is_sent_without_waiting_for_speech(self):
        observer, sent = _observer()
        await _push(observer, _aggregate("Your booking reference is 4521.", spoken=False))
        self.assertEqual(_bot_output(sent), ["Your booking reference is 4521."])

    async def test_output_is_sent_when_bot_speaking_messages_are_off(self):
        observer, sent = _observer(bot_speaking_enabled=False)
        await _push(observer, BotStartedSpeakingFrame())
        await _push(observer, _aggregate("Hello there.", spoken=True))
        self.assertEqual(_bot_output(sent), ["Hello there."])

    async def test_an_interruption_drops_queued_output(self):
        observer, sent = _observer()
        await _push(observer, _aggregate("A sentence that was cut off.", spoken=True))
        await _push(observer, InterruptionFrame())
        await _push(observer, BotStartedSpeakingFrame())
        self.assertEqual(_bot_output(sent), [])

    async def test_spoken_output_waits_for_the_bot_to_start(self):
        observer, sent = _observer()
        await _push(observer, _aggregate("Hello there.", spoken=True))
        self.assertEqual(_bot_output(sent), [])
        await _push(observer, BotStartedSpeakingFrame())
        self.assertEqual(_bot_output(sent), ["Hello there."])


class TestProgressTransforms(unittest.IsolatedAsyncioTestCase):
    async def test_empty_strings_from_a_progress_transform_are_kept(self):
        observer, sent = _observer()

        async def hide(text, agg_type, accumulated, remaining):
            return BotOutputTransformResult(text="", accumulated_text="", remaining_text="")

        observer.add_bot_output_transformer(hide)
        await _push(
            observer,
            AggregatedTextProgressFrame(
                segment_id=1,
                context_id="c",
                text="secret",
                aggregated_by=AggregationType.SENTENCE,
                accumulated_text="sec",
                remaining_text="ret",
            ),
        )
        [message] = [m for m in sent if isinstance(m, RTVI.BotOutputMessage)]
        self.assertEqual(message.data.text, "")
        self.assertEqual(message.data.spoken_progress.accumulated_text, "")
        self.assertEqual(message.data.spoken_progress.remaining_text, "")
        self.assertEqual(message.data.spoken_status, "completed")

    async def test_a_progress_transform_that_returns_none_keeps_the_original(self):
        observer, sent = _observer()

        async def text_only(text, agg_type, accumulated, remaining):
            return BotOutputTransformResult(text=text.upper())

        observer.add_bot_output_transformer(text_only)
        await _push(
            observer,
            AggregatedTextProgressFrame(
                segment_id=1,
                context_id="c",
                text="secret",
                aggregated_by=AggregationType.SENTENCE,
                accumulated_text="sec",
                remaining_text="ret",
            ),
        )
        [message] = [m for m in sent if isinstance(m, RTVI.BotOutputMessage)]
        self.assertEqual(message.data.text, "SECRET")
        self.assertEqual(message.data.spoken_progress.accumulated_text, "sec")
        self.assertEqual(message.data.spoken_status, "in-progress")


if __name__ == "__main__":
    unittest.main()
