#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class TestProtobufFrameSerializer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = ProtobufFrameSerializer()

    async def test_roundtrip(self):
        text_frame = TextFrame(text="hello world")
        frame = await self.serializer.deserialize(await self.serializer.serialize(text_frame))
        self.assertEqual(frame.text, text_frame.text)

        transcription_frame = TranscriptionFrame(
            text="Hello there!", user_id="123", timestamp="2021-01-01"
        )
        frame = await self.serializer.deserialize(
            await self.serializer.serialize(transcription_frame)
        )
        self.assertEqual(frame.text, transcription_frame.text)
        self.assertEqual(frame.user_id, transcription_frame.user_id)
        self.assertEqual(frame.timestamp, transcription_frame.timestamp)

        audio_frame = OutputAudioRawFrame(audio=b"1234567890", sample_rate=16000, num_channels=1)
        frame = await self.serializer.deserialize(await self.serializer.serialize(audio_frame))
        self.assertEqual(frame.audio, audio_frame.audio)
        self.assertEqual(frame.sample_rate, audio_frame.sample_rate)
        self.assertEqual(frame.num_channels, audio_frame.num_channels)

    async def test_an_audio_subclass_serializes_as_audio(self):
        """The WebSocket transports no longer flatten outbound audio to the base.

        `TTSAudioRawFrame` is an `OutputAudioRawFrame` and carries no field the
        audio message lacks, so it belongs in the audio slot. Matching the
        exact type instead would silently drop every chunk of bot speech with
        a warning.
        """
        tts_frame = TTSAudioRawFrame(audio=b"1234567890", sample_rate=16000, num_channels=1)
        frame = await self.serializer.deserialize(await self.serializer.serialize(tts_frame))
        self.assertEqual(frame.audio, tts_frame.audio)
        self.assertEqual(frame.sample_rate, tts_frame.sample_rate)

    async def test_an_input_audio_frame_is_still_not_serializable(self):
        """Resolving through the MRO must not widen what may be sent."""
        frame = InputAudioRawFrame(audio=b"1234567890", sample_rate=16000, num_channels=1)
        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_interruption_frame_roundtrip(self):
        interruption_frame = InterruptionFrame()
        frame = await self.serializer.deserialize(
            await self.serializer.serialize(interruption_frame)
        )
        self.assertIsInstance(frame, InterruptionFrame)


if __name__ == "__main__":
    unittest.main()
