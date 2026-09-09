#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice Activity Detection (VAD) analyzer base classes and utilities.

This module provides the abstract base class for VAD analyzers and associated
data structures for voice activity detection in audio streams. Includes state
management, parameter configuration, and audio analysis framework.
"""

import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import calculate_audio_volume, exp_smoothing

#: One pool for every analyzer in the process, created on first use.
#:
#: An analyzer is built per call, so a pool each meant one operating-system
#: thread per concurrent call to do about 31 inferences a second on it. The
#: reason the smart-turn analyzer keeps a thread per stream does not apply
#: here: an end-of-turn decision takes roughly a quarter of a second and would
#: serialise across calls, while a Silero window takes well under a
#: millisecond, so even two dozen concurrent calls leave this pool almost
#: entirely idle. ``ThreadPoolExecutor`` grows lazily -- it only starts a new
#: worker when every existing one is busy -- so the thread count follows
#: simultaneous inference rather than concurrent calls.
_INFERENCE_POOL: ThreadPoolExecutor | None = None
_INFERENCE_POOL_LOCK = threading.Lock()


def _inference_pool() -> ThreadPoolExecutor:
    """The process-wide pool VAD inference runs on."""
    global _INFERENCE_POOL
    with _INFERENCE_POOL_LOCK:
        if _INFERENCE_POOL is None:
            _INFERENCE_POOL = ThreadPoolExecutor(thread_name_prefix="vad")
        return _INFERENCE_POOL


VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers.

    Provides the framework for implementing VAD analysis with configurable
    parameters, state management, and audio processing capabilities.
    Subclasses must implement the core voice confidence calculation.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters for detection configuration.
        """
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

        # Inference runs on the process-wide pool. See _INFERENCE_POOL above
        # for why this analyzer does not own a thread of its own.
        self._executor = _inference_pool()

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            Current audio sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of audio channels (always 1 for mono).
        """
        return self._num_channels

    @property
    def params(self) -> VADParams:
        """Get the current VAD parameters.

        Returns:
            Current VAD configuration parameters.
        """
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        pass

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.
        """
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        volume = calculate_audio_volume(audio, self.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.

        Processes incoming audio data, maintains internal state, and determines
        voice activity status based on confidence and volume thresholds.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Current VAD state after processing the buffer.
        """
        # The accumulation happens here, on the event loop, rather than inside
        # the executor. A transport delivers 20 ms chunks and Silero wants a
        # 32 ms window, so more than a third of the chunks used to make a round
        # trip to a thread that appended them and returned the state unchanged.
        # Whole windows only, and the remainder waits for the next chunk.
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        whole = len(self._vad_buffer) - len(self._vad_buffer) % num_required_bytes
        audio, self._vad_buffer = self._vad_buffer[:whole], self._vad_buffer[whole:]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_analyzer, audio)

    def _run_analyzer(self, audio: bytes) -> VADState:
        """Analyze whole windows of audio and return the current VAD state.

        Runs on the inference pool. The caller has already cut ``audio`` to a
        whole number of windows, so nothing here buffers.
        """
        num_required_bytes = self._vad_frames_num_bytes

        for start in range(0, len(audio), num_required_bytes):
            audio_frames = audio[start : start + num_required_bytes]

            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    async def cleanup(self):
        """Drop what this analyzer holds at teardown.

        There is no executor to shut down: inference runs on the process-wide
        pool, which outlives every analyzer. Shutting that down here would stop
        voice detection for every other call in the process.
        """
        self._vad_buffer = b""
