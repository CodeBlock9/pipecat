"""Synthesis and playback deadlines exercised together through a real worker."""

import asyncio

import pytest

from pipecat.frames.frames import (
    EndFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy, WorkerParams
from pipecat.services.tts_service import TTS_SYNTHESIS_TIMEOUT, TTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.prewarm import warm_deferred_imports


class ScriptedTTS(TTSService):
    def __init__(self, script):
        # Preserve the production ratio of playback-idle / first-audio / gap.
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=0.15,
            synthesis_first_chunk_timeout_s=0.30,
            synthesis_chunk_gap_timeout_s=0.20,
        )
        self.script = script
        self.completed = asyncio.Queue()
        self.requested = asyncio.Event()
        self.audio = []

    async def run_tts(self, text, context_id):
        self.requested.set()
        async for frame in self.script(self, context_id):
            yield frame

    async def push_frame(self, frame, direction=None):
        if isinstance(frame, TTSAudioRawFrame):
            self.audio.append(frame)
        if direction is None:
            await super().push_frame(frame)
        else:
            await super().push_frame(frame, direction)

    async def on_audio_context_completed(self, context_id):
        await super().on_audio_context_completed(context_id)
        await self.completed.put(context_id)


def audio(context_id):
    return TTSAudioRawFrame(b"\0" * 960, 24000, 1, context_id=context_id)


async def run_contexts(script, count=1, ending="complete"):
    # Match the application's prewarmed call runner; these deadlines measure
    # provider work, not a cold NLTK/scikit-learn import on the test machine.
    warm_deferred_imports()
    service = ScriptedTTS(script)
    worker = PipelineWorker(
        Pipeline([service]),
        enable_rtvi=False,
        processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE,
    )
    errors = []
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(*args):
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def on_error(worker, frame):
        errors.append(frame)

    running = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await asyncio.wait_for(started.wait(), 5)
        for _ in range(count):
            await worker.queue_frame(TTSSpeakFrame("Hello."))
            if ending == "complete":
                await asyncio.wait_for(service.completed.get(), 3)
            else:
                await asyncio.wait_for(service.requested.wait(), 3)
                await asyncio.sleep(0.05)
                assert service._synthesis_watchdogs
        if ending == "cancel":
            await worker.cancel()
        else:
            if ending == "interrupt":
                await worker.queue_frame(InterruptionFrame())
                await asyncio.sleep(0.35)
            await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)
    finally:
        if not running.done():
            await worker.cancel()
            await running
    assert service._synthesis_watchdogs == {}
    assert service._synthesis_states == {}
    return service, errors


@pytest.mark.asyncio
async def test_slow_first_audio_survives_playback_idle_and_control_frames():
    async def script(service, context_id):
        yield TTSStartedFrame(context_id=context_id)
        await asyncio.sleep(0.24)  # After idle and gap, before first-audio deadline.
        yield audio(context_id)

    service, errors = await run_contexts(script)
    assert len(service.audio) == 1
    assert errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [False, True])
async def test_first_audio_timeout_is_reported_once_despite_playback_idle(controls):
    async def script(service, context_id):
        while True:
            await asyncio.sleep(0.04 if controls else 2)
            yield TTSStartedFrame(context_id=context_id)

    service, errors = await run_contexts(script, count=3)
    assert len(errors) == 3
    assert all(TTS_SYNTHESIS_TIMEOUT in frame.error for frame in errors)
    assert service._consecutive_zero_audio_contexts == 0
    assert not service.is_usable


@pytest.mark.asyncio
async def test_http_gap_uses_its_deadline_and_does_not_report_empty():
    async def script(service, context_id):
        yield audio(context_id)
        await asyncio.sleep(2)

    service, errors = await run_contexts(script)
    assert len(service.audio) == 1
    assert len(errors) == 1
    assert "further audio within 0.2s" in errors[0].error


@pytest.mark.asyncio
async def test_websocket_audio_after_idle_keeps_original_context():
    async def script(service, context_id):
        async def receive():
            await asyncio.sleep(0.24)
            await service.append_to_audio_context(context_id, audio(context_id))
            await service.remove_audio_context(context_id)

        service.create_task(receive())
        yield None

    service, errors = await run_contexts(script)
    assert len(service.audio) == 1
    assert errors == []


@pytest.mark.asyncio
async def test_three_explicit_empty_websocket_completions_are_permanent():
    async def script(service, context_id):
        await service.remove_audio_context(context_id)
        yield None

    service, errors = await run_contexts(script, count=3)
    assert len(errors) == 3
    assert all("completed with no audio" in frame.error for frame in errors)
    assert not service.is_usable


@pytest.mark.asyncio
async def test_another_websocket_request_does_not_restart_the_context_watchdog():
    checked = asyncio.Event()

    async def pending_audio():
        yield None

    async def script(service, context_id):
        async def send_more():
            await asyncio.sleep(0.05)
            watchdog = service._synthesis_watchdogs[context_id]
            await service.tts_process_generator(context_id, pending_audio())
            assert service._synthesis_watchdogs[context_id] is watchdog
            checked.set()

        service.create_task(send_more())
        yield None

    _, errors = await run_contexts(script)
    assert checked.is_set()
    assert len(errors) == 1
    assert TTS_SYNTHESIS_TIMEOUT in errors[0].error


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["interrupt", "cancel"])
async def test_interruption_and_cleanup_do_not_report_pending_synthesis_as_failure(ending):
    async def script(service, context_id):
        yield None

    service, errors = await run_contexts(script, ending=ending)
    assert errors == []
    assert service.is_usable
