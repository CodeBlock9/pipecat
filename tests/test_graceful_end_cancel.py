#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A cancel preempts a graceful end that has not landed.

The worker waits for an EndFrame or StopFrame without a bound, because ending
flushes what is queued and a farewell has to play out in full. A cancel used to
go onto the push queue behind it, and the push queue's own task was the one
waiting, so a graceful end that never landed could never be cancelled: not by
the idle monitor, not by an outer cancellation, not by anyone.

A processor holds the EndFrame the way a farewell still playing, or a carrier
request that never returns, does. Every await is bounded, so a regression is a
red assertion and never a stuck run.

The r10 tests hold the rule that a cancel preempting an EndFrame(TRANSFER_CALL)
hangs nothing up: the redirect's outcome is unknown, and a hangup would turn
the transfer into a disconnect. They use the real FastAPI websocket output
transport and Twilio serializer, with a redirect that never returns and a
hangup strategy that records every call.
"""

import asyncio
from unittest.mock import AsyncMock, PropertyMock

import pytest
from starlette.websockets import WebSocketState

from pipecat.frames.frames import BotSpeakingFrame, CancelFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.enums import EndTaskReason
from pipecat.workers.runner import WorkerRunner

TRANSFER = EndTaskReason.TRANSFER_CALL.value


class _HoldsEnd(FrameProcessor):
    """Passes every frame on except the EndFrame, which it holds.

    With ``release_after`` unset it holds it for good, like a carrier request
    that never returns. Otherwise it passes it on after that many seconds, like
    a farewell still playing, pushing a BotSpeakingFrame every
    ``speaking_every`` seconds meanwhile, as an output transport does while its
    audio plays.
    """

    def __init__(self, release_after: float | None = None, speaking_every: float | None = None):
        super().__init__()
        self.holding = asyncio.Event()
        self.released = False
        self._release_after = release_after
        self._speaking_every = speaking_every

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            self.holding.set()
            if self._release_after is None:
                return
            await self._play_out(self._release_after)
            self.released = True
        await self.push_frame(frame, direction)

    async def _play_out(self, seconds: float):
        loop = asyncio.get_running_loop()
        until = loop.time() + seconds
        while (left := until - loop.time()) > 0:
            if self._speaking_every:
                await self.push_frame(BotSpeakingFrame())
            await asyncio.sleep(min(left, self._speaking_every or left))


class _ReleasesEndWithTheCancel(FrameProcessor):
    """Holds the EndFrame until a CancelFrame arrives, then passes it on just
    ahead of the cancel: the EndFrame lands while the cancel is in flight."""

    def __init__(self):
        super().__init__()
        self.holding = asyncio.Event()
        self._held: EndFrame | None = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            self._held = frame
            self.holding.set()
            return
        if isinstance(frame, CancelFrame) and self._held:
            await self.push_frame(self._held, direction)
        await self.push_frame(frame, direction)


class _Stuck(FrameProcessor):
    """Holds the EndFrame for good and drops the CancelFrame as well, like a
    processor blocked in a call that ignores cancellation, so the cancel's own
    wait is the one that runs out."""

    def __init__(self):
        super().__init__()
        self.holding = asyncio.Event()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            self.holding.set()
            return
        if isinstance(frame, CancelFrame):
            return
        await self.push_frame(frame, direction)


class _StalledTransfer(TransferStrategy):
    """A conference redirect whose HTTP response never comes back.

    With ``ws_state`` given, the carrier is taken to have acted on the redirect
    and closed the media stream before the response went missing.
    """

    def __init__(self, ws_state: dict | None = None, completes: bool = False):
        self.entered = asyncio.Event()
        self.calls = 0
        self._ws_state = ws_state
        self._completes = completes

    async def execute_transfer(self, context):
        self.calls += 1
        self.entered.set()
        if self._completes:
            return True
        if self._ws_state is not None:
            self._ws_state["client"] = WebSocketState.DISCONNECTED
        await asyncio.Event().wait()
        return True


class _RecordingHangup(HangupStrategy):
    def __init__(self):
        self.calls: list = []

    async def execute_hangup(self, context):
        self.calls.append(context.get("call_sid"))
        return True


def _transport(ws_state: dict, transfer: TransferStrategy, hangup: HangupStrategy):
    ws = AsyncMock()
    type(ws).client_state = PropertyMock(side_effect=lambda: ws_state["client"])
    type(ws).application_state = PropertyMock(side_effect=lambda: ws_state["app"])
    serializer = TwilioFrameSerializer(
        stream_sid="MZ-probe",
        call_sid="CA-probe",
        account_sid="AC-probe",
        auth_token="probe-token",
        transfer_strategy=transfer,
        hangup_strategy=hangup,
    )
    params = FastAPIWebsocketParams(serializer=serializer, allowed_origins=[])
    return FastAPIWebsocketTransport(websocket=ws, params=params)


def _open_socket() -> dict:
    return {"client": WebSocketState.CONNECTED, "app": WebSocketState.CONNECTED}


def _worker(holder: FrameProcessor, **kwargs) -> tuple[PipelineWorker, list, list]:
    kwargs.setdefault("idle_timeout_secs", None)
    kwargs.setdefault("cancel_timeout_secs", 0.5)
    worker = PipelineWorker(Pipeline([holder]), enable_rtvi=False, **kwargs)
    finished: list = []
    timeouts: list = []

    @worker.event_handler("on_pipeline_finished")
    async def _finished(_worker, frame):
        finished.append(type(frame).__name__)

    @worker.event_handler("on_pipeline_timeout")
    async def _timed_out(_worker, frame):
        timeouts.append(type(frame).__name__)

    return worker, finished, timeouts


async def _started(worker: PipelineWorker, run: asyncio.Task):
    async def wait():
        while worker.started_at is None:
            if run.done():
                await run
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 10)


async def _run_like_mesa(worker: PipelineWorker):
    """Mesa's ``run_pipeline_worker``: a runner of its own, ending with the worker."""
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    await runner.run(auto_end=True)


async def _force_down(run: asyncio.Task, worker: PipelineWorker):
    """Release the held end and cancel until the run is gone, bounded."""
    for _ in range(10):
        if run.done():
            break
        worker._pipeline_end_event.set()
        run.cancel()
        await asyncio.wait({run}, timeout=1)
    await asyncio.wait({run}, timeout=5)


@pytest.mark.asyncio
async def test_a_cancel_preempts_a_graceful_end_that_never_lands():
    holder = _HoldsEnd()
    worker, finished, timeouts = _worker(holder)
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(holder.holding.wait(), 5)

        await worker.cancel(reason="caller hung up", cancel_timeout_secs=0.2)

        done, _ = await asyncio.wait({run}, timeout=3)
        assert run in done, (
            "run() was still waiting on its EndFrame 3 s after a cancel bounded at 0.2 s"
        )
        assert len(finished) == 1, f"on_pipeline_finished fired {finished}"
        assert timeouts == [], f"on_pipeline_timeout fired {timeouts}"
        assert finished == ["CancelFrame"], (
            "a preempted end finishes as the cancel that preempted it"
        )
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_the_idle_timeout_ends_a_worker_stalled_on_its_graceful_end():
    holder = _HoldsEnd()
    worker, finished, timeouts = _worker(holder, idle_timeout_secs=0.3)
    run = asyncio.create_task(_run_like_mesa(worker))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(holder.holding.wait(), 5)

        done, _ = await asyncio.wait({run}, timeout=4)
        assert run in done, (
            "the worker's idle-timeout cancel could not end a worker stalled on its "
            "EndFrame within 4 s (idle timeout 0.3 s, cancel timeout 0.5 s)"
        )
        assert len(finished) == 1, f"on_pipeline_finished fired {finished}"
        assert timeouts == [], f"on_pipeline_timeout fired {timeouts}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_an_outer_cancellation_ends_a_worker_stalled_on_its_graceful_end():
    holder = _HoldsEnd()
    worker, finished, timeouts = _worker(holder)
    run = asyncio.create_task(_run_like_mesa(worker))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(holder.holding.wait(), 5)

        run.cancel()

        done, _ = await asyncio.wait({run}, timeout=3)
        assert run in done, (
            "WorkerRunner.run() was still running 3 s after its task was cancelled "
            "during a graceful end"
        )
        assert len(finished) == 1, f"on_pipeline_finished fired {finished}"
        assert timeouts == [], f"on_pipeline_timeout fired {timeouts}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_control_a_graceful_end_that_outlasts_the_cancel_timeout_is_not_cut():
    """Nothing bounds the graceful wait: an EndFrame held 1.0 s, twice the
    worker's cancel timeout, still plays out in full."""
    holder = _HoldsEnd(release_after=1.0)
    worker, finished, _ = _worker(holder)
    run = asyncio.create_task(_run_like_mesa(worker))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())

        done, _ = await asyncio.wait({run}, timeout=5)
        assert run in done
        assert holder.released, "the EndFrame was cut before the processor holding it passed it on"
        assert finished == ["EndFrame"]
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_control_a_graceful_end_that_keeps_speaking_outlasts_the_idle_timeout():
    """A farewell longer than the idle timeout is not cut while it plays: the
    output keeps pushing BotSpeakingFrame, which keeps the idle monitor quiet."""
    holder = _HoldsEnd(release_after=1.0, speaking_every=0.1)
    worker, finished, _ = _worker(holder, idle_timeout_secs=0.3)
    run = asyncio.create_task(_run_like_mesa(worker))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())

        done, _ = await asyncio.wait({run}, timeout=5)
        assert run in done
        assert holder.released, "the EndFrame was cut while its farewell was still playing"
        assert finished == ["EndFrame"]
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_an_end_that_lands_while_a_cancel_preempts_it_finishes_the_pipeline_once():
    holder = _ReleasesEndWithTheCancel()
    worker, finished, _ = _worker(holder)
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(holder.holding.wait(), 5)

        await worker.cancel(reason="caller hung up")

        done, _ = await asyncio.wait({run}, timeout=3)
        assert run in done
        assert finished == ["EndFrame"], f"on_pipeline_finished fired {finished}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_the_cancel_that_preempts_an_end_is_waited_for_once():
    """The CancelFrame also sits on the push queue. Once the preempted wait
    returns it must not be taken up again: a second wait would time out a
    second time and report the pipeline timed out twice."""
    holder = _Stuck()
    worker, finished, timeouts = _worker(holder, cancel_timeout_secs=0.3)
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(holder.holding.wait(), 5)

        await worker.cancel(reason="caller hung up")

        done, _ = await asyncio.wait({run}, timeout=3)
        assert run in done
        assert timeouts == ["CancelFrame"], f"on_pipeline_timeout fired {timeouts}"
        assert finished == ["CancelFrame"], f"on_pipeline_finished fired {finished}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_r10_a_cancel_preempting_a_stalled_transfer_hangs_nothing_up():
    """The media socket is still open when the cancel preempts the transfer."""
    transfer, hangup = _StalledTransfer(), _RecordingHangup()
    transport = _transport(_open_socket(), transfer, hangup)
    worker, finished, _ = _worker(transport.output())
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame(reason=TRANSFER))
        await asyncio.wait_for(transfer.entered.wait(), 5)

        await worker.cancel(reason="idle timeout")

        done, _ = await asyncio.wait({run}, timeout=4)
        assert run in done, "run() was still waiting on its transfer 4 s after a cancel"
        assert hangup.calls == [], (
            f"a preempted transfer was followed by a carrier hangup of {hangup.calls}"
        )
        assert finished == ["CancelFrame"], f"on_pipeline_finished fired {finished}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_r10_the_idle_cancel_preempting_a_stalled_transfer_hangs_nothing_up():
    """Pipecat's own idle cancel, run the way Mesa runs a worker."""
    transfer, hangup = _StalledTransfer(), _RecordingHangup()
    transport = _transport(_open_socket(), transfer, hangup)
    worker, finished, _ = _worker(transport.output(), idle_timeout_secs=0.5)
    run = asyncio.create_task(_run_like_mesa(worker))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame(reason=TRANSFER))
        await asyncio.wait_for(transfer.entered.wait(), 5)

        done, _ = await asyncio.wait({run}, timeout=5)
        assert run in done, (
            "the idle cancel could not end a worker stalled on its transfer within 5 s"
        )
        assert hangup.calls == [], (
            f"a preempted transfer was followed by a carrier hangup of {hangup.calls}"
        )
        assert finished == ["CancelFrame"], f"on_pipeline_finished fired {finished}"
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_r10_control_the_redirect_took_and_closed_the_socket():
    """The carrier acted on the redirect and closed the stream; only the
    response is missing. Nothing can be written, so nothing is hung up."""
    ws_state = _open_socket()
    transfer, hangup = _StalledTransfer(ws_state=ws_state), _RecordingHangup()
    transport = _transport(ws_state, transfer, hangup)
    worker, _, _ = _worker(transport.output())
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame(reason=TRANSFER))
        await asyncio.wait_for(transfer.entered.wait(), 5)

        await worker.cancel(reason="idle timeout")

        await asyncio.wait({run}, timeout=4)
        assert hangup.calls == []
    finally:
        await _force_down(run, worker)


@pytest.mark.asyncio
async def test_r10_control_a_transfer_that_completes_hangs_nothing_up():
    transfer, hangup = _StalledTransfer(completes=True), _RecordingHangup()
    transport = _transport(_open_socket(), transfer, hangup)
    worker, finished, _ = _worker(transport.output())
    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    try:
        await _started(worker, run)
        await worker.queue_frame(EndFrame(reason=TRANSFER))

        done, _ = await asyncio.wait({run}, timeout=4)
        assert run in done
        assert hangup.calls == []
        assert finished == ["EndFrame"]
    finally:
        await _force_down(run, worker)
