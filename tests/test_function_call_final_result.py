#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A tool call's deadline and settlement follow its final result.

- ``LLMService``: only a final result ends a call's deadline, and a call whose
  final result was broadcast is not settled a second time.
- ``UserIdleController``: outstanding calls are tracked by ``tool_call_id``, and
  only a cancel or a final result settles one.

Every wait ends on an event (a settlement frame, a report, the idle event),
bounded by ``asyncio.wait_for``. Nothing sleeps a fixed settle time.
"""

import asyncio
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# Bounds every wait, so a regression is a red assertion, never a stuck run.
WAIT_BOUND = 5.0

PROGRESS = FunctionCallResultProperties(is_final=False)


def _is_final(frame: FunctionCallResultFrame) -> bool:
    return frame.properties is None or frame.properties.is_final


class _Service(LLMService):
    """An LLM service that runs real function-call tasks on a real task manager."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._setup = frame_processor_setup(
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None)
        )
        self._task_manager = TaskManager()


class _Recorder:
    """Records what a service broadcasts and reports, and when a call settles."""

    def __init__(self, service: LLMService):
        self.frames: list = []
        self.cancel_reports: list[list[str]] = []
        self.settled = asyncio.Event()
        self.cancel_reported = asyncio.Event()
        service.broadcast_frame = self._broadcast_frame
        service._call_event_handler = self._call_event_handler

    async def _broadcast_frame(self, frame_cls, **frame_kwargs):
        frame = frame_cls(**frame_kwargs)
        self.frames.append(frame)
        if isinstance(frame, FunctionCallCancelFrame) or (
            isinstance(frame, FunctionCallResultFrame) and _is_final(frame)
        ):
            self.settled.set()

    async def _call_event_handler(self, event_name, *args, **kwargs):
        if event_name == "on_function_calls_cancelled":
            self.cancel_reports.append([call.tool_call_id for call in args[0]])
            self.cancel_reported.set()

    def types(self) -> list[type]:
        return [type(frame) for frame in self.frames]


async def _run_call(service: LLMService, function_name: str, tool_call_id: str = "call_1"):
    await service.run_function_calls(
        [
            FunctionCallFromLLM(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments={},
                context=LLMContext(),
            )
        ]
    )


class TestFunctionCallSettlesOnItsFinalResult(unittest.IsolatedAsyncioTestCase):
    TIMEOUT = 0.1
    HANDLER_DURATION = 0.4

    async def test_a_progress_update_leaves_the_deadline_running(self):
        """An async call that reports progress and outlives its deadline is still cancelled.

        The deadline then reports it, as it reports any call that never sent a
        final result.
        """
        service = _Service(function_call_timeout_secs=self.TIMEOUT)
        recorder = _Recorder(service)
        finished = []

        async def lookup(params: FunctionCallParams):
            await params.result_callback({"status": "working"}, properties=PROGRESS)
            await asyncio.sleep(self.HANDLER_DURATION)
            finished.append(params.tool_call_id)
            await params.result_callback({"status": "done"})

        service.register_function("lookup", lookup, cancel_on_interruption=False)
        await _run_call(service, "lookup")
        await asyncio.wait_for(recorder.settled.wait(), WAIT_BOUND)

        self.assertEqual(finished, [])
        self.assertEqual(
            recorder.types(),
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
                FunctionCallCancelFrame,
            ],
        )
        self.assertFalse(_is_final(recorder.frames[2]))
        self.assertTrue(recorder.frames[3].run_llm)
        await asyncio.wait_for(recorder.cancel_reported.wait(), WAIT_BOUND)
        self.assertEqual(recorder.cancel_reports, [["call_1"]])

    async def test_a_call_that_sent_only_progress_is_still_cancelled_on_request(self):
        """A progress update does not count as the call's final result."""
        service = _Service()
        recorder = _Recorder(service)
        progress_sent = asyncio.Event()
        unwound = []

        async def lookup(params: FunctionCallParams):
            await params.result_callback({"status": "working"}, properties=PROGRESS)
            progress_sent.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                unwound.append(params.tool_call_id)
                raise

        service.register_function("lookup", lookup, cancel_on_interruption=False)
        await _run_call(service, "lookup")
        await asyncio.wait_for(progress_sent.wait(), WAIT_BOUND)
        await service._cancel_function_calls_by_tool_call_id("call_1")

        self.assertEqual(
            recorder.types(),
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
                FunctionCallCancelFrame,
            ],
        )
        self.assertEqual(recorder.cancel_reports, [["call_1"]])
        self.assertEqual(unwound, ["call_1"])

    async def test_a_reported_call_is_not_settled_again_by_an_interruption(self):
        """A call whose final result went out, still unwinding when the user interrupts.

        Its task is still cancelled, but it gets no cancel frame and no
        cancellation report: it has already settled once.
        """
        service = _Service()
        recorder = _Recorder(service)
        result_sent = asyncio.Event()
        unwound = []

        async def book(params: FunctionCallParams):
            await params.result_callback({"booked": True})
            result_sent.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                unwound.append(params.tool_call_id)
                raise

        service.register_function("book", book)
        await _run_call(service, "book")
        await asyncio.wait_for(result_sent.wait(), WAIT_BOUND)
        await service._handle_interruptions(InterruptionFrame())

        self.assertEqual(
            recorder.types(),
            [FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
        )
        self.assertEqual(recorder.cancel_reports, [])
        self.assertEqual(unwound, ["call_1"])
        self.assertEqual(service._function_call_tasks, {})

    async def test_an_interruption_during_the_deadline_cancel_settles_the_call_once(self):
        """The result callback is cancelled before its broadcast.

        A final result settles the call and then waits for its deadline to be
        cancelled before it broadcasts. An interruption landing in that wait
        cancels the callback, so the result never goes out, and the call must
        still get its one settlement: the cancel.
        """
        service = _Service(function_call_timeout_secs=30.0)
        recorder = _Recorder(service)
        deadline_cancel_started = asyncio.Event()
        cancel_task = service.cancel_task

        async def cancel_task_that_waits_on_the_deadline(task, timeout=1.0):
            # The first task cancelled that is not a function call is the
            # deadline, cancelled by the final result. Its cancellation waits
            # on an event nothing sets, until the interruption cancels it.
            if task not in service._function_call_tasks and not deadline_cancel_started.is_set():
                deadline_cancel_started.set()
                await asyncio.Event().wait()
            await cancel_task(task, timeout)

        service.cancel_task = cancel_task_that_waits_on_the_deadline

        async def book(params: FunctionCallParams):
            await params.result_callback({"booked": True})

        service.register_function("book", book)
        await _run_call(service, "book")
        await asyncio.wait_for(deadline_cancel_started.wait(), WAIT_BOUND)
        await service._handle_interruptions(InterruptionFrame())

        self.assertEqual(
            recorder.types(),
            [FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallCancelFrame],
        )
        self.assertEqual(recorder.cancel_reports, [["call_1"]])


USER_IDLE_TIMEOUT = 0.2


def _call(tool_call_id: str, name: str = "lookup") -> FunctionCallFromLLM:
    return FunctionCallFromLLM(
        function_name=name, tool_call_id=tool_call_id, arguments={}, context=LLMContext()
    )


def _result(tool_call_id: str, *, final: bool = True, name: str = "lookup"):
    return FunctionCallResultFrame(
        function_name=name,
        tool_call_id=tool_call_id,
        arguments={},
        result="ok",
        properties=None if final else PROGRESS,
    )


class TestIdleTracksOutstandingCallsById(unittest.IsolatedAsyncioTestCase):
    """The frame sequences are the ones ``LLMService`` broadcasts.

    ``FunctionCallsStartedFrame`` carries only the user-visible calls, a result
    frame goes out per update, and a cancel frame per cancelled call.
    """

    async def asyncSetUp(self):
        self.task_manager = TaskManager()

    async def _idle_fires(self, frames) -> bool:
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))
        fired = asyncio.Event()

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            fired.set()

        try:
            for frame in frames:
                await controller.process_frame(frame)
            try:
                await asyncio.wait_for(fired.wait(), USER_IDLE_TIMEOUT + 0.15)
            except TimeoutError:
                return False
            return True
        finally:
            await controller.cleanup()

    async def test_a_final_result_after_progress_settles_only_that_call(self):
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1"), _call("2")]),
                _result("1", final=False),
                _result("1"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertFalse(fired, "idle fired while call 2 was still running")

    async def test_a_result_for_an_untracked_id_settles_nothing(self):
        """The built-in cancel tool's result carries an id no started frame did."""
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                _result("cancel-1", name="cancel_lookup"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertFalse(fired, "idle fired while call 1 was still running")

    async def test_a_cancel_after_a_final_result_settles_nothing_more(self):
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1"), _call("2")]),
                _result("1"),
                FunctionCallCancelFrame(function_name="lookup", tool_call_id="1"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertFalse(fired, "idle fired while call 2 was still running")

    async def test_a_progress_update_keeps_the_call_outstanding(self):
        """The bot speaks an async call's progress update while the call runs on."""
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                _result("1", final=False),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertFalse(fired, "idle fired while call 1 was still running")

    async def test_control_one_call_finishing_lets_idle_fire(self):
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                _result("1"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertTrue(fired)

    async def test_control_a_final_result_after_progress_lets_idle_fire(self):
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                _result("1", final=False),
                _result("1"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertTrue(fired)

    async def test_a_cancel_settles_its_call(self):
        """An interrupted or timed-out call is settled by its cancel frame."""
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                FunctionCallCancelFrame(function_name="lookup", tool_call_id="1"),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertTrue(fired, "idle never fired after call 1 was cancelled")

    async def test_a_barge_in_while_a_call_runs_keeps_idle_off(self):
        fired = await self._idle_fires(
            [
                FunctionCallsStartedFrame(function_calls=[_call("1")]),
                BotStartedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
                UserStoppedSpeakingFrame(),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
            ]
        )
        self.assertFalse(fired, "idle fired while call 1 was still running")


if __name__ == "__main__":
    unittest.main()
