#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An event loop closes even when a pipeline worker is still running.

asyncio.run ends by cancelling every task still alive, all at once, and then
waiting for all of them without a bound. pytest-asyncio closes its loops the
same way. PipelineWorker.run() answers that cancellation by cancelling its
pipeline and waiting for the pipeline to finish. It used to wait on
_finished_event, which only the pipeline's own tasks set. The same sweep had
just cancelled those tasks, so the close waited forever. A WebRTC call still up
when a server exits, or a worker that a test left behind, hung its process.

Each scenario runs as a whole process from worker_loop_close_scenarios.py,
because a hang in a loop's close cannot be caught inside that process. It
leaves a worker in one state when its main returns. The process must then
close its loop within the scenario's grace, 10 s, and its worker must report
itself finished. The controls, orderly and cancelling, closed before the fix
too.

The last control stays in this process. On a loop that is not closing, an outer
cancellation must still take the CancelFrame through the pipeline and clean it
up before run() returns, as it always has.
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import pipecat
from pipecat.frames.frames import CancelFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager

SCENARIOS = Path(__file__).with_name("worker_loop_close_scenarios.py")
PIPECAT = Path(pipecat.__file__).resolve()
# A hang is caught by the scenario's own watchdog, 10 s after main returned.
# This bound only stops a process that never got that far. It has to cover a
# cold import and pipeline setup: about 9 s on an idle Windows box, and more
# than 30 s while other test runs load the machine.
PROCESS_TIMEOUT_SECS = 120


def _run(scenario: str) -> str:
    env = dict(os.environ)
    # The subprocess imports the pipecat under test, wherever it lives.
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(PIPECAT.parents[1]), env.get("PYTHONPATH")) if path
    )
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, str(SCENARIOS), scenario],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=PROCESS_TIMEOUT_SECS,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{scenario} exited {result.returncode}:\n{output}"
    assert f"PIPECAT {PIPECAT}" in result.stdout, output
    return result.stdout


def _assert_closed_finished(stdout: str):
    closed = [line for line in stdout.splitlines() if line.startswith("CLOSED ")]
    assert len(closed) == 1, stdout
    assert "has_finished=True" in closed[0], closed[0]
    assert "finished_event=True" in closed[0], closed[0]


@pytest.mark.parametrize("scenario", ["prestart", "idle", "detached", "ending"])
def test_the_loop_closes_on_a_worker_still_running(scenario):
    _assert_closed_finished(_run(scenario))


@pytest.mark.parametrize("scenario", ["orderly", "cancelling"])
def test_control_the_loop_closes_on_a_worker_stopped_or_cancelling(scenario):
    _assert_closed_finished(_run(scenario))


class _Records(FrameProcessor):
    """Passes every frame on, and records the CancelFrame and its own cleanup."""

    def __init__(self):
        super().__init__()
        self.saw_cancel = False
        self.cleaned_up = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, CancelFrame):
            self.saw_cancel = True
        await self.push_frame(frame, direction)

    async def cleanup(self):
        await super().cleanup()
        self.cleaned_up = True


@pytest.mark.asyncio
async def test_control_an_outer_cancel_on_a_live_loop_still_cancels_the_pipeline():
    recorder = _Records()
    worker = PipelineWorker(Pipeline([recorder]))
    finished = []

    @worker.event_handler("on_pipeline_finished")
    async def _finished(_worker, frame):
        finished.append(type(frame).__name__)

    run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
    deadline = time.monotonic() + 30
    while worker.started_at is None and not run.done() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert worker.started_at is not None, "the pipeline did not start"

    run.cancel()

    done, _ = await asyncio.wait({run}, timeout=10)
    assert run in done, "run() was still running 10 s after it was cancelled"
    assert recorder.saw_cancel, "the CancelFrame never crossed the pipeline"
    assert recorder.cleaned_up, "the pipeline was not cleaned up"
    assert finished == ["CancelFrame"], f"on_pipeline_finished fired {finished}"
    assert worker.has_finished()
