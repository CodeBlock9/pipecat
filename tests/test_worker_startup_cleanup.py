"""External cancellation cannot abandon partially initialized worker resources."""

import asyncio

import pytest

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest.mark.asyncio
@pytest.mark.parametrize("during_callback", [False, True])
async def test_cancel_during_setup_joins_handlers_and_cleans_all_tasks(during_callback):
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    class Blocker(FrameProcessor):
        async def setup(self, setup):
            await super().setup(setup)
            if not during_callback:
                entered.set()
            await asyncio.Event().wait()

        async def cleanup(self):
            order.append("processor.cleanup")
            await super().cleanup()

    manager = TaskManager()
    worker = PipelineWorker(
        Pipeline([Blocker()]),
        setup_timeout_secs=0.05,
        enable_rtvi=False,
    )

    @worker.event_handler("on_setup_timeout")
    async def timed_out(worker):
        entered.set()
        await release.wait()
        order.append("finalized")

    running = asyncio.create_task(worker.run(WorkerParams(task_manager=manager)))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        running.cancel()
        await asyncio.sleep(0.03)
        if during_callback:
            assert not running.done()
            assert not order
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert order == (["finalized"] if during_callback else []) + ["processor.cleanup"]
        await asyncio.sleep(0)
        assert not [task for task in manager.current_tasks() if not task.done()]
    finally:
        release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
