#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A cancel of the task awaiting ``WorkerRunner.run()`` reaches that task."""

import asyncio
import unittest

from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner


class StubWorker(BaseWorker):
    """A bus-only worker that never finishes by itself and stops on end or cancel."""

    async def _handle_worker_end(self, message):
        await super()._handle_worker_end(message)
        self._finished_event.set()

    async def _handle_worker_cancel(self, message):
        await super()._handle_worker_cancel(message)
        self._finished_event.set()


class TestRunnerCallerCancel(unittest.IsolatedAsyncioTestCase):
    async def test_a_deadline_around_run_raises_timeout_after_teardown(self):
        """A ``wait_for`` deadline reaches its caller as TimeoutError, once the workers are down."""
        runner = WorkerRunner(handle_sigint=False)
        worker = StubWorker("worker")
        await runner.add_workers(worker)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(runner.run(), timeout=0.2)

        self.assertTrue(worker._finished_event.is_set())

    async def test_a_cancelled_caller_ends_cancelled_after_teardown(self):
        """Cancelling the task that awaits ``run()`` propagates, once the workers are down."""
        runner = WorkerRunner(handle_sigint=False)
        worker = StubWorker("worker")
        await runner.add_workers(worker)
        ready = asyncio.Event()

        @runner.event_handler("on_ready")
        async def on_ready(runner):
            ready.set()

        run_task = asyncio.create_task(runner.run())
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        run_task.cancel()
        done, _ = await asyncio.wait({run_task}, timeout=5.0)

        self.assertIn(run_task, done)
        self.assertTrue(run_task.cancelled())
        self.assertTrue(worker._finished_event.is_set())

    async def test_the_runners_own_stop_returns_normally(self):
        """``end()`` and ``cancel()`` stop the runner without raising in its caller."""
        for stop in ("end", "cancel"):
            with self.subTest(stop=stop):
                runner = WorkerRunner(handle_sigint=False)
                await runner.add_workers(StubWorker("worker"))

                @runner.event_handler("on_ready")
                async def on_ready(runner, stop=stop):
                    await getattr(runner, stop)(reason="own stop")

                run_task = asyncio.create_task(runner.run())
                done, _ = await asyncio.wait({run_task}, timeout=5.0)

                self.assertIn(run_task, done)
                self.assertFalse(run_task.cancelled())
                self.assertIsNone(run_task.result())


if __name__ == "__main__":
    unittest.main()
