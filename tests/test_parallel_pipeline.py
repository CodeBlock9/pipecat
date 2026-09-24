#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `ParallelPipeline`'s deduplication of frames that leave more than one branch."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection

SEEN_IDS_BOUND = 1024


class TestSeenIdsAreBounded(unittest.IsolatedAsyncioTestCase):
    async def test_retained_frame_ids_are_bounded(self):
        """A long run of unique frames keeps at most the bound's worth of ids."""
        parallel = ParallelPipeline([IdentityFilter()], [IdentityFilter()])
        parallel.push_frame = AsyncMock()
        for _ in range(15000):
            await parallel._parallel_push_frame(TextFrame("x"), FrameDirection.DOWNSTREAM)
        self.assertLessEqual(
            len(parallel._seen_ids),
            SEEN_IDS_BOUND,
            f"{len(parallel._seen_ids)} frame ids retained after 15,000 unique frames",
        )
        self.assertEqual(parallel.push_frame.await_count, 15000)

    async def test_a_copy_within_the_bound_is_dropped(self):
        """A copy that trails its original by one frame less than the bound is still dropped."""
        parallel = ParallelPipeline([IdentityFilter()], [IdentityFilter()])
        parallel.push_frame = AsyncMock()
        first = TextFrame("x")
        await parallel._parallel_push_frame(first, FrameDirection.DOWNSTREAM)
        for _ in range(SEEN_IDS_BOUND - 1):
            await parallel._parallel_push_frame(TextFrame("x"), FrameDirection.DOWNSTREAM)
        await parallel._parallel_push_frame(first, FrameDirection.DOWNSTREAM)
        self.assertEqual(parallel.push_frame.await_count, SEEN_IDS_BOUND)

    async def test_a_duplicate_from_the_other_branch_is_dropped(self):
        """The same frame arriving twice is pushed once."""
        parallel = ParallelPipeline([IdentityFilter()], [IdentityFilter()])
        parallel.push_frame = AsyncMock()
        frame = TextFrame("x")
        await parallel._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)
        await parallel._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(parallel.push_frame.await_count, 1)


if __name__ == "__main__":
    unittest.main()
