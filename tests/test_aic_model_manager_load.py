#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AICModelManager shares one model load among its waiters.

aic_sdk is not installed here (tests/test_aic_filter.py skips for that reason),
so aic_filter is imported with a stand-in aic_sdk module, for the import only.
The code under test is the real ``AICModelManager.acquire``; only the loader it
schedules is replaced, by one that waits on an event.
"""

import asyncio
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy  # noqa: F401 -- imported before the stand-in, so restoring sys.modules keeps it
import pytest

import pipecat.audio.filters.base_audio_filter  # noqa: F401 -- likewise
import pipecat.frames.frames  # noqa: F401 -- likewise

MODULE = "pipecat.audio.filters.aic_filter"
_MODULE_CACHE = []


def _aic_filter_module():
    if _MODULE_CACHE:
        return _MODULE_CACHE[0]
    stub = types.ModuleType("aic_sdk")
    for name in ("Model", "ProcessorAsync", "ProcessorConfig", "ProcessorParameter"):
        setattr(stub, name, type(name, (), {}))
    stub.ParameterOutOfRangeError = type("ParameterOutOfRangeError", (Exception,), {})
    stub.set_sdk_id = lambda _id: None
    with patch.dict(sys.modules, {"aic_sdk": stub}):
        sys.modules.pop(MODULE, None)
        _MODULE_CACHE.append(importlib.import_module(MODULE))
    return _MODULE_CACHE[0]


@pytest.mark.asyncio
async def test_a_cancelled_waiter_leaves_the_shared_load_to_the_others():
    manager = _aic_filter_module().AICModelManager
    model = object()
    gate = asyncio.Event()
    loads = 0

    async def load(cache_key, **kwargs):
        nonlocal loads
        loads += 1
        await gate.wait()
        return model

    path = Path("cancelled-waiter.aicmodel")
    with patch.object(manager, "_load_model_from_file", load):
        first = asyncio.create_task(manager.acquire(model_path=path))
        second = asyncio.create_task(manager.acquire(model_path=path))
        await asyncio.sleep(0.01)

        first.cancel()
        await asyncio.sleep(0.01)
        third = asyncio.create_task(manager.acquire(model_path=path))
        await asyncio.sleep(0.01)
        gate.set()

        results = await asyncio.gather(second, third, return_exceptions=True)
        with pytest.raises(asyncio.CancelledError):
            await first

    assert [r if isinstance(r, BaseException) else r[0] for r in results] == [model, model]
    assert loads == 1, f"the model was loaded {loads} times"
    key = results[0][1]
    manager.release(key)
    manager.release(key)


@pytest.mark.asyncio
async def test_uncancelled_waiters_share_one_load():
    manager = _aic_filter_module().AICModelManager
    model = object()
    loads = 0

    async def load(cache_key, **kwargs):
        nonlocal loads
        loads += 1
        await asyncio.sleep(0.01)
        return model

    path = Path("shared-load.aicmodel")
    with patch.object(manager, "_load_model_from_file", load):
        results = await asyncio.gather(
            manager.acquire(model_path=path), manager.acquire(model_path=path)
        )

    assert [r[0] for r in results] == [model, model] and loads == 1
    manager.release(results[0][1])
    manager.release(results[0][1])
