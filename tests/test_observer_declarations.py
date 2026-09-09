#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Every observer's declaration covers every type its handler branches on.

``observed_frame_types`` tells ``WorkerObserver`` which frames to bother
queueing. A declaration that omits a type the handler acts on produces no error
anywhere: the branch simply never runs, and a turn stays open or a span is
never started. So the declarations are held against the handlers themselves --
each ``on_push_frame`` is parsed, every ``isinstance(data.frame, X)`` in it is
resolved to a class, and the declaration must admit it.

The guard walks the source rather than the runtime because that is where the
branches are; it is the shape ``test_retention_invariants.py`` uses in the app
repository for the same reason.
"""

import ast
import asyncio
import inspect
import sys

import pytest

from pipecat.frames.frames import BotSpeakingFrame, Frame, TextFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.worker import IdleFrameObserver
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver
from pipecat.services.google.rtvi import GoogleRTVIObserver
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

#: Every leaf observer in the tree, and how to build one. Listed rather than
#: discovered so that a new observer fails this file until someone adds it and
#: thinks about its declaration.
OBSERVERS = {
    TurnTrackingObserver: lambda: TurnTrackingObserver(),
    UserBotLatencyObserver: lambda: UserBotLatencyObserver(),
    StartupTimingObserver: lambda: StartupTimingObserver(),
    TurnTraceObserver: lambda: TurnTraceObserver(
        TurnTrackingObserver(), latency_tracker=UserBotLatencyObserver()
    ),
    IdleFrameObserver: lambda: IdleFrameObserver(
        idle_event=asyncio.Event(), idle_timeout_frames=(BotSpeakingFrame,)
    ),
    DebugLogObserver: lambda: DebugLogObserver(frame_types=(TextFrame,)),
    LLMLogObserver: lambda: LLMLogObserver(),
    MetricsLogObserver: lambda: MetricsLogObserver(),
    TranscriptionLogObserver: lambda: TranscriptionLogObserver(),
    RTVIObserver: lambda: RTVIObserver(),
    GoogleRTVIObserver: lambda: GoogleRTVIObserver(None),
}

#: isinstance arguments that are not a class name in the module's namespace.
#: ``IdleFrameObserver`` tests against the tuple it was constructed with, which
#: is exactly what its declaration is built from.
UNRESOLVABLE = {"self._idle_timeout_frames"}


def _isinstance_frame_types(cls: type) -> set[str]:
    """Names every ``isinstance(data.frame, X)`` in this class's push handler.

    Walks the whole MRO, because a subclass handler that delegates upward --
    ``GoogleRTVIObserver`` does -- runs the base class's branches too.
    """
    names: set[str] = set()
    for ancestor in cls.__mro__:
        handler = ancestor.__dict__.get("on_push_frame")
        if handler is None:
            continue
        tree = ast.parse(inspect.getsource(handler).lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "isinstance" or len(node.args) != 2:
                continue
            if not ast.unparse(node.args[0]).endswith("frame"):
                continue
            argument = node.args[1]
            entries = argument.elts if isinstance(argument, (ast.Tuple, ast.List)) else [argument]
            names.update(ast.unparse(entry) for entry in entries)
    return names - UNRESOLVABLE


def _resolve(cls: type, name: str) -> type | None:
    """Find the class an isinstance argument names, from any handler's module."""
    for ancestor in cls.__mro__:
        module = sys.modules.get(ancestor.__module__)
        resolved = getattr(module, name, None)
        if isinstance(resolved, type):
            return resolved
    return None


@pytest.mark.parametrize("observer_class", list(OBSERVERS), ids=lambda cls: cls.__name__)
def test_the_declaration_admits_every_type_the_handler_branches_on(observer_class):
    declared = OBSERVERS[observer_class]().observed_frame_types
    branches = _isinstance_frame_types(observer_class)

    if declared is None:
        # Unfiltered is always safe; it is what an out-of-tree observer gets.
        return

    missing = []
    for name in sorted(branches):
        frame_type = _resolve(observer_class, name)
        assert frame_type is not None, f"cannot resolve {name}"
        if not issubclass(frame_type, declared):
            missing.append(name)

    assert not missing, (
        f"{observer_class.__name__}.on_push_frame branches on {missing}, which "
        f"observed_frame_types does not admit — those branches would never run"
    )


@pytest.mark.parametrize("observer_class", list(OBSERVERS), ids=lambda cls: cls.__name__)
def test_every_declared_type_is_a_frame(observer_class):
    declared = OBSERVERS[observer_class]().observed_frame_types
    if declared is None:
        return
    for entry in declared:
        assert issubclass(entry, Frame), entry


def test_an_unfiltered_debug_observer_still_sees_everything():
    """Its own contract: no frame_types means log every frame."""
    assert DebugLogObserver().observed_frame_types is None


def test_the_proxy_itself_declares_nothing():
    """``WorkerObserver`` is the fan-out, not a leaf; filtering it would be a bug."""
    assert WorkerObserver.observed_frame_types is None


def test_the_base_class_default_is_unfiltered():
    """An observer written against an older base must keep receiving everything."""
    assert BaseObserver.observed_frame_types is None
