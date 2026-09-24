#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic hands every ``toolUse`` event to the function-call runner or skips it.

Nova Sonic streams one whole tool call per ``toolUse`` event:
``{"toolUse": {"toolName", "toolUseId", "content": <JSON string>}}``. Anything
``_handle_tool_use_event`` raises reaches the receive loop, whose ``except``
pushes an error and resets the whole session mid-call. So:

- arguments that do not parse are skipped with the chat loop's warning,
  "Failed to parse function call arguments";
- a function name nobody registered goes to ``run_function_calls`` like any
  other, whose missing-function path answers the model with a tool result.

The service is imported with ``pytest.importorskip`` so the module is skipped
rather than failing collection when the optional AWS dependencies aren't
installed.
"""

import unittest
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext


def _service():
    mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
    service = mod.AWSNovaSonicLLMService(
        secret_access_key="test", access_key_id="test", region="us-east-1"
    )
    service._content_being_received = True
    service._context = LLMContext()
    service._report_user_transcription_ended = AsyncMock()
    service.run_function_calls = AsyncMock()

    async def transfer_call(params):
        await params.result_callback({"ok": True})

    service.register_function("transfer_call", transfer_call)
    return service


def _tool_use(name, content):
    return {"toolUse": {"toolName": name, "toolUseId": "tooluse_rec_1", "content": content}}


def _dispatched(service):
    (calls,) = service.run_function_calls.await_args.args
    return [(c.function_name, c.tool_call_id, c.arguments) for c in calls]


class TestNovaSonicToolArguments(unittest.IsolatedAsyncioTestCase):
    async def test_unparseable_arguments_skip_the_call_without_raising(self):
        service = _service()

        with patch("pipecat.services.aws.nova_sonic.llm.logger") as mock_logger:
            await service._handle_tool_use_event(
                _tool_use("transfer_call", '{"destination": "+614')
            )

        service.run_function_calls.assert_not_awaited()
        (message,) = mock_logger.warning.call_args.args
        self.assertIn('Failed to parse function call arguments: {"destination": "+614', message)

    async def test_control_parseable_arguments_dispatch(self):
        service = _service()

        await service._handle_tool_use_event(
            _tool_use("transfer_call", '{"destination": "+61400000000"}')
        )

        service.run_function_calls.assert_awaited_once()
        self.assertEqual(
            _dispatched(service),
            [("transfer_call", "tooluse_rec_1", {"destination": "+61400000000"})],
        )

    async def test_empty_arguments_dispatch_as_an_empty_object(self):
        service = _service()

        await service._handle_tool_use_event(_tool_use("transfer_call", ""))

        service.run_function_calls.assert_awaited_once()
        self.assertEqual(_dispatched(service), [("transfer_call", "tooluse_rec_1", {})])


class TestNovaSonicUnknownTool(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_function_name_goes_to_the_runner_without_raising(self):
        service = _service()

        await service._handle_tool_use_event(_tool_use("book_table", '{"party": 2}'))

        service.run_function_calls.assert_awaited_once()
        self.assertEqual(_dispatched(service), [("book_table", "tooluse_rec_1", {"party": 2})])

    async def test_control_registered_name_dispatches(self):
        service = _service()

        await service._handle_tool_use_event(_tool_use("transfer_call", '{"party": 2}'))

        service.run_function_calls.assert_awaited_once()
        self.assertEqual(_dispatched(service), [("transfer_call", "tooluse_rec_1", {"party": 2})])
