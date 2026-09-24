#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCPClient lists every page of an MCP server's tool catalog.

``tools/list`` is paginated: a result carries ``nextCursor`` while more tools
remain, and the client asks again with that cursor. The fake session below
answers with the MCP SDK's own ``ListToolsResult`` and ``Tool`` types and takes
the cursor the way ``ClientSession.list_tools`` does, as the deprecated
``cursor=`` or as ``params=PaginatedRequestParams(...)``. The transport and
session are patched as ``tests/test_mcp_service.py`` patches them.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

import pytest

# MCP is an optional dependency (the `mcp` extra); skip the whole module if it
# isn't installed.
pytest.importorskip("mcp")

from mcp import types  # noqa: E402
from mcp.client.session_group import StreamableHttpParameters  # noqa: E402

from pipecat.services.mcp_service import MCPClient  # noqa: E402


def _tool(name, description=None):
    return types.Tool(
        name=name,
        description=description or f"{name} tool",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


class _Transport:
    async def __aenter__(self):
        return (MagicMock(), MagicMock(), MagicMock())

    async def __aexit__(self, *exc):
        return False


class _PagedSession:
    """Answers tools/list from a map of cursor to page, recording each cursor asked for."""

    def __init__(self, pages):
        self._pages = pages
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        pass

    async def list_tools(self, cursor=None, *, params=None):
        if params is not None and cursor is not None:
            raise ValueError("Cannot specify both cursor and params")
        key = params.cursor if params is not None else cursor
        self.requests.append(key)
        # A real request yields to the event loop, so a caller's timeout can end
        # a listing that never stops; the fake does the same.
        await asyncio.sleep(0)
        return self._pages[key]


TWO_PAGES = {
    None: types.ListToolsResult(
        tools=[_tool("lookup_order"), _tool("check_hours")], nextCursor="page-2"
    ),
    "page-2": types.ListToolsResult(tools=[_tool("book_table")], nextCursor=None),
}

ONE_PAGE = {
    None: types.ListToolsResult(tools=[_tool("lookup_order"), _tool("check_hours")]),
}

# A server that ignores the cursor: every request gets page 1 and the same nextCursor.
CURSOR_IGNORED = {
    None: TWO_PAGES[None],
    "page-2": TWO_PAGES[None],
}

# A well-paged server that lists one tool name on two pages.
NAME_ON_TWO_PAGES = {
    None: types.ListToolsResult(
        tools=[_tool("lookup_order", "the first listing")], nextCursor="page-2"
    ),
    "page-2": types.ListToolsResult(
        tools=[_tool("lookup_order", "the second listing"), _tool("book_table")],
        nextCursor=None,
    ),
}


class TestMCPToolPages(unittest.IsolatedAsyncioTestCase):
    def _client(self, pages, **kwargs):
        session = _PagedSession(pages)
        ctx = patch.multiple(
            "pipecat.services.mcp_service",
            streamable_http_client=lambda *a, **k: _Transport(),
            ClientSession=lambda read, write: session,
        )
        ctx.start()
        self.addCleanup(ctx.stop)
        client = MCPClient(server_params=StreamableHttpParameters(url="http://test/mcp"), **kwargs)
        self.addAsyncCleanup(client.close)
        return client, session

    async def _schema_names(self, client):
        # The read Mesa's McpToolSession.start and tool discovery make.
        await client.start()
        with self.assertWarns(DeprecationWarning):
            schema = await client.get_tools_schema()
        return [t.name for t in schema.standard_tools]

    async def test_get_tools_schema_reads_every_page(self):
        client, session = self._client(TWO_PAGES)

        names = await self._schema_names(client)

        self.assertEqual(names, ["lookup_order", "check_hours", "book_table"])
        self.assertEqual(session.requests, [None, "page-2"])

    async def test_live_tools_read_every_page(self):
        client, session = self._client(TWO_PAGES)

        schema = await client.tools()

        self.assertEqual(
            [t.name for t in schema.standard_tools], ["lookup_order", "check_hours", "book_table"]
        )
        self.assertTrue(all(t.handler is not None for t in schema.standard_tools))
        self.assertEqual(session.requests, [None, "page-2"])

    async def test_filter_keeps_a_tool_from_a_later_page(self):
        client, _ = self._client(TWO_PAGES, tools_filter=["book_table"])

        names = await self._schema_names(client)

        self.assertEqual(names, ["book_table"])

    async def test_control_one_page(self):
        client, session = self._client(ONE_PAGE)

        names = await self._schema_names(client)

        self.assertEqual(names, ["lookup_order", "check_hours"])
        self.assertEqual(session.requests, [None])

    async def test_a_server_that_repeats_a_cursor_ends_with_each_tool_once(self):
        client, session = self._client(CURSOR_IGNORED)

        # Bounded here: without the repeated-cursor stop the listing never ends.
        names = await asyncio.wait_for(self._schema_names(client), timeout=5)

        self.assertEqual(names, ["lookup_order", "check_hours"])
        self.assertEqual(session.requests, [None, "page-2"])

    async def test_a_name_listed_on_two_pages_keeps_its_first_listing(self):
        client, _ = self._client(NAME_ON_TWO_PAGES)

        schema = await client.tools()

        self.assertEqual([t.name for t in schema.standard_tools], ["lookup_order", "book_table"])
        self.assertEqual(schema.standard_tools[0].description, "the first listing")
