#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pytest configuration for the CLI test suite.

The CLI lives behind the optional ``pipecat-ai[cli]`` extra (typer, rich,
jinja2, questionary, ruff). When that extra is not installed, skip collecting
``tests/cli`` entirely instead of failing collection with ImportErrors, so a
lean ``uv run pytest`` still works. CI installs the ``cli`` extra so these run.
"""

import importlib.util

import pytest

# Skip the whole directory if the CLI dependencies are not available.
if importlib.util.find_spec("questionary") is None:
    collect_ignore_glob = ["*"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_context_hub: run init's real Context Hub setup instead of the stub every other "
        "CLI test gets",
    )


@pytest.fixture(autouse=True)
def _isolate_context_hub(request, monkeypatch, tmp_path):
    """Keep ``pipecat init`` away from the real Context Hub.

    ``init`` registers the hub's MCP server by running
    ``python -m pipecat_context_hub install``, which writes to the coding agents'
    configuration on the machine running the tests. Every CLI test gets a stub
    for that setup and a throwaway hub data directory. The tests of the setup
    itself carry ``real_context_hub`` and stub what lies beneath it.
    """
    monkeypatch.setenv("PIPECAT_HUB_DATA_DIR", str(tmp_path / "hub"))
    if request.node.get_closest_marker("real_context_hub") is None:
        monkeypatch.setattr("pipecat.cli.commands.init._setup_context_hub", lambda: None)
