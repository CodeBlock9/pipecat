#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Merging provider options over a request copies the options, not the request."""

from pipecat.services.ai_service import _deep_merge_dicts


def test_the_merge_does_not_copy_the_payload():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    payload = {"messages": messages, "tools": tools, "stream_options": {"include_usage": True}}

    merged = _deep_merge_dicts(payload, {})
    assert merged is not payload
    assert merged["messages"] is messages
    assert merged["tools"] is tools

    merged = _deep_merge_dicts(payload, {"stream_options": {"include_obfuscation": False}})
    assert merged["messages"] is messages
    assert merged["stream_options"] == {"include_usage": True, "include_obfuscation": False}
    assert payload["stream_options"] == {"include_usage": True}  # the base is never mutated


def test_control_override_values_are_copied():
    override = {"extra_body": {"thinking": {"type": "disabled"}}}

    merged = _deep_merge_dicts({"model": "m"}, override)
    merged["extra_body"]["thinking"]["type"] = "enabled"

    assert override["extra_body"]["thinking"]["type"] == "disabled"
