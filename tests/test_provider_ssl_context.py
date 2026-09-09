#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider clients verify against one TLS context, not one each.

A call builds one to three OpenAI-compatible clients and keeps them for its
whole length. httpx builds an `ssl.SSLContext` per `AsyncClient` and loads the
system trust store into each, so the store was being parsed and held once per
client per call.
"""

import ssl
import unittest
from unittest.mock import patch

import pipecat.services.openai.base_llm as base_llm
from pipecat.services.openai.base_llm import BaseOpenAILLMService, shared_ssl_context


class TestSharedSSLContext(unittest.TestCase):
    def test_the_context_is_built_once_and_reused(self):
        first = shared_ssl_context()
        second = shared_ssl_context()

        self.assertIsInstance(first, ssl.SSLContext)
        self.assertIs(first, second)

    def test_a_client_verifies_against_the_shared_context(self):
        service = BaseOpenAILLMService(api_key="not-a-key", model="gpt-4o")

        captured = {}
        original = base_llm.DefaultAsyncHttpxClient

        def recording(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        with patch.object(base_llm, "DefaultAsyncHttpxClient", recording):
            service.create_client(api_key="not-a-key")

        self.assertIs(captured["verify"], shared_ssl_context())

    def test_the_context_is_not_built_at_import(self):
        """A process that never talks to a provider should not load a trust store."""
        with patch.object(base_llm, "_SSL_CONTEXT", None):
            with patch("httpx.create_ssl_context") as create:
                self.assertEqual(create.call_count, 0)
                base_llm.shared_ssl_context()
                self.assertEqual(create.call_count, 1)
                base_llm.shared_ssl_context()
                self.assertEqual(create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
