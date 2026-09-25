#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for service settings and initialization patterns.

Settings objects operate in two modes:

- **Store mode** (``self._settings``): the live state inside a service.
  Every field must hold a real value (``None`` is fine, ``NOT_GIVEN`` is not).
- **Delta mode** (``FooSettings()`` with no args): a sparse update.
  Every field must default to ``NOT_GIVEN`` so ``apply_update()`` skips
  untouched fields and doesn't accidentally overwrite the store.

These tests verify both sides of that contract automatically:

1. **Delta defaults** — Instantiate every ``ServiceSettings`` subclass with
   no arguments and assert that every field is ``NOT_GIVEN``.  Catches the
   bug where a field defaults to ``None`` instead of ``NOT_GIVEN``, which
   would cause partial deltas to silently overwrite unrelated store values.

2. **Store completeness** — Instantiate every concrete service with dummy
   args and assert that ``_settings`` contains no ``NOT_GIVEN`` values.
   This is the same check that ``validate_complete()`` runs in ``start()``,
   but caught here at unit-test time without needing a running pipeline.
   Catches services that forget to initialize a field in ``default_settings``.

All Settings and Service classes are auto-discovered via ``pkgutil``;
new services are covered automatically with no per-service maintenance.
Only classes defined in pipecat itself are swept: the stand-ins other test
modules define are not services, and must not change what this module checks.
"""

import importlib
import inspect
import pkgutil
import sys
from dataclasses import fields

import pytest

import pipecat.services
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.utils.types import is_given

# Modules that define abstract base service classes (not concrete services).
_BASE_MODULES = frozenset(
    {
        "pipecat.services.ai_service",
        "pipecat.services.llm_service",
        "pipecat.services.stt_service",
        "pipecat.services.tts_service",
        "pipecat.services.image_service",
        "pipecat.services.vision_service",
    }
)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------


def _all_subclasses(cls):
    """Every subclass of ``cls`` that pipecat itself defines."""
    result = set()
    for sub in cls.__subclasses__():
        if sub.__module__.startswith("pipecat."):
            result.add(sub)
        result.update(_all_subclasses(sub))
    return result


def _import_all_service_modules() -> dict[str, BaseException]:
    """Import every module under pipecat.services, returning the ones that failed.

    A module whose optional dependency is not installed cannot be imported, and
    its services drop out of the sweep. The failures are returned so that
    ``test_only_missing_optional_dependencies_block_an_import`` can tell that
    case from a module that is broken.
    """
    failures: dict[str, BaseException] = {}
    package = pipecat.services
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__,
        prefix=package.__name__ + ".",
        onerror=lambda name: failures.setdefault(name, sys.exc_info()[1]),
    ):
        try:
            importlib.import_module(modname)
        except Exception as e:
            failures[modname] = e
    return failures


IMPORT_FAILURES = _import_all_service_modules()

ALL_SETTINGS_CLASSES = sorted(_all_subclasses(ServiceSettings), key=lambda c: c.__qualname__)
assert ALL_SETTINGS_CLASSES, "No settings classes discovered"


# ---------------------------------------------------------------------------
# Service instantiation helpers
# ---------------------------------------------------------------------------


# Dummy credentials and endpoints, passed wherever a constructor names them,
# so that services which refuse to build without them are still swept.
_DUMMY_ARGS = {
    "api_key": "test",
    "region": "eastus",
    "endpoint": "https://example.invalid",
    "base_url": "https://example.invalid/v1",
}


def _try_instantiate(cls):
    """Instantiate a service with dummy values.

    Passes "test" for every required parameter, and the ``_DUMMY_ARGS`` value
    for each of their names the signature has. A constructor that takes
    ``**kwargs`` but does not name ``api_key`` (the OpenAI family reads it from
    there) gets a dummy ``api_key`` too, unless it rejects the keyword (Vertex
    services refuse one in favour of credentials).
    """
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    takes_kwargs = False
    for name, param in sig.parameters.items():
        if name == "self" or param.kind is param.VAR_POSITIONAL:
            continue
        if param.kind is param.VAR_KEYWORD:
            takes_kwargs = True
        elif name in _DUMMY_ARGS:
            kwargs[name] = _DUMMY_ARGS[name]
        elif param.default is param.empty:
            kwargs[name] = "test"
    if takes_kwargs and "api_key" not in kwargs:
        try:
            return cls(**kwargs, api_key=_DUMMY_ARGS["api_key"])
        except (TypeError, ValueError) as e:
            if "api_key" not in str(e):
                raise
    return cls(**kwargs)


def _concrete_service_classes():
    """Return concrete (non-abstract) service classes under pipecat.services.

    Pure class-hierarchy walk with no instantiation, so collection stays fast.
    Construction happens inside the test, which skips services that can't be
    built with dummy args (e.g. those that need a local model or real files).
    """
    return [
        cls
        for cls in sorted(_all_subclasses(AIService), key=lambda c: c.__qualname__)
        if cls.__module__.startswith("pipecat.services.")
        and not inspect.isabstract(cls)
        # The framework's own base classes are not services.
        and cls.__module__ not in _BASE_MODULES
    ]


ALL_SERVICE_CLASSES = _concrete_service_classes()
assert ALL_SERVICE_CLASSES, "No service classes discovered"


# ---------------------------------------------------------------------------
# 0. Discovery: an import that fails must be a missing optional dependency
# ---------------------------------------------------------------------------


def _missing_third_party_module(exc: BaseException) -> str | None:
    """Name the non-pipecat module whose absence caused ``exc``, if any."""
    seen: BaseException | None = exc
    while seen is not None:
        if (
            isinstance(seen, ModuleNotFoundError)
            and seen.name
            and seen.name.split(".")[0] != "pipecat"
        ):
            return seen.name
        seen = seen.__cause__ or seen.__context__
    return None


def test_only_missing_optional_dependencies_block_an_import():
    """A service module that fails to import for any other reason is broken.

    Its services would otherwise leave the sweep while every test stays green.
    """
    broken = {
        modname: f"{type(e).__name__}: {e}"
        for modname, e in IMPORT_FAILURES.items()
        if _missing_third_party_module(e) is None
    }
    assert not broken, broken


# ---------------------------------------------------------------------------
# 1. Settings defaults: delta-mode safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("settings_cls", ALL_SETTINGS_CLASSES, ids=lambda c: c.__qualname__)
def test_delta_defaults_are_not_given(settings_cls):
    """Every field must default to NOT_GIVEN so empty deltas are no-ops.

    A field that defaults to None instead of NOT_GIVEN will cause
    apply_update() to overwrite the corresponding store value whenever
    a partial delta is applied.
    """
    instance = settings_cls()
    for f in fields(instance):
        if f.name == "extra":
            continue
        val = getattr(instance, f.name)
        assert not is_given(val), (
            f"{settings_cls.__qualname__}.{f.name} defaults to {val!r}, expected NOT_GIVEN"
        )


# ---------------------------------------------------------------------------
# 2. Service construction: store-mode completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", ALL_SERVICE_CLASSES, ids=lambda c: c.__qualname__)
async def test_service_settings_complete(service_cls):
    """After construction, _settings must have no NOT_GIVEN values.

    This is what validate_complete() checks in start().  Catching it
    here means we don't need a running pipeline to find missing defaults.
    Construction runs inside an event loop, as it does in an app: the Google
    clients ask for the current loop, and after an earlier test module has
    closed its own there is none outside one.
    """
    try:
        svc = _try_instantiate(service_cls)
    except Exception:
        pytest.skip("Cannot instantiate with dummy args (needs real args or files)")
    for f in fields(svc._settings):
        if f.name == "extra":
            continue
        val = getattr(svc._settings, f.name)
        assert is_given(val), (
            f"{service_cls.__qualname__}._settings.{f.name} is NOT_GIVEN after construction"
        )
