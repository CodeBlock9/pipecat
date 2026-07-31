#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base AI service implementation.

Provides the foundation for all AI services in the Pipecat framework, including
model management, settings handling, and frame processing lifecycle methods.
"""

import copy
import warnings
from collections.abc import AsyncGenerator, Mapping
from dataclasses import fields as dataclass_fields
from typing import Any, get_args

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    ServiceMetadataFrame,
    ServiceSwitcherRequestMetadataFrame,
    StartFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import ServiceSettings, is_given


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _pydantic_model_type(annotation: Any) -> type[BaseModel] | None:
    candidates = get_args(annotation) or (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


class AIService(FrameProcessor):
    """Base class for all AI services.

    Provides common functionality for AI services including model management,
    settings handling, session properties, and frame processing lifecycle.
    Subclasses should implement specific AI functionality while leveraging
    this base infrastructure.
    """

    def __init__(self, settings: ServiceSettings | None = None, **kwargs):
        """Initialize the AI service.

        Args:
            settings: The runtime-updatable settings for the AI service.
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._settings: ServiceSettings = (
            settings
            # Here in case subclass doesn't implement more specific settings
            # (which hopefully should be rare)
            or ServiceSettings()
        )
        self._provider_options: dict[str, Any] = {}
        self._sync_model_name_to_metrics()
        self._session_properties: dict[str, Any] = {}
        self._tracing_enabled: bool = False
        self._tracing_context = None

    @property
    def provider_options(self) -> dict[str, Any]:
        """Return a defensive copy of advanced provider protocol options."""
        return copy.deepcopy(self._provider_options)

    def apply_provider_options(self, options: dict[str, Any] | None) -> None:
        """Apply advanced provider options before the service starts.

        Keys matching declared service settings update those settings so values
        used in URLs and connection setup are effective immediately. Unknown
        keys remain in ``settings.extra`` and the full mapping is retained for
        provider integrations to merge into their wire payloads.

        Args:
            options: Provider-specific request/session options.
        """
        self._provider_options = copy.deepcopy(options or {})
        if not self._provider_options:
            return

        delta = type(self._settings).from_mapping(self._provider_options)
        for settings_field in dataclass_fields(delta):
            if settings_field.name == "extra":
                continue
            incoming_value = getattr(delta, settings_field.name)
            current_value = getattr(self._settings, settings_field.name, None)
            if not is_given(incoming_value) or not is_given(current_value):
                continue
            if isinstance(current_value, BaseModel) and isinstance(
                incoming_value, (BaseModel, Mapping)
            ):
                incoming_mapping = (
                    incoming_value.model_dump(mode="python", exclude_unset=True)
                    if isinstance(incoming_value, BaseModel)
                    else dict(incoming_value)
                )
                merged = _deep_merge_dicts(
                    current_value.model_dump(mode="python"),
                    incoming_mapping,
                )
                setattr(delta, settings_field.name, type(current_value).model_validate(merged))
            elif isinstance(current_value, dict) and isinstance(
                incoming_value, (BaseModel, Mapping)
            ):
                incoming_mapping = (
                    incoming_value.model_dump(mode="python", exclude_unset=True)
                    if isinstance(incoming_value, BaseModel)
                    else dict(incoming_value)
                )
                setattr(
                    delta,
                    settings_field.name,
                    _deep_merge_dicts(current_value, incoming_mapping),
                )
            elif current_value is None and isinstance(incoming_value, Mapping):
                model_type = _pydantic_model_type(settings_field.type)
                if model_type is not None:
                    setattr(
                        delta,
                        settings_field.name,
                        model_type.model_validate(dict(incoming_value)),
                    )
        changed = self._settings.apply_update(delta)
        if "model" in changed:
            self._sync_model_name_to_metrics()

    def merge_provider_options(
        self,
        payload: dict[str, Any],
        *,
        include_declared: bool = True,
    ) -> dict[str, Any]:
        """Merge provider options over a wire payload.

        Raw values remain authoritative even when a key also maps to a declared
        setting. This preserves future nested fields that a local settings model
        does not yet understand and supports providers whose wire name differs
        from the generic setting used during connection setup.
        """
        options = self._provider_options if include_declared else self._settings.extra
        return _deep_merge_dicts(payload, options)

    def _sync_model_name_to_metrics(self):
        """Sync the current AI model name (in `self._settings.model`) for usage in metrics.

        We don't store model name here because there's already a single source
        of truth for it in `self._settings.model`. This method is just for
        syncing the model name to the metrics data.

        Args:
            model: The name of the AI model to use.
        """
        model = self._settings.model
        self.set_core_metrics_data(
            MetricsData(processor=self.name, model=model if isinstance(model, str) else "")
        )

    def service_metadata_frame(self) -> ServiceMetadataFrame | None:
        """The metadata frame this service broadcasts at start, or None.

        Override to return a populated
        :class:`~pipecat.frames.frames.ServiceMetadataFrame` (or a subtype such as
        ``STTMetadataFrame``) describing this service for downstream processors, for
        example the ``user_turn_strategies`` an STT with server-side end-of-turn
        detection recommends. Return None to broadcast nothing.

        Returns:
            The metadata frame to broadcast, or None.
        """
        return None

    async def broadcast_service_metadata(self):
        """Broadcast this service's metadata frame, if any."""
        frame = self.service_metadata_frame()
        if frame is not None:
            await self.broadcast_frame_instance(frame)

    async def start(self, frame: StartFrame):
        """Start the AI service.

        Called when the service should begin processing. Subclasses should
        override this method to perform service-specific initialization.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._settings.validate_complete()
        self._tracing_enabled = frame.enable_tracing
        self._tracing_context = frame.tracing_context

    async def stop(self, frame: EndFrame):
        """Stop the AI service on a graceful end (``EndFrame``).

        Runs in frame order, after pending frames drain. Override for graceful
        shutdown behavior, such as flushing in-flight work before stopping.

        Args:
            frame: The end frame.
        """
        pass

    async def cancel(self, frame: CancelFrame):
        """Cancel the AI service immediately (``CancelFrame``).

        Runs at once, ahead of any queued frames, to abort active work fast (for
        example, stop producing audio now). Override only for that time-sensitive
        subset.

        Args:
            frame: The cancel frame.
        """
        pass

    async def _update_settings(self, delta: ServiceSettings) -> dict[str, Any]:
        """Apply a settings delta and return the changed fields.

        The delta is applied to ``_settings`` and a dict mapping each changed
        field name to its **pre-update** value is returned.  The ``model``
        field is handled specially: when it changes, ``set_model_name`` is
        called.

        Concrete services should override this method (calling ``super()``)
        to react to specific changed fields (e.g. reconnect on voice change).

        Args:
            delta: A delta-mode settings object.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = self._settings.apply_update(delta)

        if "model" in changed:
            self._sync_model_name_to_metrics()

        if changed:
            logger.info(f"{self.name}: updated settings fields: {set(changed)}")

        return changed

    def _warn_init_param_moved_to_settings(
        self,
        param_name: str,
        settings_field: str | None = None,
        stacklevel: int = 3,
    ):
        """Warn that an ``__init__`` param has moved to ``Settings``.

        Emits a ``DeprecationWarning`` directing users to the canonical
        ``settings=ServiceClass.Settings(field=...)`` API.

        Args:
            param_name: Name of the deprecated ``__init__`` parameter.
            settings_field: The corresponding field on the ``Settings``
                dataclass, if different from *param_name*.  When ``None``
                the message omits the field hint.
            stacklevel: Stack depth for the warning.  Default ``3`` targets
                the caller's caller (i.e. user code that instantiated the
                service).
        """
        label = f"{type(self).__name__}.Settings"
        if settings_field:
            msg = (
                f"The `{param_name}` parameter is deprecated. "
                f"Use `settings={label}({settings_field}=...)` instead. "
                f"If both are provided, `settings` takes precedence."
            )
        else:
            msg = (
                f"The `{param_name}` parameter is deprecated. "
                f"Use `settings={label}(...)` instead. "
                f"If both are provided, `settings` takes precedence."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)

    def _warn_unhandled_updated_settings(self, unhandled):
        """Log a warning for settings changes that won't take effect at runtime.

        Convenience helper for ``_update_settings`` overrides.  Accepts any
        iterable of field names (a ``dict``, ``set``, ``dict_keys``, etc.).

        Args:
            unhandled: Field names that changed but are not applied.
        """
        if unhandled:
            fields = ", ".join(sorted(unhandled))
            logger.warning(f"{self.name}: runtime update of [{fields}] is not currently supported")

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and broadcast service metadata once the service starts.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

        # Broadcast metadata after StartFrame goes downstream, so downstream sees
        # StartFrame first.
        if isinstance(frame, StartFrame):
            await self.broadcast_service_metadata()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle service lifecycle.

        Automatically handles StartFrame, EndFrame, and CancelFrame by calling
        the appropriate lifecycle methods.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
        elif isinstance(frame, ServiceSwitcherRequestMetadataFrame):
            await self.broadcast_service_metadata()

    async def process_generator(self, generator: AsyncGenerator[Frame | None, None]):
        """Process frames from an async generator.

        Takes an async generator that yields frames and processes each one,
        handling error frames specially by pushing them as errors.

        Args:
            generator: An async generator that yields Frame objects or None.
        """
        async for f in generator:
            if f:
                if isinstance(f, ErrorFrame):
                    await self.push_error_frame(f)
                else:
                    await self.push_frame(f)

    async def _start(self, frame: StartFrame):
        try:
            await self.start(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")

    async def _stop(self, frame: EndFrame):
        try:
            await self.stop(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")

    async def _cancel(self, frame: CancelFrame):
        try:
            await self.cancel(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")
