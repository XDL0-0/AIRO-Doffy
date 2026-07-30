"""Compose optional gravity compensation with wrench filtering."""

from __future__ import annotations

from collections.abc import Iterable

from ...core.errors import ModelValidationError
from ...core.types import WrenchSample
from .compensation import GravityCompensator
from .filters import WrenchFilter


class WrenchProcessor:
    """Transform immutable raw samples while preserving their metadata."""

    def __init__(
        self,
        wrench_filter: WrenchFilter,
        *,
        gravity_compensator: GravityCompensator | None = None,
    ) -> None:
        self._filter = wrench_filter
        self._gravity = gravity_compensator

    def process(
        self,
        sample: WrenchSample,
        *,
        rotation_tool_to_base: Iterable[Iterable[object]] | None = None,
    ) -> WrenchSample:
        values = sample.values
        if self._gravity is not None:
            if rotation_tool_to_base is None:
                raise ModelValidationError(
                    "rotation_tool_to_base is required for gravity compensation"
                )
            values = self._gravity.compensate(values, rotation_tool_to_base)
        return WrenchSample(
            sequence=sample.sequence,
            source_timestamp_ns=sample.source_timestamp_ns,
            receive_timestamp_ns=sample.receive_timestamp_ns,
            clock_domain=sample.clock_domain,
            values=self._filter.process(values),
            frame_id=sample.frame_id,
        )

    def reset_filters(self) -> None:
        self._filter.reset()

    def add_baseline_sample(
        self,
        sample: WrenchSample,
        *,
        rotation_tool_to_base: Iterable[Iterable[object]],
    ) -> None:
        if self._gravity is None:
            raise ModelValidationError("no gravity compensator is configured")
        self._gravity.add_calibration_sample(
            sample.values,
            rotation_tool_to_base,
        )

    def finish_baseline_calibration(self) -> tuple[float, ...]:
        if self._gravity is None:
            raise ModelValidationError("no gravity compensator is configured")
        self._filter.reset()
        return self._gravity.finish_calibration()

    def reset_baseline(self, bias: Iterable[object] | None = None) -> None:
        if self._gravity is None:
            raise ModelValidationError("no gravity compensator is configured")
        self._gravity.reset_baseline(bias)
        self._filter.reset()
