from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriveLimits:
    max_forward_wheel_speed: float = 25.0
    max_reverse_wheel_speed: float = 8.0


DEFAULT_DRIVE_LIMITS = DriveLimits()

