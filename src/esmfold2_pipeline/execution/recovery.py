from __future__ import annotations

from datetime import datetime, timedelta, timezone


MIN_STALE_RECOVERY_SECONDS = 90.0
STALE_RECOVERY_HEARTBEAT_MULTIPLIER = 3.0


def resolve_stale_after_seconds(
    *,
    heartbeat_interval_seconds: float,
    stale_after_seconds: float | None,
) -> float:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    if stale_after_seconds is None:
        return max(
            MIN_STALE_RECOVERY_SECONDS,
            STALE_RECOVERY_HEARTBEAT_MULTIPLIER * heartbeat_interval_seconds,
        )
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    return stale_after_seconds


def stale_before_timestamp(stale_after_seconds: float) -> str:
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    return stale_before.isoformat(timespec="milliseconds").replace("+00:00", "Z")
