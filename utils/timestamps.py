"""UTC timestamps for training metadata and recorded matches."""

from datetime import UTC, datetime


def utc_now() -> str:
    """Return the current UTC time as a timezone-aware ISO 8601 string."""
    return datetime.now(UTC).isoformat()
