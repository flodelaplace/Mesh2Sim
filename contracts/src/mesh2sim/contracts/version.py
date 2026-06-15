"""Schema versioning.

A single semver string covers every contract in this package. The minor digit bumps when we
add optional fields or capabilities; the major digit bumps when we make a breaking change to
the wire format.
"""

from __future__ import annotations

SCHEMA_VERSION: str = "0.1.0"
"""Current schema version (semver: MAJOR.MINOR.PATCH)."""


def parse_semver(v: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH``. Raises ValueError on malformed input."""
    parts = v.split(".")
    if len(parts) != 3:
        raise ValueError(f"malformed semver {v!r} (expected MAJOR.MINOR.PATCH)")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as e:
        raise ValueError(f"malformed semver {v!r}: {e}") from None


def is_compatible(version: str, *, current: str = SCHEMA_VERSION) -> bool:
    """Compatibility rule: identical major required, any minor tolerated.

    Reading a file written by an older minor is fine (optional fields may be missing).
    Reading a file written by a newer minor is also fine (unknown optional fields are
    ignored or surface as None). A different major means the wire format changed and we
    refuse to load.
    """
    a = parse_semver(version)
    b = parse_semver(current)
    return a[0] == b[0]


def check_compatible(version: str, *, current: str = SCHEMA_VERSION) -> None:
    """Raise ValueError if ``version`` is not compatible with ``current``."""
    if not is_compatible(version, current=current):
        raise ValueError(
            f"incompatible schema_version {version!r} (current {current!r}); "
            "major mismatch — wire format changed"
        )
