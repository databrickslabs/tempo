"""Internal helpers for emitting deprecation warnings.

Tempo v0.2 keeps the v0.1.x public API working through thin compatibility
shims that emit a :class:`DeprecationWarning`. All shimmed APIs are scheduled
for removal in v1.0.0. See ``MIGRATION_GUIDE.md`` and
``BACKWARDS_COMPATIBILITY_PLAN.md`` for the full mapping.
"""

import warnings

# Version in which deprecated v0.1.x APIs will be removed.
REMOVAL_VERSION = "v1.0.0"


def warn_deprecated(old: str, new: str, stacklevel: int = 3) -> None:
    """Emit a standardized ``DeprecationWarning`` for a v0.1.x API.

    :param old: description of the deprecated API (e.g. ``"TSDF.vwap()"`` or
        ``"the 'partition_cols' parameter"``).
    :param new: the recommended replacement (e.g. ``"tempo.stats.vwap()"``).
    :param stacklevel: how far up the call stack to attribute the warning so it
        points at the user's call site rather than this helper. Callers invoked
        directly from user code should use the default of ``3``.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in {REMOVAL_VERSION}. "
        f"Use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
