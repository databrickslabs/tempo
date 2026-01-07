# Backwards Compatibility Plan for v0.2 Integration

This plan outlines how to make the v0.2-integration branch backwards compatible with v0.1.x code through deprecation warnings and feature flags.

---

## Goals

1. Existing v0.1.x code continues to work with deprecation warnings
2. Users have time to migrate before breaking changes take effect
3. Clear deprecation timeline communicated to users

---

## Implementation Strategy

### Phase 1: Add Compatibility Shims (v0.2.0)
- Old APIs work but emit deprecation warnings
- Feature flags control behavior

### Phase 2: Default to New Behavior (v0.2.x)
- Feature flags default to new behavior
- Old APIs still work with warnings

### Phase 3: Remove Old APIs (v0.3.0)
- Breaking change release
- Old APIs removed

---

## 1. TSDF Constructor Compatibility

### File: `python/tempo/tsdf.py`

Add `partition_cols` as deprecated alias for `series_ids`:

```python
import warnings
from typing import Optional, Collection

def __init__(
    self,
    df: DataFrame,
    ts_schema: Optional[TSSchema] = None,
    ts_col: Optional[str] = None,
    series_ids: Optional[Collection[str]] = None,
    partition_cols: Optional[Collection[str]] = None,  # DEPRECATED
    sequence_col: Optional[str] = None,  # DEPRECATED
    resample_freq: Optional[str] = None,
    resample_func: Optional[Union[Callable, str]] = None,
) -> None:
    # Handle deprecated partition_cols parameter
    if partition_cols is not None:
        warnings.warn(
            "The 'partition_cols' parameter is deprecated and will be removed in v0.3.0. "
            "Use 'series_ids' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if series_ids is not None:
            raise ValueError("Cannot specify both 'partition_cols' and 'series_ids'")
        series_ids = list(partition_cols)

    # Handle deprecated sequence_col parameter
    if sequence_col is not None:
        warnings.warn(
            "The 'sequence_col' parameter is deprecated and will be removed in v0.3.0. "
            "Use TSDF.fromSubsequenceCol() factory method instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Create SubsequenceTSIndex if sequence_col provided
        # ... implementation details

    # Rest of constructor...
```

### Add `partitionCols` Property Alias

```python
@property
def partitionCols(self) -> list[str]:
    """Deprecated: Use series_ids instead."""
    warnings.warn(
        "The 'partitionCols' attribute is deprecated and will be removed in v0.3.0. "
        "Use 'series_ids' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.series_ids

@property
def sequence_col(self) -> Optional[str]:
    """Deprecated: sequence_col is no longer supported."""
    warnings.warn(
        "The 'sequence_col' attribute is deprecated and will be removed in v0.3.0. "
        "Use ts_schema.subsequence_col for SubsequenceTSIndex.",
        DeprecationWarning,
        stacklevel=2
    )
    # Return empty string for backwards compatibility
    return ""
```

---

## 2. Import Path Compatibility

### File: `python/tempo/resample.py`

Re-export moved functions with deprecation warnings:

```python
import warnings

def _deprecated_import(name: str, new_module: str):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Importing '{name}' from 'tempo.resample' is deprecated. "
            f"Import from '{new_module}' instead. This will be removed in v0.3.0.",
            DeprecationWarning,
            stacklevel=2
        )
        from tempo import resample_utils
        return getattr(resample_utils, name)(*args, **kwargs)
    return wrapper

# Re-export with deprecation warnings
floor = _deprecated_import("floor", "tempo.resample_utils")
ceiling = _deprecated_import("ceiling", "tempo.resample_utils")
min = _deprecated_import("min", "tempo.resample_utils")
max = _deprecated_import("max", "tempo.resample_utils")
average = _deprecated_import("average", "tempo.resample_utils")
checkAllowableFreq = _deprecated_import("checkAllowableFreq", "tempo.resample_utils")
FreqDict = _deprecated_import("FreqDict", "tempo.resample_utils")
```

### File: `python/tempo/utils.py`

Re-export moved function:

```python
import warnings

def calculate_time_horizon(*args, **kwargs):
    """Deprecated: Use tempo.resample.calculate_time_horizon instead."""
    warnings.warn(
        "Importing 'calculate_time_horizon' from 'tempo.utils' is deprecated. "
        "Import from 'tempo.resample' instead. This will be removed in v0.3.0.",
        DeprecationWarning,
        stacklevel=2
    )
    from tempo.resample import calculate_time_horizon as _calculate_time_horizon
    return _calculate_time_horizon(*args, **kwargs)
```

---

## 3. Interpolation API Compatibility

### File: `python/tempo/interpol.py`

Keep the `Interpolation` class as deprecated wrapper:

```python
import warnings

class Interpolation:
    """
    Deprecated: Use the interpolate() function instead.

    This class is maintained for backwards compatibility and will be
    removed in v0.3.0.
    """

    def __init__(self, is_resampled: bool = False):
        warnings.warn(
            "The Interpolation class is deprecated and will be removed in v0.3.0. "
            "Use the interpolate() function instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.is_resampled = is_resampled

    def interpolate(
        self,
        tsdf,
        partition_cols=None,  # ignored, extracted from tsdf
        target_cols=None,
        freq=None,  # ignored
        ts_col=None,  # ignored, extracted from tsdf
        func=None,  # ignored
        method="ffill",
        show_interpolated=False,  # ignored
    ):
        """Deprecated wrapper that calls the new interpolate function."""
        warnings.warn(
            "Interpolation.interpolate() is deprecated. "
            "Use the interpolate() function directly.",
            DeprecationWarning,
            stacklevel=2
        )

        # Map old method strings to new functions
        method_map = {
            "ffill": forward_fill,
            "bfill": backward_fill,
            "zero": zero_fill,
            "linear": "linear",
            "null": "null",
        }

        mapped_method = method_map.get(method, method)

        return interpolate(
            tsdf,
            target_cols=target_cols,
            method=mapped_method,
            leading_margin=0,
            lagging_margin=0,
        )
```

---

## 4. As-Of Join Compatibility

### File: `python/tempo/tsdf.py`

Support deprecated `sql_join_opt` parameter:

```python
def asofJoin(
    self,
    right_tsdf: TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: Optional[bool] = None,  # DEPRECATED
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    strategy: Optional[str] = None,
) -> TSDF:
    # Handle deprecated sql_join_opt parameter
    if sql_join_opt is not None:
        warnings.warn(
            "The 'sql_join_opt' parameter is deprecated and will be removed in v0.3.0. "
            "Use 'strategy=\"broadcast\"' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if strategy is not None:
            raise ValueError("Cannot specify both 'sql_join_opt' and 'strategy'")
        if sql_join_opt:
            strategy = 'broadcast'

    # Rest of implementation...
```

---

## 5. Feature Flags

### File: `python/tempo/config.py` (NEW)

Create a configuration module for feature flags:

```python
"""Tempo configuration and feature flags."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TempoConfig:
    """Configuration settings for Tempo behavior."""

    # When True, use new v0.2 behavior; when False, use v0.1 compatibility mode
    use_new_interpolation_api: bool = True
    use_new_join_strategies: bool = True
    emit_deprecation_warnings: bool = True

    # Environment variable overrides
    @classmethod
    def from_environment(cls) -> "TempoConfig":
        return cls(
            use_new_interpolation_api=os.getenv("TEMPO_NEW_INTERPOLATION", "true").lower() == "true",
            use_new_join_strategies=os.getenv("TEMPO_NEW_JOIN_STRATEGIES", "true").lower() == "true",
            emit_deprecation_warnings=os.getenv("TEMPO_DEPRECATION_WARNINGS", "true").lower() == "true",
        )

# Global config instance
_config: Optional[TempoConfig] = None

def get_config() -> TempoConfig:
    global _config
    if _config is None:
        _config = TempoConfig.from_environment()
    return _config

def configure(**kwargs) -> None:
    """Configure Tempo behavior."""
    global _config
    if _config is None:
        _config = TempoConfig()
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown config option: {key}")
```

### Usage in Code

```python
from tempo.config import get_config

def some_function():
    config = get_config()
    if config.emit_deprecation_warnings:
        warnings.warn("...", DeprecationWarning, stacklevel=2)
```

---

## 6. Removed Methods - Stub Implementations

### File: `python/tempo/tsdf.py`

Add stubs for removed methods that raise helpful errors:

```python
def vwap(self, *args, **kwargs):
    """Removed in v0.2. Use tempo.stats.vwap() instead."""
    raise NotImplementedError(
        "The vwap() method was removed in v0.2.0. "
        "Use tempo.stats.vwap() function instead, or implement using "
        "tsdf.withColumn() with appropriate VWAP calculation."
    )

def EMA(self, *args, **kwargs):
    """Removed in v0.2."""
    raise NotImplementedError(
        "The EMA() method was removed in v0.2.0. "
        "Implement using tsdf.rollingApply() with an EMA function, or use "
        "pyspark.sql.functions for exponential moving average calculations."
    )

def withLookbackFeatures(self, *args, **kwargs):
    """Removed in v0.2."""
    raise NotImplementedError(
        "The withLookbackFeatures() method was removed in v0.2.0. "
        "Use tsdf.rollingApply() or tsdf.rollingAgg() instead."
    )

def withRangeStats(self, *args, **kwargs):
    """Removed in v0.2."""
    raise NotImplementedError(
        "The withRangeStats() method was removed in v0.2.0. "
        "Use tsdf.rollingAgg() with appropriate aggregation functions."
    )

def withGroupedStats(self, *args, **kwargs):
    """Removed in v0.2."""
    raise NotImplementedError(
        "The withGroupedStats() method was removed in v0.2.0. "
        "Use tsdf.aggBySeries() instead."
    )
```

---

## 7. Type Compatibility

### File: `python/tempo/resample_utils.py`

Add backwards-compatible return type handling:

```python
import warnings
from tempo.config import get_config

def checkAllowableFreq(freq: str) -> tuple:
    """
    Check if frequency string is valid.

    Returns:
        tuple: (period, unit) where period is int in v0.2+, str in compatibility mode
    """
    # ... validation logic ...

    period_int = int(period_str)

    config = get_config()
    if config.use_legacy_types:
        warnings.warn(
            "Returning string period from checkAllowableFreq() is deprecated. "
            "In v0.3.0, this will always return (int, str).",
            DeprecationWarning,
            stacklevel=2
        )
        return (period_str, unit)  # Legacy: (str, str)

    return (period_int, unit)  # New: (int, str)
```

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `python/tempo/tsdf.py` | Add deprecated params, property aliases, method stubs |
| `python/tempo/resample.py` | Re-export functions with deprecation warnings |
| `python/tempo/resample_utils.py` | Add type compatibility option |
| `python/tempo/utils.py` | Re-export calculate_time_horizon |
| `python/tempo/interpol.py` | Keep Interpolation class as deprecated wrapper |
| `python/tempo/config.py` | **NEW** - Feature flags module |
| `python/tempo/__init__.py` | Export config module |

---

## 9. Testing Strategy

### Add Deprecation Warning Tests

```python
import warnings
import pytest

class TestDeprecationWarnings:
    def test_partition_cols_deprecation(self, spark):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsdf = TSDF(df, ts_col="ts", partition_cols=["symbol"])
            assert len(w) == 1
            assert "partition_cols" in str(w[0].message)
            assert issubclass(w[0].category, DeprecationWarning)

    def test_partitionCols_attribute_deprecation(self, spark):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsdf = TSDF(df, ts_col="ts", series_ids=["symbol"])
            _ = tsdf.partitionCols
            assert len(w) == 1
            assert "partitionCols" in str(w[0].message)

    def test_sql_join_opt_deprecation(self, spark):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = left_tsdf.asofJoin(right_tsdf, sql_join_opt=True)
            assert len(w) == 1
            assert "sql_join_opt" in str(w[0].message)
```

---

## 10. Documentation Updates

### Update docstrings with deprecation notices

```python
def __init__(
    self,
    df: DataFrame,
    ts_schema: Optional[TSSchema] = None,
    ts_col: Optional[str] = None,
    series_ids: Optional[Collection[str]] = None,
    partition_cols: Optional[Collection[str]] = None,
    sequence_col: Optional[str] = None,
    resample_freq: Optional[str] = None,
    resample_func: Optional[Union[Callable, str]] = None,
) -> None:
    """
    Create a TSDF (Time Series DataFrame).

    Parameters
    ----------
    df : DataFrame
        The underlying Spark DataFrame.
    ts_schema : TSSchema, optional
        Schema defining the time series structure.
    ts_col : str, optional
        Name of the timestamp column.
    series_ids : Collection[str], optional
        Column names that identify unique time series.
    partition_cols : Collection[str], optional
        .. deprecated:: 0.2.0
           Use `series_ids` instead. Will be removed in v0.3.0.
    sequence_col : str, optional
        .. deprecated:: 0.2.0
           Use `TSDF.fromSubsequenceCol()` instead. Will be removed in v0.3.0.
    resample_freq : str, optional
        Resampling frequency.
    resample_func : Callable or str, optional
        Resampling aggregation function.
    """
```

---

## 11. Deprecation Timeline

| Version | Status | Changes |
|---------|--------|---------|
| v0.2.0 | Current | Deprecated APIs work with warnings |
| v0.2.x | Transition | Default to new behavior, old APIs still work |
| v0.3.0 | Breaking | Remove deprecated APIs |

### Changelog Entry

```markdown
## [0.2.0] - YYYY-MM-DD

### Deprecated
- `partition_cols` parameter in TSDF constructor - use `series_ids` instead
- `sequence_col` parameter - use `TSDF.fromSubsequenceCol()` factory method
- `tsdf.partitionCols` attribute - use `tsdf.series_ids` instead
- `tsdf.sequence_col` attribute - no longer supported
- `sql_join_opt` parameter in `asofJoin()` - use `strategy='broadcast'`
- `Interpolation` class - use `interpolate()` function
- Importing from `tempo.resample`: `floor`, `ceiling`, `min`, `max`, `average`, `checkAllowableFreq`, `FreqDict` - import from `tempo.resample_utils`
- Importing `calculate_time_horizon` from `tempo.utils` - import from `tempo.resample`

### Removed
- `vwap()` method - use `tempo.stats.vwap()` or custom implementation
- `EMA()` method - implement using `rollingApply()`
- `withLookbackFeatures()` method - use `rollingApply()` or `rollingAgg()`
- `withRangeStats()` method - use `rollingAgg()`
- `withGroupedStats()` method - use `aggBySeries()`
```

---

## 12. Implementation Order

1. **Create `tempo/config.py`** - Feature flags infrastructure
2. **Update `tempo/tsdf.py`** - Constructor compatibility, property aliases, method stubs
3. **Update `tempo/resample.py`** - Re-export with deprecation warnings
4. **Update `tempo/utils.py`** - Re-export calculate_time_horizon
5. **Update `tempo/interpol.py`** - Keep Interpolation class wrapper
6. **Add deprecation tests** - Verify warnings are emitted
7. **Update documentation** - Deprecation notices in docstrings
8. **Update CHANGELOG.md** - Document all deprecations
