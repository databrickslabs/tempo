# Tempo v0.2 Migration Guide

This guide helps users migrate existing code from Tempo v0.1.x to v0.2.

> **Backwards Compatibility**: v0.2 maintains backwards compatibility with v0.1.x APIs through deprecation warnings. Your existing code will continue to work, but you'll see warnings for deprecated features. All deprecated APIs will be removed in v1.0.0.

---

## Quick Reference

| Change | Old (v0.1) | New (v0.2) |
|--------|-----------|------------|
| Constructor param | `partition_cols=` | `series_ids=` |
| Attribute access | `tsdf.partitionCols` | `tsdf.series_ids` |
| Resample imports | `from tempo.resample import floor` | `from tempo.resample_utils import floor` |
| Interpolation | `Interpolation` class | `interpolate()` function |
| As-of join optimization | `sql_join_opt=True` | `strategy='broadcast'` |

---

## 1. TSDF Constructor Changes

### Parameter Rename: `partition_cols` → `series_ids`

```python
# v0.1 (old)
tsdf = TSDF(df, ts_col="event_ts", partition_cols=["symbol"])

# v0.2 (new)
tsdf = TSDF(df, ts_col="event_ts", series_ids=["symbol"])
```

### Deprecated: `sequence_col` Parameter

The `sequence_col` parameter is deprecated and will be removed in v1.0.0. It still works but emits a deprecation warning. Consider refactoring to use the new factory method:

```python
# v0.1 (old) - deprecated, emits warning
tsdf = TSDF(df, ts_col="event_ts", partition_cols=["symbol"], sequence_col="seq")

# v0.2 (recommended) - use fromSubsequenceCol factory method instead
tsdf = TSDF.fromSubsequenceCol(df, ts_col="event_ts", subsequence_col="seq", series_ids=["symbol"])
```

### Default Value Change: `ts_col`

The `ts_col` parameter no longer defaults to `"event_ts"`. You must explicitly provide it.

```python
# v0.1 (old) - ts_col defaulted to "event_ts"
tsdf = TSDF(df, partition_cols=["symbol"])

# v0.2 (new) - ts_col must be explicit
tsdf = TSDF(df, ts_col="event_ts", series_ids=["symbol"])
```

---

## 2. Attribute Changes

### `partitionCols` → `series_ids`

```python
# v0.1 (old)
columns = tsdf.partitionCols

# v0.2 (new)
columns = tsdf.series_ids
```

### Deprecated: `sequence_col` Attribute

```python
# v0.1 (old) - deprecated, emits warning
seq = tsdf.sequence_col

# v0.2 (recommended)
# Access via ts_schema if needed
seq = tsdf.ts_schema.subsequence_col  # If using SubsequenceTSIndex
```

---

## 3. Import Path Changes

### Resample Utilities

```python
# v0.1 (old)
from tempo.resample import floor, ceiling, min, max, average
from tempo.resample import checkAllowableFreq, FreqDict

# v0.2 (new)
from tempo.resample_utils import floor, ceiling, min, max, average
from tempo.resample_utils import checkAllowableFreq, FreqDict
```

### Time Horizon Calculation

```python
# v0.1 (old)
from tempo.utils import calculate_time_horizon
calculate_time_horizon(df, ts_col, freq, partition_cols)

# v0.2 (new)
from tempo.resample import calculate_time_horizon
calculate_time_horizon(tsdf, freq)  # Takes TSDF object directly
```

---

## 4. Interpolation API Changes

The `Interpolation` class has been replaced with a function-based API.

### Basic Usage

```python
# v0.1 (old)
from tempo.interpol import Interpolation

interp = Interpolation(is_resampled=False)
result = tsdf.interpolate(
    ts_col="event_ts",
    partition_cols=["symbol"],
    target_cols=["price"],
    freq="1 minute",
    func="mean",
    method="ffill"
)

# v0.2 (new)
from tempo.interpol import interpolate, forward_fill

result = interpolate(
    tsdf,
    target_cols=["price"],
    method=forward_fill,
    leading_margin=0,
    lagging_margin=0
)
```

### Method Mapping

| v0.1 String | v0.2 Function |
|-------------|---------------|
| `"ffill"` | `forward_fill` |
| `"bfill"` | `backward_fill` |
| `"zero"` | `zero_fill` |
| `"linear"` | `"linear"` (unchanged) |
| `"null"` | `"null"` (unchanged) |

---

## 5. As-Of Join Changes

### Strategy Parameter Replaces `sql_join_opt`

```python
# v0.1 (old)
result = left_tsdf.asofJoin(right_tsdf, sql_join_opt=True)

# v0.2 (new)
result = left_tsdf.asofJoin(right_tsdf, strategy='broadcast')
```

### Available Strategies

| Strategy | Use Case |
|----------|----------|
| `'broadcast'` | Small right datasets (<30MB) |
| `'union'` | General cases (default) |
| `'skew'` | Skewed data with AQE optimization |
| `None` | Automatic selection (recommended) |

---

## 6. Deprecated Methods

The following methods are deprecated in v0.2 and will be removed in v1.0.0. They still work but emit deprecation warnings:

| Method | Status | Recommended Alternative |
|--------|--------|------------------------|
| `vwap()` | Deprecated | Use `tempo.stats.vwap()` |
| `EMA()` | Deprecated | Use `tsdf.withColumn()` with custom EMA calculation |
| `withLookbackFeatures()` | Deprecated | Use `rollingApply()` or `rollingAgg()` |
| `withRangeStats()` | Deprecated | Use `rollingAgg()` |
| `withGroupedStats()` | Deprecated | Use `aggBySeries()` |

These methods now act as wrappers that call the new APIs internally. You can continue using them during the migration period, but you'll see deprecation warnings encouraging you to update your code before v1.0.0.

---

## 7. New Methods Available

v0.2 adds several new convenience methods:

```python
# Repartitioning
tsdf.repartitionBySeries(numPartitions=10)
tsdf.repartitionByTime(numPartitions=10)

# Natural ordering
tsdf.withNaturalOrdering()

# Time filtering
tsdf.earliest(n=5)  # First n records per series
tsdf.latest(n=5)    # Last n records per series

# DataFrame operations (now TSDF-aware)
tsdf.where(condition)
tsdf.select(*cols)
tsdf.withColumn(name, expr)
tsdf.drop(*cols)

# Aggregations
tsdf.describe(*cols)
tsdf.union(other_tsdf)
tsdf.unionByName(other_tsdf, allowMissingColumns=True)
```

---

## 8. Type Changes

### `checkAllowableFreq()` Return Type

```python
# v0.1 (old) - returned (str, str)
result = checkAllowableFreq("1 MICROSECOND")
# result = ("1", "microsec")

# v0.2 (new) - returns (int, str)
result = checkAllowableFreq("1 MICROSECOND")
# result = (1, "microsec")
```

### Exception Changes

```python
# v0.1 (old) - raised TypeError for None freq
try:
    checkAllowableFreq(None)
except TypeError:
    pass

# v0.2 (new) - raises ValueError for None freq
try:
    checkAllowableFreq(None)
except ValueError:
    pass
```

---

## 9. Complete Migration Example

### Before (v0.1)

```python
from tempo.tsdf import TSDF
from tempo.resample import floor, checkAllowableFreq
from tempo.interpol import Interpolation
from tempo.utils import calculate_time_horizon

# Create TSDF
tsdf = TSDF(df, ts_col="event_ts", partition_cols=["symbol"])

# Access partition columns
print(tsdf.partitionCols)

# Interpolate
interp = Interpolation(is_resampled=False)
result = tsdf.interpolate(
    ts_col="event_ts",
    partition_cols=["symbol"],
    target_cols=["price"],
    method="ffill"
)

# As-of join with optimization
joined = left_tsdf.asofJoin(right_tsdf, sql_join_opt=True)

# Calculate time horizon
calculate_time_horizon(df, "event_ts", "1 minute", ["symbol"])
```

### After (v0.2)

```python
from tempo.tsdf import TSDF
from tempo.resample_utils import floor, checkAllowableFreq
from tempo.interpol import interpolate, forward_fill
from tempo.resample import calculate_time_horizon

# Create TSDF
tsdf = TSDF(df, ts_col="event_ts", series_ids=["symbol"])

# Access series IDs
print(tsdf.series_ids)

# Interpolate
result = interpolate(
    tsdf,
    target_cols=["price"],
    method=forward_fill,
    leading_margin=0,
    lagging_margin=0
)

# As-of join with strategy
joined = left_tsdf.asofJoin(right_tsdf, strategy='broadcast')

# Calculate time horizon
calculate_time_horizon(tsdf, "1 minute")
```

---

## 10. Deprecation Timeline

| Version | Status | What to Expect |
|---------|--------|----------------|
| **v0.2.0** | Current | Deprecated APIs work with warnings. Old code continues to function. |
| **v0.2.x** | Transition | New behavior is default. Old APIs still work with warnings. |
| **v1.0.0** | Breaking | Deprecated APIs removed. Migration required. |

All deprecated parameters, attributes, and methods will emit `DeprecationWarning` when used. To see these warnings, ensure your Python warnings filter is configured appropriately:

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning)
```

---

## 11. Getting Help

If you encounter issues during migration:
1. Check the [CHANGELOG.md](CHANGELOG.md) for detailed release notes
2. Review the [API documentation](docs/api.md)
3. Open an issue at https://github.com/databrickslabs/tempo/issues
