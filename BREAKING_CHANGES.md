# Complete Breaking Changes: v0.2-integration vs master

This document catalogs **every breaking change** between the `v0.2-integration` branch and `master`.

---

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| TSDF Constructor Changes | 6 parameters | CRITICAL |
| TSDF Method Signature Changes | 7+ methods | CRITICAL |
| Removed Methods/Classes | 8+ | HIGH |
| Module Restructuring | 3 modules | HIGH |
| Import Path Changes | 20+ | HIGH |
| New Required Dependencies | 3 modules | MEDIUM |

---

## 1. TSDF Class Breaking Changes

### 1.1 Constructor Signature (CRITICAL)

**MASTER:**
```python
def __init__(
    self,
    df: DataFrame,
    ts_col: str = "event_ts",
    partition_cols: Optional[list[str]] = None,
    sequence_col: Optional[str] = None,
):
```

**v0.2-integration:**
```python
def __init__(
    self,
    df: DataFrame,
    ts_schema: Optional[TSSchema] = None,
    ts_col: Optional[str] = None,
    series_ids: Optional[Collection[str]] = None,
    resample_freq: Optional[str] = None,
    resample_func: Optional[Union[Callable, str]] = None,
) -> None:
```

| Parameter | Change |
|-----------|--------|
| `partition_cols` | **REMOVED** → use `series_ids` |
| `sequence_col` | **REMOVED** entirely |
| `ts_col` | Changed from `str = "event_ts"` to `Optional[str] = None` |
| `ts_schema` | **NEW** - alternative to ts_col |
| `series_ids` | **NEW** - replaces partition_cols |
| `resample_freq` | **NEW** |
| `resample_func` | **NEW** |

### 1.2 Class Inheritance Change

```python
# MASTER
class TSDF:

# v0.2-integration
class TSDF(WindowBuilder):
```

TSDF now inherits from `WindowBuilder` class.

### 1.3 Removed Instance Attributes

| Attribute | Status |
|-----------|--------|
| `self.partitionCols` | **REMOVED** → use `self.series_ids` property |
| `self.sequence_col` | **REMOVED** entirely |
| `self.ts_col` | Changed to property delegating to `self.ts_schema.ts_idx.colname` |

### 1.4 Method Signature Changes

#### `asofJoin()` - MAJOR CHANGE
```python
# MASTER
def asofJoin(
    self,
    right_tsdf: "TSDF",
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,           # REMOVED
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
) -> "TSDF"

# v0.2-integration
def asofJoin(
    self,
    right_tsdf: TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    strategy: Optional[str] = None,       # NEW
) -> TSDF
```

- `sql_join_opt` parameter **REMOVED**
- `strategy` parameter **ADDED** (for manual strategy selection)

#### `select()` - TYPE CHANGE
```python
# MASTER
def select(self, *cols: Union[str, List[str]]) -> "TSDF":
    # Enforces mandatory columns must be present

# v0.2-integration
def select(self, *cols: Union[str, Column]) -> TSDF:
    # No mandatory column enforcement
```

- Accepted type changed: `List[str]` → `Column`
- Removed validation that ts_col/partitionCols must be present

#### Time Filtering Methods - TYPE CHANGE
```python
# MASTER
def at(self, ts: Union[str, int]) -> "TSDF"
def before(self, ts: Union[str, int]) -> "TSDF"
def after(self, ts: Union[str, int]) -> "TSDF"
def atOrBefore(self, ts: Union[str, int]) -> "TSDF"
def atOrAfter(self, ts: Union[str, int]) -> "TSDF"
def between(self, start_ts: Union[str, int], end_ts: Union[str, int], inclusive: bool = True) -> "TSDF"

# v0.2-integration
def at(self, ts: Any) -> TSDF
def before(self, ts: Any) -> TSDF
# etc. - all changed to Any
```

### 1.5 Removed Methods

| Method | Type | Description |
|--------|------|-------------|
| `parse_nanos_timestamp()` | Static | Replaced with `fromSubsequenceCol()` and `fromStringTimestamp()` |
| `__slice()` | Private | Functionality reimplemented |
| `__add_double_ts()` | Private | Removed |
| `__validate_ts_string()` | Private | Moved to TSSchema |
| `__validated_column()` | Static | Moved to TSSchema |
| `__validated_columns()` | Instance | Moved to TSSchema |

### 1.6 New Methods Added

- `__repr__()`, `__eq__()`
- `buildEmptyLattice()` - class method
- `fromSubsequenceCol()` - class method
- `fromStringTimestamp()` - class method
- `where()` - public method
- `withNaturalOrdering()`, `withColumn()`, `withColumnRenamed()`, `withColumnTypeChanged()`
- `drop()`, `mapInPandas()`, `union()`, `unionByName()`
- `rollingAgg()`, `rollingApply()`, `summarize()`, `agg()`
- `describe()`, `metricSummary()`
- `groupBySeries()`, `aggBySeries()`, `applyToSeries()`
- `groupByCycles()`, `aggByCycles()`, `applyToCycles()`
- `extractStateIntervals()`
- `baseWindow()`, `rowsBetweenWindow()`, `rangeBetweenWindow()`
- `repartitionBySeries()`, `repartitionByTime()`
- Properties: `ts_index`, `columns`, `series_ids`, `structural_cols`, `observational_cols`, `metric_cols`

---

## 2. Module Restructuring

### 2.1 Deleted Module: `python/tempo/as_of_join.py`

**Entire module deleted** - functionality moved to `python/tempo/joins/strategies.py`

### 2.2 New Module: `python/tempo/joins/`

New package with:
- `joins/__init__.py`
- `joins/strategies.py` (1232 lines)

**Exports from `tempo.joins`:**
```python
from tempo.joins import (
    AsOfJoiner,
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
)
```

### 2.3 New Schema Module: `python/tempo/tsschema.py`

New module containing:
- `TSSchema` - Timestamp schema class
- `TSIndex` - Timestamp index class
- `WindowBuilder` - Base class for TSDF
- `ParsedTSIndex`, `SimpleTSIndex`, `SubsequenceTSIndex`
- `DEFAULT_TIMESTAMP_FORMAT`
- Helper functions: `identify_fractional_second_separator()`, `is_time_format()`, `sub_seconds_precision_digits()`

### 2.4 New Typing Module: `python/tempo/typing.py`

New type definitions:
```python
from tempo.typing import (
    ColumnOrName,
    PandasMapIterFunction,
    PandasGroupedMapFunction,
)
```

### 2.5 New Time Unit Module: `python/tempo/timeunit.py`

```python
from tempo.timeunit import TimeUnit, StandardTimeUnits, TimeUnitsType
```

---

## 3. Utils Module Changes

### 3.1 Removed Function: `calculate_time_horizon()`

**MASTER location:** `tempo.utils`
```python
def calculate_time_horizon(
    df: DataFrame,
    ts_col: str,
    freq: str,
    partition_cols: Optional[List[str]],
    local_freq_dict: Optional[t_resample.FreqDict] = None,
) -> None:
```

**v0.2-integration location:** `tempo.resample`
```python
def calculate_time_horizon(
    tsdf: t_tsdf.TSDF,
    freq: str,
    local_freq_dict: Optional[FreqDict] = None,
) -> None:
```

- Now takes TSDF object instead of separate parameters
- Import path changed

### 3.2 Changed Function: `get_display_df()`

```python
# MASTER
def get_display_df(tsdf: t_tsdf.TSDF, k: int) -> DataFrame:
    orderCols = tsdf.partitionCols.copy()
    orderCols.append(tsdf.ts_col)
    if tsdf.sequence_col:
        orderCols.append(tsdf.sequence_col)
    return tsdf.latest(k).df.orderBy(orderCols)

# v0.2-integration
def get_display_df(tsdf: t_tsdf.TSDF, k: int) -> DataFrame:
    return tsdf.latest(k).withNaturalOrdering().df
```

### 3.3 New Function: `time_range()`

```python
def time_range(
    spark: SparkSession,
    start_time: dt,
    end_time: Optional[dt] = None,
    step_size: Optional[td] = None,
    num_intervals: Optional[int] = None,
    ts_colname: str = "ts",
    include_interval_ends: bool = False,
) -> DataFrame:
```

---

## 4. Resample Module Changes

### 4.1 Utilities Moved to `tempo.resample_utils`

**Imports that will break:**
```python
# MASTER
from tempo.resample import floor, ceiling, min, max, average
from tempo.resample import ALLOWED_FREQ_KEYS, checkAllowableFreq, FreqDict

# v0.2-integration
from tempo.resample_utils import floor, ceiling, min, max, average
from tempo.resample_utils import ALLOWED_FREQ_KEYS, checkAllowableFreq, FreqDict
```

**Moved items:**
- `ALLOWED_FREQ_KEYS`
- `FreqDict` class
- `average`, `ceiling`, `floor`, `min`, `max` functions
- `checkAllowableFreq()`
- `is_valid_allowed_freq_keys()`
- `validateFuncExists()`

---

## 5. Interpolation Module Changes

### 5.1 Removed Class: `Interpolation`

**MASTER:**
```python
from tempo.interpol import Interpolation
interp = Interpolation(is_resampled=True)
```

**v0.2-integration:** Class removed, replaced with functional API

```python
from tempo.interpol import interpolate
# or use tsdf.interpolate() method
```

---

## 6. Import Path Migration Guide

### TSDF Creation
```python
# MASTER
tsdf = TSDF(df, ts_col="event_ts", partition_cols=["symbol"])

# v0.2-integration
tsdf = TSDF(df, ts_col="event_ts", series_ids=["symbol"])
```

### Accessing Partition Columns
```python
# MASTER
cols = tsdf.partitionCols

# v0.2-integration
cols = tsdf.series_ids
```

### Resample Utilities
```python
# MASTER
from tempo.resample import floor, ceiling, checkAllowableFreq

# v0.2-integration
from tempo.resample_utils import floor, ceiling, checkAllowableFreq
```

### Calculate Time Horizon
```python
# MASTER
from tempo.utils import calculate_time_horizon
calculate_time_horizon(df, ts_col, freq, partition_cols)

# v0.2-integration
from tempo.resample import calculate_time_horizon
calculate_time_horizon(tsdf, freq)
```

### Interpolation
```python
# MASTER
from tempo.interpol import Interpolation
interp = Interpolation(is_resampled=True)

# v0.2-integration - use TSDF method or functional API
tsdf.interpolate(...)
```

### As-Of Join (with new strategy)
```python
# MASTER
result = left.asofJoin(right, sql_join_opt=True)

# v0.2-integration
result = left.asofJoin(right, strategy='broadcast')
```

---

## 7. Files Changed Summary

| File | Status |
|------|--------|
| `python/tempo/as_of_join.py` | **DELETED** |
| `python/tempo/joins/__init__.py` | **NEW** |
| `python/tempo/joins/strategies.py` | **NEW** (1232 lines) |
| `python/tempo/tsschema.py` | **NEW** |
| `python/tempo/typing.py` | **NEW** |
| `python/tempo/timeunit.py` | **NEW** |
| `python/tempo/resample_utils.py` | **NEW** |
| `python/tempo/tsdf.py` | **MAJOR CHANGES** |
| `python/tempo/utils.py` | **MODIFIED** |
| `python/tempo/interpol.py` | **REFACTORED** |
| `python/tempo/resample.py` | **REFACTORED** |
| `python/tests/as_of_join_tests.py` | **DELETED** → moved to `tests/joins/` |

---

## 8. Behavior Changes

1. **sequence_col support removed** - No longer supported in constructor or methods
2. **Validation moved** - Column validation now in TSSchema class, not TSDF
3. **select() no longer enforces columns** - No validation that structural columns are included
4. **LEFT JOIN standardized** - All as-of join strategies now use LEFT JOIN semantics
5. **NULL handling improved** - Better handling of NULL values in joins

---

## 9. New Dependencies

Code in v0.2-integration requires these new internal imports:
- `from tempo.tsschema import TSSchema, TSIndex, WindowBuilder`
- `from tempo.typing import ColumnOrName`
- `from tempo.timeunit import TimeUnit`

---

## 10. REMOVED Feature Engineering Methods (CRITICAL)

These methods were **completely removed** from TSDF:

| Method | Description | Status |
|--------|-------------|--------|
| `vwap()` | Volume-weighted average price | **REMOVED** |
| `EMA()` | Exponential moving average | **REMOVED** |
| `withLookbackFeatures()` | Add lookback window features | **REMOVED** |
| `withRangeStats()` | Add range-based statistics | **REMOVED** |
| `withGroupedStats()` | Add grouped statistics | **REMOVED** |
| `__baseWindow()` | Internal window helper | **REMOVED** |
| `__rangeBetweenWindow()` | Internal range window | **REMOVED** |
| `__rowsBetweenWindow()` | Internal rows window | **REMOVED** |

---

## 11. Interpolation API Complete Redesign (CRITICAL)

### Class-Based → Function-Based

**MASTER (removed):**
```python
from tempo.interpol import Interpolation

interp = Interpolation(is_resampled=False)
result = interp.interpolate(
    tsdf,
    partition_cols,
    target_cols,
    freq,
    ts_col,
    func,
    method="ffill",
    show_interpolated=True
)
```

**v0.2-integration:**
```python
from tempo.interpol import interpolate, forward_fill, backward_fill, zero_fill

result = interpolate(
    tsdf,
    target_cols,
    method=forward_fill,  # function object, not string
    leading_margin=0,
    lagging_margin=0
)
```

### Parameter Changes

| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `partition_cols` | removed | Extracted from TSDF |
| `freq` | removed | Not needed |
| `ts_col` | removed | Extracted from TSDF |
| `func` | removed | Not needed |
| `show_interpolated` | removed | Not supported |
| `method="ffill"` | `method=forward_fill` | Function object |
| `method="bfill"` | `method=backward_fill` | Function object |
| `method="zero"` | `method=zero_fill` | Function object |
| - | `leading_margin` | **NEW** |
| - | `lagging_margin` | **NEW** |

### Removed Validation Methods
- `_Interpolation__validate_fill()` - removed
- `_Interpolation__validate_col()` - removed
- `_Interpolation__validate_ts_col_data_type_is_not_timestamp()` - removed

---

## 12. Resample API Changes

### Return Type Change
```python
# MASTER
checkAllowableFreq("1 MICROSECOND")  # Returns ("1", "microsec") - (str, str)

# v0.2-integration
checkAllowableFreq("1 MICROSECOND")  # Returns (1, "microsec") - (int, str)
```

### Exception Type Change
```python
# MASTER - TypeError when freq is None
# v0.2-integration - ValueError when freq is None
```

### New Function
```python
from tempo.resample import resample

result = resample(tsdf, freq="min", func="floor", prefix=None, fill=False)
```

---

## 13. Test Infrastructure Changes

### Test Data Builder Pattern
```python
# MASTER
self.get_test_df_builder("init").as_tsdf()

# v0.2-integration
self.get_test_function_df_builder("input_data").as_tsdf()
# or
self.get_test_function_df_builder(self.data_type, "init").as_tsdf()
```

### Test Base Import
```python
# MASTER
from tests.tsdf_tests import SparkTest

# v0.2-integration
from tests.base import SparkTest
```

### Parameterized Tests
```python
# v0.2-integration uses parameterized_class decorator
from parameterized import parameterized_class

@parameterized_class(("data_type", "interpol_cols"), [...])
class InterpolationTests(SparkTest):
    pass
```

---

## 14. Removed Test Files

| File | Status | Replacement |
|------|--------|-------------|
| `python/tests/as_of_join_tests.py` | **DELETED** | `python/tests/joins/*.py` |
| `python/tests/unit_test_data/as_of_join_tests.json` | **DELETED** | `python/tests/unit_test_data/joins/*.json` |

---

## 15. New TSDF Methods Added

| Method | Description |
|--------|-------------|
| `repartitionBySeries(numPartitions)` | Repartition by series identifiers |
| `repartitionByTime(numPartitions)` | Repartition by time |
| `withNaturalOrdering()` | Apply natural ordering |
| `earliest(n)` | Get first n records per series |
| `latest(n)` | Get last n records per series |
| `describe(*cols)` | Get statistics |
| `union(other)` | Positional union |
| `unionByName(other, allowMissingColumns)` | Union by column name |
| `where(condition)` | Filter rows |

---

## 16. TSSchema Required Classes

New classes that must be understood for advanced usage:

```python
from tempo.tsschema import (
    TSSchema,
    TSIndex,
    SimpleTSIndex,
    ParsedTSIndex,
    SubsequenceTSIndex,
    OrdinalTSIndex,
    ParsedDateIndex,
    ParsedTimestampIndex,
    SimpleDateIndex,
    SimpleTimestampIndex,
    WindowBuilder,
)
```

---

## Complete Migration Checklist

### Constructor Migration
- [ ] Change `partition_cols=` → `series_ids=`
- [ ] Remove `sequence_col=` parameter
- [ ] Handle `ts_col` default change (no longer defaults to "event_ts")

### Attribute Migration
- [ ] Change `tsdf.partitionCols` → `tsdf.series_ids`
- [ ] Remove references to `tsdf.sequence_col`

### Import Migration
- [ ] `from tempo.resample import floor, ceiling` → `from tempo.resample_utils import floor, ceiling`
- [ ] `from tempo.utils import calculate_time_horizon` → `from tempo.resample import calculate_time_horizon`
- [ ] `from tempo.interpol import Interpolation` → `from tempo.interpol import interpolate, forward_fill, backward_fill, zero_fill`

### Method Migration
- [ ] `asofJoin(..., sql_join_opt=True)` → `asofJoin(..., strategy='broadcast')`
- [ ] Remove calls to `vwap()`, `EMA()`, `withLookbackFeatures()`, `withRangeStats()`, `withGroupedStats()`
- [ ] Update interpolation code to use new function-based API

### Type Changes
- [ ] Update code expecting `checkAllowableFreq()` to return `(str, str)` → now returns `(int, str)`
- [ ] Handle `ValueError` instead of `TypeError` for invalid freq
