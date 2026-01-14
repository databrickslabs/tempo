# Proposal: `ResampledTSDF` Intermediate Object

## Summary

Introduce a `ResampledTSDF` class that acts as a restricted intermediate object returned by `TSDF.resample()`. This enforces a safe chaining pattern where only valid follow-up operations are exposed, preventing users from accidentally invalidating resample metadata.

## Motivation

Currently, `TSDF.resample()` returns a regular `TSDF` with `resample_freq` and `resample_func` stored as instance attributes. This metadata enables chaining with `interpolate()`:

```python
tsdf.resample(freq="min", func="mean").interpolate(method="linear")
```

**The problem:** Any TSDF operation can be called on the resampled result, but many would invalidate the resample metadata:

```python
# These could produce incorrect results if interpolate() trusts the metadata
tsdf.resample(freq="min", func="mean").filter(...).interpolate(method="linear")
tsdf.resample(freq="min", func="mean").union(other).interpolate(method="linear")
```

The metadata propagates blindly through `__withTransformedDF()`, creating subtle bugs.

## Proposed Solution

### The `ResampledTSDF` Class

A restricted wrapper that only exposes operations valid after resampling:

```python
class ResampledTSDF:
    """
    Intermediate result of a resample operation.

    This class restricts the available operations to prevent invalid
    transformations that would break the resample -> interpolate chain.

    Similar to Spark's GroupedData, which only allows .agg() after .groupBy().
    """

    def __init__(self, tsdf: TSDF, freq: str, func: Union[Callable, str]) -> None:
        self._tsdf = tsdf
        self._freq = freq
        self._func = func

    @property
    def df(self) -> DataFrame:
        """Access the underlying DataFrame (read-only)."""
        return self._tsdf.df

    @property
    def ts_col(self) -> str:
        """The timestamp column name."""
        return self._tsdf.ts_col

    @property
    def series_ids(self) -> List[str]:
        """The series identifier columns."""
        return self._tsdf.series_ids

    def interpolate(
        self,
        method: str,
        target_cols: Optional[List[str]] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ) -> TSDF:
        """
        Interpolate missing values in the resampled time series.

        Uses the freq and func from the preceding resample() call.

        :param method: Interpolation method ('linear', 'zero', 'ffill', 'bfill', 'null')
        :param target_cols: Columns to interpolate (default: all numeric)
        :param show_interpolated: Add column indicating interpolated rows
        :param perform_checks: Validate time horizon (default: True)
        :return: TSDF with interpolated values
        """
        return self._tsdf.interpolate(
            method=method,
            freq=self._freq,
            func=self._func,
            target_cols=target_cols,
            show_interpolated=show_interpolated,
            perform_checks=perform_checks,
        )

    def as_tsdf(self) -> TSDF:
        """
        Finalize and return the underlying TSDF.

        Use this when you want the resampled data without interpolation.
        The returned TSDF will NOT carry resample metadata.

        :return: A regular TSDF
        """
        return self._tsdf

    def show(self, n: int = 20) -> None:
        """Display the resampled data."""
        self._tsdf.df.show(n)

    def __repr__(self) -> str:
        return f"ResampledTSDF(freq={self._freq!r}, func={self._func!r}, tsdf={self._tsdf!r})"
```

### Changes to `TSDF.resample()`

```python
def resample(
    self,
    freq: str,
    func: Union[Callable, str],
    metricCols: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    fill: Optional[bool] = None,
    perform_checks: bool = True,
) -> ResampledTSDF:  # Changed return type
    """
    Resample the time series to a regular frequency.

    :return: ResampledTSDF that can be interpolated or finalized
    """
    # ... existing aggregation logic ...

    return ResampledTSDF(
        tsdf=TSDF(enriched_df, ts_schema=copy.deepcopy(self.ts_schema)),
        freq=freq,
        func=func,
    )
```

### Changes to `TSDF.__init__()`

Remove `resample_freq` and `resample_func` parameters - they're no longer needed:

```python
def __init__(
    self,
    df: DataFrame,
    ts_schema: Optional[TSSchema] = None,
    ts_col: Optional[str] = None,
    series_ids: Optional[Collection[str]] = None,
    # REMOVED: resample_freq and resample_func
) -> None:
```

## Usage Examples

### Valid Usage

```python
# Chain resample -> interpolate (primary use case)
result = tsdf.resample(freq="min", func="mean").interpolate(method="linear")

# Get resampled data without interpolation
resampled = tsdf.resample(freq="min", func="mean").as_tsdf()

# Inspect before interpolating
resampled = tsdf.resample(freq="min", func="mean")
resampled.show()
result = resampled.interpolate(method="linear")
```

### Invalid Usage (Now Prevented)

```python
# These will raise AttributeError - operations not available on ResampledTSDF
tsdf.resample(freq="min", func="mean").filter(...)        # AttributeError
tsdf.resample(freq="min", func="mean").withColumn(...)    # AttributeError
tsdf.resample(freq="min", func="mean").union(other)       # AttributeError

# If you need those operations, finalize first:
tsdf.resample(freq="min", func="mean").as_tsdf().filter(...)  # OK
```

## Design Rationale

### Why a Separate Class?

1. **Type Safety**: IDE autocompletion only shows valid operations
2. **Fail Fast**: Invalid operations fail immediately with clear errors
3. **Self-Documenting**: The class name and methods indicate the expected workflow
4. **No Stale State**: Metadata can't become invalid because it's never stored on TSDF

### Precedent: Spark's `GroupedData`

This pattern mirrors PySpark's `GroupedData` class:

```python
# Spark's pattern
df.groupBy("key").agg(...)    # OK
df.groupBy("key").filter(...)  # AttributeError - must aggregate first

# Our pattern
tsdf.resample(...).interpolate(...)  # OK
tsdf.resample(...).filter(...)       # AttributeError - must finalize first
```

### Why `as_tsdf()` Instead of Automatic Conversion?

Explicit finalization makes the user acknowledge they're leaving the safe chain:

```python
# User must consciously "exit" the restricted context
resampled_data = tsdf.resample(freq="min", func="mean").as_tsdf()
# Now they own the responsibility for what they do with it
```

## Migration Path

### Breaking Changes

1. `TSDF.resample()` returns `ResampledTSDF` instead of `TSDF`
2. `TSDF.__init__()` no longer accepts `resample_freq`/`resample_func`
3. Code that chains other operations after `resample()` will break

### Migration Examples

```python
# Before (v0.1)
resampled = tsdf.resample(freq="min", func="mean")
filtered = resampled.filter(col("value") > 0)

# After (v0.2)
resampled = tsdf.resample(freq="min", func="mean").as_tsdf()
filtered = resampled.filter(col("value") > 0)
```

## Implementation Checklist

- [ ] Create `ResampledTSDF` class in `tempo/tsdf.py` (or new `tempo/resampled.py`)
- [ ] Update `TSDF.resample()` to return `ResampledTSDF`
- [ ] Remove `resample_freq`/`resample_func` from `TSDF.__init__()`
- [ ] Remove metadata propagation from `__withTransformedDF()`
- [ ] Update `TSDF.interpolate()` to require explicit `freq`/`func` args
- [ ] Add tests for `ResampledTSDF`
- [ ] Update documentation
- [ ] Add migration guide entry

## Open Questions

1. Should `ResampledTSDF` expose read-only properties like `columns`, `schema`?
2. Should we add a `describe()` or `summary()` method for inspection?
3. Should `as_tsdf()` be named something else? (`finalize()`, `to_tsdf()`, `unwrap()`)
4. Should we keep backward compatibility with a deprecation warning period?
