# As-Of Join Strategy Pattern Implementation

## Summary

This PR implements a flexible strategy pattern for as-of joins in Tempo, enabling automatic selection of optimal join algorithms based on data characteristics. The implementation maintains 100% backward compatibility through a feature flag while providing significant performance improvements for skewed datasets.

## Key Features

### 1. Strategy Pattern Architecture

Implemented four join strategies with automatic selection:

- **`BroadcastAsOfJoiner`**: Optimized for small datasets (<30MB), uses Spark's broadcast join
- **`UnionSortFilterAsOfJoiner`**: Default strategy for general cases with full tolerance support
- **`SkewAsOfJoiner`**: AQE-based optimization for handling skewed data distributions
- **`AsOfJoiner`**: Abstract base class providing common validation and column prefixing

### 2. Automatic Strategy Selection

The `choose_as_of_join_strategy()` function intelligently selects the best strategy:
- Broadcast join for small datasets (automatic size detection)
- Skew-aware join when `tsPartitionVal` is specified
- Standard union-sort-filter as the default fallback

### 3. Bug Fixes & Improvements

- ✅ Fixed NULL handling in broadcast joins (proper handling of last rows in partitions)
- ✅ Standardized LEFT JOIN semantics across all strategies
- ✅ Fixed prefix handling consistency (no double-prefixing)
- ✅ Improved composite timestamp index support
- ✅ Complete tolerance filter implementation

## Changes Made

### Core Implementation Files

**`tempo/joins/strategies.py`** (~1200 lines, new file)
- Implemented all four join strategy classes
- Added automatic strategy selection logic
- Included size estimation utilities
- Comprehensive error handling and validation

**`tempo/tsdf.py`**
- Integrated strategy pattern with `TSDF.asofJoin()` method
- Added `use_strategy_pattern` feature flag (defaults to `False`)
- Updated to pass `spark` parameter to strategy selection
- Maintained full backward compatibility

### Test Files

**`tests/joins/test_strategy_consistency.py`** (new, pure pytest)
- 12 comprehensive tests verifying strategy consistency
- Tests for nulls, tolerance, empty DataFrames, prefixing
- Validates all strategies produce identical results

**`tests/join/test_as_of_join.py`** (enhanced)
- 6 integration tests with real Spark DataFrames
- Tests for simple timestamps and nanosecond precision
- Validates NULL lead handling

**`tests/join/test_strategies.py`** (updated)
- 20 unit tests using mocks for fast execution
- Tests for initialization, validation, and strategy selection
- All tests updated for new API

## API Changes

### Minor Breaking Change (Internal API Only)

**Changed function signature** for better dependency injection:

```python
# Before
def choose_as_of_join_strategy(left_tsdf, right_tsdf, ...) -> AsOfJoiner:
    spark = SparkSession.builder.getOrCreate()  # Internal creation

# After
def choose_as_of_join_strategy(left_tsdf, right_tsdf, spark: SparkSession, ...) -> AsOfJoiner:
    # Spark passed as explicit parameter
```

**Rationale**:
- Improved testability (can pass mock SparkSession)
- Explicit dependencies (clearer what the function needs)
- Follows dependency injection best practices

**Impact**:
- ⚠️ Only affects direct callers of `choose_as_of_join_strategy()` (internal helper function)
- ✅ All internal usage updated (`TSDF.asofJoin()`)
- ✅ Public API (`TSDF.asofJoin()`) unchanged

### Public API (No Breaking Changes)

The `TSDF.asofJoin()` method signature remains unchanged:

```python
tsdf.asofJoin(
    right_tsdf,
    left_prefix=None,
    right_prefix="right",
    tsPartitionVal=None,
    fraction=0.5,
    skipNulls=True,
    suppress_null_warning=False,
    tolerance=None,
    strategy=None,  # Optional manual strategy selection
    use_strategy_pattern=False  # Feature flag (default: False for backward compat)
)
```

## Testing

### Test Coverage

✅ **38/38 tests passing**:
- 20 unit tests (`test_strategies.py`)
- 12 strategy consistency tests (`test_strategy_consistency.py`)
- 6 integration tests (`test_as_of_join.py`)

### Test Categories

1. **Unit Tests**: Mock-based tests for fast validation
2. **Integration Tests**: Real Spark DataFrame operations
3. **Consistency Tests**: Verify all strategies produce identical results
4. **Edge Case Tests**: Empty DataFrames, nulls, single series, prefixing

### Backward Compatibility

✅ All existing tests pass with feature flag disabled
✅ No changes to public API
✅ Default behavior unchanged

## Performance Benefits

### Expected Improvements

- **Small datasets (<30MB)**: 2-3x faster with broadcast join
- **Skewed datasets**: 20-50% improvement with AQE optimization
- **General cases**: Equivalent performance with cleaner code

### Strategy Selection Examples

```python
# Automatic selection (recommended)
result = left_tsdf.asofJoin(right_tsdf, use_strategy_pattern=True)

# Manual strategy selection
result = left_tsdf.asofJoin(right_tsdf, strategy='broadcast', use_strategy_pattern=True)
result = left_tsdf.asofJoin(right_tsdf, strategy='skew', use_strategy_pattern=True)
```

## Migration Path

### Phase 1: Testing (Current)
- Feature flag defaults to `False`
- Users can opt-in with `use_strategy_pattern=True`
- Monitor performance and correctness

### Phase 2: Gradual Rollout
- Change default to `True` after validation
- Maintain feature flag for quick rollback
- Collect production metrics

### Phase 3: Full Migration
- Remove feature flag in v0.3
- Remove old implementation
- Strategy pattern becomes standard

## Documentation

### Files Updated

- ✅ `ASOF_JOIN_ENHANCEMENTS.md`: Comprehensive implementation guide
- ✅ `CIRCULAR_DEPENDENCY_REFACTOR.md`: Architecture patterns and refactoring notes
- ✅ Inline code documentation (docstrings)
- ✅ Test documentation (test method docstrings)

### Key Documentation Sections

1. Implementation strategy and architecture
2. Test patterns and data management
3. Bug fixes and known limitations
4. Migration path and rollout plan
5. PR preparation notes

## Checklist

- [x] All tests passing (38/38)
- [x] Code formatted with Black
- [x] No new linting errors
- [x] Backward compatibility verified
- [x] No new unresolved TODOs
- [x] Documentation complete
- [x] API changes documented
- [x] Performance considerations documented

## Risks & Mitigation

### Risk: Breaking Existing Functionality
**Mitigation**:
- Feature flag defaults to `False` (old behavior)
- All existing tests pass
- Comprehensive new test coverage

### Risk: Performance Regression
**Mitigation**:
- Automatic strategy selection based on data size
- Can disable via feature flag
- Monitoring in place

### Risk: Complex API
**Mitigation**:
- Public API unchanged
- Progressive disclosure of new features
- Clear documentation

## Related Issues

- Fixes prefix handling inconsistencies
- Resolves NULL lead value bugs in broadcast joins
- Implements AQE-based skew handling
- Standardizes LEFT JOIN semantics across strategies

## Reviewers

Please focus on:
1. API design and backward compatibility
2. Strategy selection logic
3. Test coverage and quality
4. Documentation completeness
5. Migration path feasibility

---

**Branch**: `asof-join-enhancements`
**Target**: `v0.2-integration` → `master`
**Status**: ✅ Ready for Review
