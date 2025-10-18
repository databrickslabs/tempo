# Changelog

All notable changes to Tempo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### As-Of Join Strategy Pattern
- **Strategy-based join selection**: Four join strategies with automatic selection based on data characteristics
  - `BroadcastAsOfJoiner`: For small datasets (<30MB)
  - `UnionSortFilterAsOfJoiner`: Default for general cases
  - `SkewAsOfJoiner`: AQE-based optimization for skewed data
  - `AsOfJoiner`: Abstract base class with validation
- **Automatic optimization**: `choose_as_of_join_strategy()` selects optimal strategy based on data size and skew
- **Manual strategy selection**: New `strategy` parameter in `asofJoin()` to override automatic selection
- **Feature flag**: `use_strategy_pattern` parameter for gradual rollout (defaults to `False`)

#### Testing Improvements
- Pure pytest test suite for new as-of join strategies (38 tests total)
- Strategy consistency tests verifying all strategies produce identical results
- Comprehensive edge case coverage (nulls, empty DataFrames, single series, prefixing)

#### Documentation
- Design proposals for future refactoring work (organized in `docs/proposals/`)
- Architecture documentation for tuple pattern approach

### Changed

#### API Changes (Internal)
- **Breaking (Internal API)**: `choose_as_of_join_strategy()` now requires `spark` parameter as 3rd argument
  - Rationale: Better testability, explicit dependencies, dependency injection pattern
  - Impact: Only affects direct callers (internal helper function)
  - All internal usage updated

#### Architectural Improvements
- **Circular dependency resolution**: Strategies now return `(DataFrame, TSSchema)` tuples instead of TSDF objects
  - **Tuple pattern approach**: Separates data (DataFrame) from metadata (TSSchema) for loose coupling
  - **Benefits**:
    - Clean separation of concerns - strategies are pure DataFrame transformers
    - Better testability - can test without TSDF dependency
    - Improved reusability - functions work with raw DataFrames
    - No runtime import workarounds needed
    - Performance gains from avoiding unnecessary object creation
  - **Establishes best practice**: This pattern should be adopted for future data transformations in Tempo

### Fixed

#### As-Of Join Bug Fixes
- **NULL handling in broadcast joins**: Proper handling of NULL lead values for last rows in partitions
- **LEFT JOIN semantics**: Standardized all strategies to use LEFT JOIN (preserves all left rows)
- **Prefix handling**: Fixed double-prefixing issue in `SkewAsOfJoiner` (e.g., `left_left_metric`)
- **Composite timestamps**: Proper extraction of comparable expressions for nanosecond precision
- **skipNulls behavior**: Correct implementation in `UnionSortFilterAsOfJoiner`
- **Empty DataFrame handling**: Strategies now handle empty right DataFrames correctly

### Performance

- **Expected improvements**:
  - Small datasets (<30MB): 2-3x faster with broadcast join
  - Skewed datasets: 20-50% improvement with AQE optimization
  - General cases: Equivalent performance with cleaner code

### Deprecated

- `sql_join_opt` parameter removed from internal strategy selection (replaced by automatic size detection)

---

## Migration Guide

### For End Users

**No changes required** - existing code works as-is. The `use_strategy_pattern` feature flag defaults to `False`.

**To use new strategy pattern**:
```python
# Automatic strategy selection
result = left_tsdf.asofJoin(right_tsdf, use_strategy_pattern=True)

# Manual strategy selection
result = left_tsdf.asofJoin(right_tsdf, strategy='broadcast', use_strategy_pattern=True)
```

### Rollout Plan

1. **Current**: Feature flag defaults to `False` (existing implementation)
2. **Future v0.2.x**: Change default to `True` after validation period
3. **Future v0.3**: Remove feature flag and old implementation

---

## Known Limitations

- **Nanosecond precision**: Double precision loses nanosecond-level differences (fundamental constraint)
- **Spark timestamp precision**: Limited to microsecond precision (6 decimal places)

---

## [Previous Releases]

*Previous release notes to be added here*
