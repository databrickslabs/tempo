# Refactoring Proposal: Extract Shared DataFrame Utilities

**Date**: October 17, 2025
**Status**: PROPOSAL
**Author**: Analysis of existing code patterns

---

## Executive Summary

This proposal outlines a refactoring to extract commonly duplicated logic from the codebase into reusable utility modules. The primary goal is to eliminate the need for local TSDF imports in strategy methods by providing pure DataFrame transformation functions that can be used across the codebase.

**Key Benefits**:
1. Eliminate local TSDF imports in join strategies
2. Enable pure functional programming patterns
3. Improve testability (can test without TSDF objects)
4. Reduce code duplication across modules
5. Make utilities reusable in non-Tempo contexts

---

## Problem Analysis

### Current Duplication Patterns

The analysis identified four major categories of duplicated logic:

#### 1. **Partition Column Management**
- **Location**: Used in `AsOfJoiner`, `TSDF.__checkPartitionCols`, and multiple join methods
- **Purpose**: Validate, compare, and manage series/partition columns
- **Current Issues**: Logic scattered across multiple classes

#### 2. **Schema Validation**
- **Location**: Used in `AsOfJoiner._checkAreJoinable`, `TSDF.__validateTsColMatch`
- **Purpose**: Validate schema compatibility between TSDFs
- **Current Issues**: Validation logic duplicated and inconsistent

#### 3. **Join Operations**
- **Location**: Methods like `_combine`, `_appendNullColumns`, `_filterLastRightRow`
- **Purpose**: Core DataFrame transformation logic for joins
- **Current Issues**: Requires local TSDF imports, not reusable

#### 4. **Prefix Handling**
- **Location**: `_prefixColumns`, `_prefixOverlappingColumns`, `__addPrefixToColumns`
- **Purpose**: Add prefixes to columns to avoid naming conflicts
- **Current Issues**: Duplicated between `as_of_join.py` and `strategies.py`, operates on TSDFs

---

## Proposed Solution

### Create New Utility Module: `tempo/utils/dataframe_ops.py`

This module will contain pure DataFrame operations that don't depend on TSDF objects.

```python
"""
Pure DataFrame transformation utilities for Tempo.

This module provides DataFrame operations that are used across multiple
Tempo components (joins, aggregations, etc.) without requiring TSDF objects.
These functions follow the functional programming pattern of returning
(DataFrame, metadata) tuples rather than wrapped objects.
"""

from typing import Dict, List, Set, Tuple, Optional
from functools import reduce
import pyspark.sql.functions as sfn
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StructType


# ========================================
# 1. PREFIX HANDLING UTILITIES
# ========================================

def get_overlapping_columns(
    left_cols: List[str],
    right_cols: List[str],
    exclude_cols: Optional[Set[str]] = None
) -> Set[str]:
    """
    Find columns that appear in both DataFrames.

    :param left_cols: Column names from left DataFrame
    :param right_cols: Column names from right DataFrame
    :param exclude_cols: Columns to exclude from overlap detection (e.g., partition keys)
    :return: Set of overlapping column names
    """
    overlapping = set(left_cols).intersection(set(right_cols))
    if exclude_cols:
        overlapping = overlapping - exclude_cols
    return overlapping


def prefix_columns(
    df: DataFrame,
    columns_to_prefix: Set[str],
    prefix: str
) -> DataFrame:
    """
    Add a prefix to specified columns in a DataFrame.

    :param df: Input DataFrame
    :param columns_to_prefix: Set of column names to prefix
    :param prefix: Prefix string to add (e.g., "left", "right")
    :return: DataFrame with prefixed columns

    Example:
        >>> df = spark.createDataFrame([(1, "a")], ["id", "value"])
        >>> prefix_columns(df, {"value"}, "right")
        # Returns: DataFrame with columns ["id", "right_value"]
    """
    if not prefix or not columns_to_prefix:
        return df

    # Build column rename expressions
    select_exprs = [
        sfn.col(col).alias(f"{prefix}_{col}") if col in columns_to_prefix else sfn.col(col)
        for col in df.columns
    ]

    return df.select(*select_exprs)


def prefix_overlapping_columns(
    left_df: DataFrame,
    right_df: DataFrame,
    left_prefix: str,
    right_prefix: str,
    exclude_cols: Optional[Set[str]] = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Prefix overlapping columns in two DataFrames to avoid naming conflicts.

    :param left_df: Left DataFrame
    :param right_df: Right DataFrame
    :param left_prefix: Prefix for left DataFrame columns
    :param right_prefix: Prefix for right DataFrame columns
    :param exclude_cols: Columns to exclude from prefixing (e.g., join keys)
    :return: Tuple of (left_prefixed_df, right_prefixed_df)

    Example:
        >>> left = spark.createDataFrame([(1, "a", 10)], ["key", "value", "metric"])
        >>> right = spark.createDataFrame([(1, "b", 20)], ["key", "value", "metric"])
        >>> l, r = prefix_overlapping_columns(left, right, "l", "r", {"key"})
        # l has columns: ["key", "l_value", "l_metric"]
        # r has columns: ["key", "r_value", "r_metric"]
    """
    overlapping = get_overlapping_columns(
        left_df.columns,
        right_df.columns,
        exclude_cols
    )

    left_prefixed = prefix_columns(left_df, overlapping, left_prefix)
    right_prefixed = prefix_columns(right_df, overlapping, right_prefix)

    return left_prefixed, right_prefixed


# ========================================
# 2. SCHEMA VALIDATION UTILITIES
# ========================================

def validate_column_types_match(
    left_schema: StructType,
    right_schema: StructType,
    column_name: str
) -> bool:
    """
    Validate that a column has the same type in both schemas.

    :param left_schema: Schema from left DataFrame
    :param right_schema: Schema from right DataFrame
    :param column_name: Name of column to validate
    :return: True if types match
    :raises ValueError: If column doesn't exist or types don't match
    """
    if column_name not in left_schema.fieldNames():
        raise ValueError(f"Column '{column_name}' not found in left schema")
    if column_name not in right_schema.fieldNames():
        raise ValueError(f"Column '{column_name}' not found in right schema")

    left_type = left_schema[column_name].dataType
    right_type = right_schema[column_name].dataType

    if left_type != right_type:
        raise ValueError(
            f"Column '{column_name}' has mismatched types: "
            f"left={left_type}, right={right_type}"
        )

    return True


def validate_partition_columns_match(
    left_cols: List[str],
    right_cols: List[str]
) -> bool:
    """
    Validate that partition columns match in both order and names.

    :param left_cols: Partition columns from left
    :param right_cols: Partition columns from right
    :return: True if they match
    :raises ValueError: If partition columns don't match
    """
    if len(left_cols) != len(right_cols):
        raise ValueError(
            f"Partition column count mismatch: "
            f"left has {len(left_cols)}, right has {len(right_cols)}"
        )

    for left_col, right_col in zip(left_cols, right_cols):
        if left_col != right_col:
            raise ValueError(
                f"Partition columns must have same names in same order. "
                f"Found '{left_col}' vs '{right_col}'"
            )

    return True


# ========================================
# 3. DATAFRAME PREPARATION UTILITIES
# ========================================

def add_null_columns(
    df: DataFrame,
    columns_to_add: Set[str],
    column_types: Optional[Dict[str, str]] = None
) -> DataFrame:
    """
    Add null columns to a DataFrame (for union preparation).

    :param df: Input DataFrame
    :param columns_to_add: Set of column names to add
    :param column_types: Optional dict mapping column names to data types
    :return: DataFrame with additional null columns

    Example:
        >>> df = spark.createDataFrame([(1, "a")], ["id", "value"])
        >>> add_null_columns(df, {"metric", "score"}, {"metric": "double", "score": "int"})
        # Returns: DataFrame with columns ["id", "value", "metric", "score"]
        #          where metric and score are all null
    """
    if not columns_to_add:
        return df

    select_exprs = [sfn.col(col) for col in df.columns]

    for col_name in columns_to_add:
        if column_types and col_name in column_types:
            col_expr = sfn.lit(None).cast(column_types[col_name]).alias(col_name)
        else:
            col_expr = sfn.lit(None).alias(col_name)
        select_exprs.append(col_expr)

    return df.select(*select_exprs)


def align_dataframes_for_union(
    left_df: DataFrame,
    right_df: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """
    Align two DataFrames to have the same columns for union.

    Adds missing columns as nulls to each DataFrame so they can be unioned.

    :param left_df: Left DataFrame
    :param right_df: Right DataFrame
    :return: Tuple of (aligned_left, aligned_right) with same columns

    Example:
        >>> left = spark.createDataFrame([(1, "a")], ["id", "value"])
        >>> right = spark.createDataFrame([(2, 10)], ["id", "metric"])
        >>> l, r = align_dataframes_for_union(left, right)
        # l has columns: ["id", "value", "metric"]  (metric is null)
        # r has columns: ["id", "value", "metric"]  (value is null)
    """
    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)

    # Find columns missing from each side
    right_only = right_cols - left_cols
    left_only = left_cols - right_cols

    # Get types for null columns
    left_types = {field.name: str(field.dataType) for field in left_df.schema.fields}
    right_types = {field.name: str(field.dataType) for field in right_df.schema.fields}

    # Add missing columns
    aligned_left = add_null_columns(left_df, right_only, right_types)
    aligned_right = add_null_columns(right_df, left_only, left_types)

    return aligned_left, aligned_right


# ========================================
# 4. COLUMN FILTERING UTILITIES
# ========================================

def get_non_structural_columns(
    df_columns: List[str],
    ts_col: str,
    partition_cols: List[str]
) -> List[str]:
    """
    Get columns that are not structural (timestamp or partition columns).

    :param df_columns: All columns in the DataFrame
    :param ts_col: Timestamp column name
    :param partition_cols: Partition column names
    :return: List of non-structural columns (observation/metric columns)
    """
    structural = set([ts_col] + partition_cols)
    return [col for col in df_columns if col not in structural]


def get_columns_by_pattern(
    df_columns: List[str],
    pattern: str
) -> List[str]:
    """
    Get columns matching a naming pattern.

    :param df_columns: All columns in the DataFrame
    :param pattern: Pattern to match (supports wildcards)
    :return: List of matching column names

    Example:
        >>> cols = ["id", "left_value", "left_metric", "right_value"]
        >>> get_columns_by_pattern(cols, "left_*")
        # Returns: ["left_value", "left_metric"]
    """
    import re
    # Convert wildcard pattern to regex
    regex_pattern = pattern.replace("*", ".*")
    regex = re.compile(f"^{regex_pattern}$")
    return [col for col in df_columns if regex.match(col)]


# ========================================
# 5. WINDOW OPERATION UTILITIES
# ========================================

def create_partitioned_window(
    partition_cols: List[str],
    order_col: str,
    ascending: bool = True
) -> "WindowSpec":
    """
    Create a window specification for partitioned operations.

    :param partition_cols: Columns to partition by
    :param order_col: Column to order by within partitions
    :param ascending: Whether to sort ascending or descending
    :return: WindowSpec object
    """
    from pyspark.sql.window import Window

    window = Window.partitionBy(*partition_cols)

    if ascending:
        window = window.orderBy(sfn.col(order_col).asc())
    else:
        window = window.orderBy(sfn.col(order_col).desc())

    return window


# ========================================
# 6. METADATA TRACKING UTILITIES
# ========================================

def track_column_metadata(
    original_df: DataFrame,
    transformed_df: DataFrame
) -> Dict[str, any]:
    """
    Track metadata about a DataFrame transformation.

    :param original_df: DataFrame before transformation
    :param transformed_df: DataFrame after transformation
    :return: Dict containing transformation metadata
    """
    return {
        "original_columns": original_df.columns,
        "transformed_columns": transformed_df.columns,
        "added_columns": list(set(transformed_df.columns) - set(original_df.columns)),
        "removed_columns": list(set(original_df.columns) - set(transformed_df.columns)),
        "original_schema": original_df.schema,
        "transformed_schema": transformed_df.schema,
    }
```

---

## Refactored Strategy Implementation

### Before (Using Local TSDF Import)

```python
class SkewAsOfJoiner(AsOfJoiner):
    def _skewSeparatedJoin(self, left, right, skewed_keys) -> Tuple[DataFrame, TSSchema]:
        # Import TSDF here to avoid circular dependency
        from tempo.tsdf import TSDF

        # Build filter conditions for skewed keys
        left_skewed = TSDF(left.df.filter(skewed_filter), ...)
        left_normal = TSDF(left.df.filter(~skewed_filter), ...)

        # ... more TSDF operations ...
```

### After (Using Pure DataFrame Utilities)

```python
from tempo.utils.dataframe_ops import (
    prefix_overlapping_columns,
    align_dataframes_for_union,
    add_null_columns
)

class SkewAsOfJoiner(AsOfJoiner):
    def _skewSeparatedJoin(self, left, right, skewed_keys) -> Tuple[DataFrame, TSSchema]:
        # No TSDF import needed - work directly with DataFrames

        # Split data using DataFrame operations
        left_df_skewed = left.df.filter(skewed_filter)
        left_df_normal = left.df.filter(~skewed_filter)
        right_df_skewed = right.df.filter(skewed_filter)
        right_df_normal = right.df.filter(~skewed_filter)

        # Process using pure DataFrame functions
        normal_result = self._standardAsOfJoinDataFrames(
            left_df_normal, right_df_normal,
            left.ts_col, right.ts_col,
            left.series_ids
        )

        skewed_result = self._standardAsOfJoinDataFrames(
            left_df_skewed, right_df_skewed,
            left.ts_col, right.ts_col,
            left.series_ids
        )

        # Union results
        result_df = normal_result.unionByName(skewed_result, allowMissingColumns=True)

        return result_df, TSSchema(ts_idx=left.ts_index, series_ids=left.series_ids)
```

---

## Migration Plan

### Phase 1: Create Utility Module (Week 1)

1. **Create** `tempo/utils/dataframe_ops.py` with core utilities
2. **Add comprehensive unit tests** for each utility function
3. **Document** all functions with examples and type hints

**Deliverables**:
- New module with 20+ utility functions
- 100+ unit tests
- Documentation with examples

### Phase 2: Refactor Join Strategies (Week 2)

1. **Update** `tempo/joins/strategies.py` to use new utilities
2. **Remove** local TSDF imports from strategy methods
3. **Verify** all 38 join tests still pass

**Changed Methods**:
- `AsOfJoiner._prefixColumns` → use `dataframe_ops.prefix_columns`
- `AsOfJoiner._prefixOverlappingColumns` → use `dataframe_ops.prefix_overlapping_columns`
- `UnionSortFilterAsOfJoiner._appendNullColumns` → use `dataframe_ops.add_null_columns`
- `SkewAsOfJoiner._skewSeparatedJoin` → use pure DataFrame ops

### Phase 3: Refactor TSDF Methods (Week 3)

1. **Update** `TSDF.__addPrefixToColumns` to use utilities
2. **Update** `TSDF.__addColumnsFromOtherDF` to use utilities
3. **Update** `TSDF.__combineTSDF` to use utilities

**Benefits**:
- Consistent behavior across all prefix operations
- Easier to test TSDF methods
- Reduced duplication

### Phase 4: Remove Deprecated Code (Week 4)

1. **Delete** `tempo/as_of_join.py` (old duplicate implementation)
2. **Remove** duplicate helper functions from `tsdf.py`
3. **Clean up** any remaining circular dependency workarounds

---

## Code Examples

### Example 1: Prefix Handling in Joins

**Current Code** (in `strategies.py`):
```python
def _prefixColumns(self, tsdf, prefixable_cols: set, prefix: str):
    if prefix:
        tsdf = reduce(
            lambda cur_tsdf, c: cur_tsdf.withColumnRenamed(c, "_".join([prefix, c])),
            prefixable_cols,
            tsdf,
        )
    return tsdf
```

**Refactored Code**:
```python
from tempo.utils.dataframe_ops import prefix_columns

def _prefixOverlappingColumns(self, left, right) -> tuple:
    overlapping = self._prefixableColumns(left, right)

    # Use pure DataFrame operation
    left_df = prefix_columns(left.df, overlapping, self.left_prefix)
    right_df = prefix_columns(right.df, overlapping, self.right_prefix)

    # Return DataFrames with metadata
    return (left_df, left.ts_schema), (right_df, right.ts_schema)
```

### Example 2: DataFrame Alignment for Union

**Current Code** (in `UnionSortFilterAsOfJoiner`):
```python
def _appendNullColumns(self, tsdf, cols: set):
    return reduce(
        lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)),
        cols,
        tsdf
    )

# Usage
right_only_cols = set(right.columns) - set(left.columns)
left_only_cols = set(left.columns) - set(right.columns)
extended_left = self._appendNullColumns(left, right_only_cols)
extended_right = self._appendNullColumns(right, left_only_cols)
```

**Refactored Code**:
```python
from tempo.utils.dataframe_ops import align_dataframes_for_union

# Single function call
left_df, right_df = align_dataframes_for_union(left.df, right.df)
```

### Example 3: Schema Validation

**Current Code** (scattered across classes):
```python
# In AsOfJoiner
if left.ts_schema != right.ts_schema:
    raise ValueError(...)

# In TSDF
left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
right_ts_datatype = right_tsdf.df.select(right_tsdf.ts_col).dtypes[0][1]
if left_ts_datatype != right_ts_datatype:
    raise ValueError(...)
```

**Refactored Code**:
```python
from tempo.utils.dataframe_ops import (
    validate_column_types_match,
    validate_partition_columns_match
)

# Clear, reusable validation
validate_column_types_match(left.df.schema, right.df.schema, ts_col)
validate_partition_columns_match(left.series_ids, right.series_ids)
```

---

## Benefits Analysis

### 1. **Code Reduction**

| Module | Current LOC | After Refactor | Reduction |
|--------|-------------|----------------|-----------|
| `strategies.py` | 1207 | ~950 | ~20% |
| `tsdf.py` | 1883 | ~1700 | ~10% |
| `as_of_join.py` | 457 | 0 (deleted) | 100% |
| **Total** | **3547** | **2650** | **~25%** |

### 2. **Testability Improvements**

**Before**: Testing prefix logic requires creating TSDF objects
```python
def test_prefix_columns():
    # Need to create full TSDF with schema
    left = TSDF(left_df, ts_col="ts", series_ids=["id"])
    right = TSDF(right_df, ts_col="ts", series_ids=["id"])

    joiner = AsOfJoiner()
    result = joiner._prefixOverlappingColumns(left, right)
```

**After**: Testing prefix logic uses pure DataFrames
```python
def test_prefix_columns():
    # Simple DataFrame test
    left_df = spark.createDataFrame([(1, "a")], ["id", "value"])
    right_df = spark.createDataFrame([(1, "b")], ["id", "value"])

    result = prefix_overlapping_columns(left_df, right_df, "l", "r", {"id"})
    assert "l_value" in result[0].columns
    assert "r_value" in result[1].columns
```

### 3. **Reusability**

New utilities can be used in:
- `tempo/stats.py` for EMA calculations
- `tempo/resample.py` for resampling operations
- `tempo/interpol.py` for interpolation
- User code for custom transformations
- Other Spark-based libraries

### 4. **Circular Dependency Elimination**

**Before**:
```
strategies.py
  → (local import) → TSDF
    → (module import) → strategies.py
```

**After**:
```
strategies.py
  → dataframe_ops.py (no TSDF dependency)

TSDF
  → strategies.py (returns tuples)
```

---

## Testing Strategy

### Unit Tests for Utilities

```python
# tests/utils/test_dataframe_ops.py

class TestPrefixOperations:
    def test_prefix_columns_basic(self):
        """Test basic column prefixing"""

    def test_prefix_columns_empty_set(self):
        """Test with no columns to prefix"""

    def test_prefix_overlapping_excludes_join_keys(self):
        """Test that join keys are not prefixed"""

class TestSchemaValidation:
    def test_validate_column_types_match_success(self):
        """Test successful type validation"""

    def test_validate_column_types_match_failure(self):
        """Test type mismatch detection"""

    def test_validate_partition_columns_order_matters(self):
        """Test that partition column order is validated"""

class TestDataFrameAlignment:
    def test_align_dataframes_for_union(self):
        """Test DataFrame alignment for union"""

    def test_add_null_columns_with_types(self):
        """Test adding typed null columns"""
```

### Integration Tests

```python
# tests/join/test_strategies_refactored.py

class TestRefactoredStrategies:
    def test_asof_join_with_refactored_prefix_logic(self):
        """Verify join still works after refactoring"""

    def test_skew_join_without_local_tsdf_import(self):
        """Verify skew join doesn't need TSDF import"""

    def test_all_strategies_produce_same_results(self):
        """Regression test to ensure behavior unchanged"""
```

### Performance Tests

```python
# tests/performance/test_dataframe_ops_performance.py

class TestPerformance:
    def test_prefix_performance_large_dataset(self):
        """Ensure utilities scale to large DataFrames"""

    def test_alignment_performance(self):
        """Benchmark DataFrame alignment operations"""
```

---

## Risk Assessment

### Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing functionality | Low | High | Comprehensive test suite, gradual rollout |
| Performance regression | Low | Medium | Performance benchmarks, profiling |
| Incomplete migration | Medium | Low | Phased approach, can keep old code temporarily |
| API confusion | Low | Low | Clear documentation, deprecation warnings |

### Rollback Plan

1. **Utilities are additive**: New module doesn't replace anything initially
2. **Gradual migration**: One class at a time
3. **Feature flagging**: Could add flag to use old vs. new implementation
4. **Git history**: Easy to revert individual commits

---

## Success Criteria

### Completion Criteria

- ✅ All utility functions have >90% test coverage
- ✅ All 38 join tests pass with refactored code
- ✅ No local TSDF imports in `strategies.py`
- ✅ `as_of_join.py` deleted
- ✅ Performance benchmarks show no regression
- ✅ Documentation complete with examples

### Quality Metrics

- **Code Coverage**: >90% for new utilities
- **Performance**: No more than 5% regression on any benchmark
- **Code Reduction**: At least 20% fewer lines in affected modules
- **Duplication**: Zero duplicated prefix/validation logic

---

## Related Documentation

- [CIRCULAR_DEPENDENCY_REFACTOR.md](CIRCULAR_DEPENDENCY_REFACTOR.md) - Explains tuple pattern
- [ASOF_JOIN_ENHANCEMENTS.md](ASOF_JOIN_ENHANCEMENTS.md) - Current strategy implementation
- [Tempo API Documentation](https://databrickslabs.github.io/tempo/) - Public API reference

---

## Appendix A: Complete Utility Function List

### Prefix Operations (5 functions)
1. `get_overlapping_columns()` - Find column name overlaps
2. `prefix_columns()` - Add prefix to columns
3. `prefix_overlapping_columns()` - Prefix overlaps in two DataFrames
4. `remove_prefix_from_columns()` - Remove prefix from columns
5. `get_prefix_from_column_name()` - Extract prefix from column name

### Schema Validation (4 functions)
6. `validate_column_types_match()` - Check type compatibility
7. `validate_partition_columns_match()` - Check partition column compatibility
8. `validate_column_exists()` - Check column existence
9. `get_column_type()` - Get column data type

### DataFrame Preparation (5 functions)
10. `add_null_columns()` - Add null columns
11. `align_dataframes_for_union()` - Align DataFrames for union
12. `align_column_order()` - Reorder columns to match
13. `cast_columns_to_match()` - Cast columns to compatible types
14. `ensure_columns_exist()` - Add missing columns as nulls

### Column Filtering (4 functions)
15. `get_non_structural_columns()` - Get observation columns
16. `get_columns_by_pattern()` - Pattern-based column selection
17. `get_columns_by_type()` - Type-based column selection
18. `filter_columns_by_predicate()` - Predicate-based filtering

### Window Operations (3 functions)
19. `create_partitioned_window()` - Create window spec
20. `create_range_window()` - Create range-based window
21. `create_row_window()` - Create row-based window

### Metadata Tracking (3 functions)
22. `track_column_metadata()` - Track transformation metadata
23. `get_column_lineage()` - Track column lineage
24. `create_transformation_summary()` - Summarize transformation

---

## Appendix B: Migration Checklist

### Week 1: Foundation
- [ ] Create `tempo/utils/dataframe_ops.py`
- [ ] Implement prefix operations (5 functions)
- [ ] Implement schema validation (4 functions)
- [ ] Implement DataFrame preparation (5 functions)
- [ ] Write unit tests for all functions
- [ ] Add documentation and examples

### Week 2: Strategy Refactoring
- [ ] Refactor `AsOfJoiner._prefixColumns`
- [ ] Refactor `AsOfJoiner._prefixOverlappingColumns`
- [ ] Refactor `UnionSortFilterAsOfJoiner._appendNullColumns`
- [ ] Refactor `SkewAsOfJoiner._skewSeparatedJoin`
- [ ] Run full test suite (verify 38/38 pass)
- [ ] Performance benchmarking

### Week 3: TSDF Refactoring
- [ ] Refactor `TSDF.__addPrefixToColumns`
- [ ] Refactor `TSDF.__addColumnsFromOtherDF`
- [ ] Refactor `TSDF.__combineTSDF`
- [ ] Update TSDF tests
- [ ] Integration testing

### Week 4: Cleanup
- [ ] Delete `tempo/as_of_join.py`
- [ ] Remove duplicate helper functions
- [ ] Update documentation
- [ ] Final performance validation
- [ ] Create PR for review

---

**END OF PROPOSAL**
