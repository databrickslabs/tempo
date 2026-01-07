# Refactoring Summary: Shared DataFrame Utilities

**TL;DR**: Extract duplicated DataFrame transformation logic into a new `tempo/utils/dataframe_ops.py` module to eliminate circular dependencies, improve testability, and enable pure functional programming patterns. **This includes refactoring TSDF methods to use the same utilities**, ensuring logic exists in only one place.

---

## The Problem

Currently, DataFrame transformation logic is duplicated across multiple locations:

1. **`strategies.py`** - Has prefix/validation logic for joins
2. **`as_of_join.py`** - Duplicate implementation of the same logic
3. **`tsdf.py`** - Has its own prefix/validation methods

This creates these issues:
- **Local TSDF imports** in `strategies.py` to avoid circular dependencies
- **Duplicated logic** - same operations implemented 2-3 times
- **Inconsistent behavior** - subtle differences between implementations
- **Tight coupling** between strategies and TSDF class
- **Testing complexity** - need full TSDF objects to test simple operations

### Example of Current Duplication

**In `strategies.py` (lines 105-117)**:
```python
def _prefixColumns(self, tsdf, prefixable_cols: set, prefix: str):
    if prefix:
        tsdf = reduce(
            lambda cur_tsdf, c: cur_tsdf.withColumnRenamed(
                c, "_".join([prefix, c])
            ),
            prefixable_cols,
            tsdf,
        )
    return tsdf
```

**In `as_of_join.py` (lines 67-81)** - EXACT DUPLICATE:
```python
def _prefixColumns(
    self, tsdf: t_tsdf.TSDF, prefixable_cols: set[str], prefix: str
) -> t_tsdf.TSDF:
    if prefix:
        tsdf = reduce(
            lambda cur_tsdf, c: cur_tsdf.withColumnRenamed(
                c, "_".join([prefix, c])
            ),
            prefixable_cols,
            tsdf,
        )
    return tsdf
```

**In `tsdf.py` (lines 409-435)** - SIMILAR LOGIC:
```python
def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> TSDF:
    if not prefix:
        return self

    col_map = {col: "_".join([prefix, col]) for col in col_list}
    select_exprs = [
        sfn.col(col).alias(col_map[col]) if col in col_map else sfn.col(col)
        for col in self.df.columns
    ]
    renamed_df = self.df.select(*select_exprs)
    # ... update structural columns ...
    return TSDF(renamed_df, ts_col=ts_col, series_ids=partition_cols)
```

**Problem**: Same logic implemented 3 different ways!

---

## The Solution

Create `tempo/utils/dataframe_ops.py` with **24 pure DataFrame utility functions**. Then refactor **BOTH** strategies AND TSDF to use these utilities, eliminating all duplication.

### Architecture After Refactoring

```
┌─────────────────────────────────────┐
│   tempo/utils/dataframe_ops.py      │
│   (Pure DataFrame utilities)        │
│   - NO dependencies on TSDF         │
│   - 24 reusable functions           │
└─────────────────────────────────────┘
              ▲         ▲
              │         │
              │         │
    ┌─────────┴───┐    │
    │             │    │
    │             │    │
┌───┴────────┐  ┌─┴────────────┐
│ strategies │  │  tsdf.py     │
│  .py       │  │              │
│            │  │  Uses utils  │
│ Uses utils │  │  in methods  │
└────────────┘  └──────────────┘
```

**Key principle**: DataFrame utilities exist in ONE PLACE only. Everyone uses them.

---

## Utility Functions (24 total)

### 1. Prefix Operations (5 functions)

```python
def get_overlapping_columns(
    left_cols: List[str],
    right_cols: List[str],
    exclude_cols: Optional[Set[str]] = None
) -> Set[str]:
    """Find columns that appear in both DataFrames."""

def prefix_columns(
    df: DataFrame,
    columns_to_prefix: Set[str],
    prefix: str
) -> DataFrame:
    """Add a prefix to specified columns in a DataFrame."""

def prefix_overlapping_columns(
    left_df: DataFrame,
    right_df: DataFrame,
    left_prefix: str,
    right_prefix: str,
    exclude_cols: Optional[Set[str]] = None
) -> Tuple[DataFrame, DataFrame]:
    """Prefix overlapping columns to avoid naming conflicts."""

def remove_prefix_from_columns(
    df: DataFrame,
    prefix: str
) -> DataFrame:
    """Remove a prefix from column names."""

def get_column_prefix(
    column_name: str,
    separator: str = "_"
) -> Optional[str]:
    """Extract prefix from a column name."""
```

### 2. Schema Validation (4 functions)

```python
def validate_column_types_match(
    left_schema: StructType,
    right_schema: StructType,
    column_name: str
) -> bool:
    """Validate that a column has the same type in both schemas."""

def validate_partition_columns_match(
    left_cols: List[str],
    right_cols: List[str]
) -> bool:
    """Validate that partition columns match in order and names."""

def validate_column_exists(
    schema: StructType,
    column_name: str
) -> bool:
    """Check if column exists in schema."""

def get_column_type(
    schema: StructType,
    column_name: str
) -> str:
    """Get the data type of a column."""
```

### 3. DataFrame Preparation (5 functions)

```python
def add_null_columns(
    df: DataFrame,
    columns_to_add: Set[str],
    column_types: Optional[Dict[str, str]] = None
) -> DataFrame:
    """Add null columns to a DataFrame."""

def align_dataframes_for_union(
    left_df: DataFrame,
    right_df: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Align two DataFrames to have the same columns for union."""

def align_column_order(
    df: DataFrame,
    target_order: List[str]
) -> DataFrame:
    """Reorder DataFrame columns to match target order."""

def cast_columns_to_match(
    df: DataFrame,
    target_schema: StructType
) -> DataFrame:
    """Cast columns to match target schema types."""

def ensure_columns_exist(
    df: DataFrame,
    required_columns: List[str],
    default_type: str = "string"
) -> DataFrame:
    """Ensure required columns exist, add as nulls if missing."""
```

### 4. Column Filtering (4 functions)

```python
def get_non_structural_columns(
    df_columns: List[str],
    ts_col: str,
    partition_cols: List[str]
) -> List[str]:
    """Get columns that are not structural."""

def get_columns_by_pattern(
    df_columns: List[str],
    pattern: str
) -> List[str]:
    """Get columns matching a naming pattern."""

def get_columns_by_type(
    schema: StructType,
    type_filter: Union[str, List[str]]
) -> List[str]:
    """Get columns of specific data types."""

def filter_columns_by_predicate(
    df_columns: List[str],
    predicate: Callable[[str], bool]
) -> List[str]:
    """Filter columns using a predicate function."""
```

### 5. Window Operations (3 functions)

```python
def create_partitioned_window(
    partition_cols: List[str],
    order_col: str,
    ascending: bool = True
) -> WindowSpec:
    """Create a window specification for partitioned operations."""

def create_range_window(
    partition_cols: List[str],
    order_col: str,
    range_start: int,
    range_end: int
) -> WindowSpec:
    """Create a range-based window specification."""

def create_row_window(
    partition_cols: List[str],
    order_col: str,
    rows_start: int,
    rows_end: int
) -> WindowSpec:
    """Create a row-based window specification."""
```

### 6. Metadata Tracking (3 functions)

```python
def track_column_metadata(
    original_df: DataFrame,
    transformed_df: DataFrame
) -> Dict[str, Any]:
    """Track metadata about a DataFrame transformation."""

def get_column_lineage(
    transformations: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Track column lineage through transformations."""

def create_transformation_summary(
    original_df: DataFrame,
    transformed_df: DataFrame,
    operation_name: str
) -> str:
    """Create a human-readable transformation summary."""
```

---

## Refactoring Plan: Strategies AND TSDF

### Phase 1: Create Utility Module (Week 1)

**Deliverables**:
- New `tempo/utils/dataframe_ops.py` with all 24 functions
- 100+ unit tests covering all functions
- Complete documentation with examples

### Phase 2: Refactor Join Strategies (Week 2)

**Files Modified**: `tempo/joins/strategies.py`

**Changes**:

1. **Remove local TSDF import in `_skewSeparatedJoin`** (line 816):
```python
# BEFORE
def _skewSeparatedJoin(self, left, right, skewed_keys):
    from tempo.tsdf import TSDF  # ❌ Remove this

    left_skewed = TSDF(left.df.filter(...), ...)
    # ... etc

# AFTER
from tempo.utils.dataframe_ops import (
    prefix_overlapping_columns,
    align_dataframes_for_union
)

def _skewSeparatedJoin(self, left, right, skewed_keys):
    # ✅ No TSDF import needed - work with DataFrames directly
    left_df_skewed = left.df.filter(...)
    left_df_normal = left.df.filter(...)
    # ... process as DataFrames, return tuple
```

2. **Replace `_prefixColumns` method** (lines 105-117):
```python
# BEFORE (18 lines of code)
def _prefixColumns(self, tsdf, prefixable_cols: set, prefix: str):
    if prefix:
        tsdf = reduce(
            lambda cur_tsdf, c: cur_tsdf.withColumnRenamed(
                c, "_".join([prefix, c])
            ),
            prefixable_cols,
            tsdf,
        )
    return tsdf

# AFTER (3 lines of code)
from tempo.utils.dataframe_ops import prefix_columns

def _prefixColumns(self, tsdf, prefixable_cols: set, prefix: str):
    return prefix_columns(tsdf.df, prefixable_cols, prefix)
```

3. **Replace `_appendNullColumns` in UnionSortFilterAsOfJoiner** (lines 362-372):
```python
# BEFORE (11 lines)
def _appendNullColumns(self, tsdf, cols: set):
    return reduce(
        lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)), cols, tsdf
    )

# AFTER (3 lines)
from tempo.utils.dataframe_ops import add_null_columns

def _appendNullColumns(self, tsdf, cols: set):
    return add_null_columns(tsdf.df, cols)
```

### Phase 3: Refactor TSDF Methods (Week 3)

**Files Modified**: `tempo/tsdf.py`

**Changes**:

1. **Replace `__addPrefixToColumns` method** (lines 409-435):
```python
# BEFORE (27 lines of complex logic)
def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> TSDF:
    if not prefix:
        return self

    # build a column rename map
    col_map = {col: "_".join([prefix, col]) for col in col_list}

    # build a list of column expressions to rename columns in a select
    select_exprs = [
        sfn.col(col).alias(col_map[col]) if col in col_map else sfn.col(col)
        for col in self.df.columns
    ]
    renamed_df = self.df.select(*select_exprs)

    # find the structural columns
    ts_col = col_map.get(self.ts_col, self.ts_col)
    partition_cols = [col_map.get(c, c) for c in self.series_ids]
    return TSDF(renamed_df, ts_col=ts_col, series_ids=partition_cols)

# AFTER (8 lines - cleaner and reuses utility)
from tempo.utils.dataframe_ops import prefix_columns

def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> TSDF:
    if not prefix:
        return self

    renamed_df = prefix_columns(self.df, set(col_list), prefix)

    # Update structural columns
    ts_col = f"{prefix}_{self.ts_col}" if self.ts_col in col_list else self.ts_col
    partition_cols = [f"{prefix}_{c}" if c in col_list else c for c in self.series_ids]
    return TSDF(renamed_df, ts_col=ts_col, series_ids=partition_cols)
```

2. **Replace `__addColumnsFromOtherDF` method** (lines 437-447):
```python
# BEFORE (11 lines)
def __addColumnsFromOtherDF(self, other_cols: Sequence[str]) -> TSDF:
    current_cols = [sfn.col(col) for col in self.df.columns]
    new_cols = [sfn.lit(None).alias(col) for col in other_cols]
    new_df = self.df.select(current_cols + new_cols)
    return self.__withTransformedDF(new_df)

# AFTER (4 lines - using utility)
from tempo.utils.dataframe_ops import add_null_columns

def __addColumnsFromOtherDF(self, other_cols: Sequence[str]) -> TSDF:
    new_df = add_null_columns(self.df, set(other_cols))
    return self.__withTransformedDF(new_df)
```

3. **Simplify `__combineTSDF` method** (lines 449-454):
```python
# BEFORE (uses manual union and coalesce)
def __combineTSDF(self, ts_df_right: TSDF, combined_ts_col: str) -> TSDF:
    combined_df = self.df.unionByName(ts_df_right.df).withColumn(
        combined_ts_col, sfn.coalesce(self.ts_col, ts_df_right.ts_col)
    )
    return TSDF(combined_df, ts_col=combined_ts_col, series_ids=self.series_ids)

# AFTER (can use alignment utility first if needed)
from tempo.utils.dataframe_ops import align_dataframes_for_union

def __combineTSDF(self, ts_df_right: TSDF, combined_ts_col: str) -> TSDF:
    # Ensure DataFrames are aligned before union
    left_aligned, right_aligned = align_dataframes_for_union(self.df, ts_df_right.df)

    combined_df = left_aligned.unionByName(right_aligned).withColumn(
        combined_ts_col, sfn.coalesce(self.ts_col, ts_df_right.ts_col)
    )
    return TSDF(combined_df, ts_col=combined_ts_col, series_ids=self.series_ids)
```

4. **Replace `__checkPartitionCols` validation** (lines 393-398):
```python
# BEFORE (6 lines with manual loop)
def __checkPartitionCols(self, tsdf_right: TSDF) -> None:
    for left_col, right_col in zip(self.series_ids, tsdf_right.series_ids):
        if left_col != right_col:
            raise ValueError(
                "left and right dataframe partition columns should have same name in same order"
            )

# AFTER (2 lines using utility)
from tempo.utils.dataframe_ops import validate_partition_columns_match

def __checkPartitionCols(self, tsdf_right: TSDF) -> None:
    validate_partition_columns_match(self.series_ids, tsdf_right.series_ids)
```

5. **Replace `__validateTsColMatch` method** (lines 400-407):
```python
# BEFORE (8 lines with manual type checking)
def __validateTsColMatch(self, right_tsdf: TSDF) -> None:
    left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
    right_ts_datatype = right_tsdf.df.select(right_tsdf.ts_col).dtypes[0][1]
    if left_ts_datatype != right_ts_datatype:
        raise ValueError(
            "left and right dataframe timestamp index columns should have same type"
        )

# AFTER (2 lines using utility)
from tempo.utils.dataframe_ops import validate_column_types_match

def __validateTsColMatch(self, right_tsdf: TSDF) -> None:
    validate_column_types_match(self.df.schema, right_tsdf.df.schema, self.ts_col)
```

### Phase 4: Delete Duplicate Code (Week 4)

**Files Deleted**:
- ❌ `tempo/as_of_join.py` - Complete duplicate of `strategies.py`

**Files Cleaned**:
- 📝 `tempo/joins/strategies.py` - Remove duplicate helper functions
- 📝 `tempo/tsdf.py` - Remove duplicate logic now in utilities

---

## Benefits Breakdown

### 1. Eliminate ALL Duplication ✅

**Current State**: Same logic in 3 places
- `strategies.py` - prefix logic (18 LOC)
- `as_of_join.py` - prefix logic (18 LOC) - DUPLICATE
- `tsdf.py` - prefix logic (27 LOC) - SIMILAR

**After Refactoring**: Logic in ONE place
- `dataframe_ops.py` - prefix logic (15 LOC)
- `strategies.py` - calls utility (3 LOC)
- `tsdf.py` - calls utility (3 LOC)

**Total**: 63 LOC → 21 LOC = **67% reduction**

### 2. Code Reduction by File ✅

| File | Current LOC | After | Reduction |
|------|-------------|-------|-----------|
| `strategies.py` | 1,207 | ~900 | -25% |
| `tsdf.py` | 1,883 | ~1,650 | -12% |
| `as_of_join.py` | 457 | 0 (deleted) | -100% |
| `dataframe_ops.py` | 0 | +600 (new) | +600 |
| **Net Total** | **3,547** | **3,150** | **-11%** |

**Actual code reduction**: 397 lines eliminated through deduplication!

### 3. No More Circular Dependencies ✅

**Before**:
```
strategies.py
  → (local import at line 816) → TSDF
    → (module import at line 970) → strategies.py
      🔴 CIRCULAR DEPENDENCY
```

**After**:
```
dataframe_ops.py
  → No dependencies on TSDF ✅

strategies.py
  → dataframe_ops.py ✅
  → Returns tuples (DataFrame, TSSchema) ✅

TSDF
  → dataframe_ops.py ✅
  → strategies.py (imports at module level) ✅
```

### 4. Consistent Behavior ✅

**Before**: 3 different implementations of prefix logic
- `strategies.py` - Uses `reduce` with `withColumnRenamed`
- `as_of_join.py` - Uses `reduce` with `withColumnRenamed` (slightly different)
- `tsdf.py` - Uses `select` with `alias` (different approach!)

**After**: 1 implementation
- `dataframe_ops.py` - Single, well-tested implementation
- Everyone uses the same function
- Guaranteed consistent behavior

### 5. Better Testing ✅

**Before**: Need TSDF objects to test basic operations
```python
def test_prefix_in_strategies():
    # Need full TSDF setup
    left_df = spark.createDataFrame(...)
    left = TSDF(left_df, ts_col="ts", series_ids=["id"])
    right_df = spark.createDataFrame(...)
    right = TSDF(right_df, ts_col="ts", series_ids=["id"])

    joiner = AsOfJoiner()
    result = joiner._prefixColumns(left, {"value"}, "l")
    # ... assertions

def test_prefix_in_tsdf():
    # Different test for same logic!
    df = spark.createDataFrame(...)
    tsdf = TSDF(df, ts_col="ts", series_ids=["id"])
    result = tsdf.__addPrefixToColumns(["value"], "l")
    # ... assertions
```

**After**: Test once, use everywhere
```python
def test_prefix_columns():
    # Single test for the utility
    df = spark.createDataFrame([(1, "a")], ["id", "value"])
    result = prefix_columns(df, {"value"}, "l")
    assert "l_value" in result.columns
    assert "id" in result.columns

# Both strategies AND TSDF use this tested utility!
```

---

## Complete List of TSDF Methods to Refactor

| TSDF Method | Lines | Uses Utility | LOC Saved |
|-------------|-------|--------------|-----------|
| `__addPrefixToColumns` | 409-435 | `prefix_columns` | 19 |
| `__addColumnsFromOtherDF` | 437-447 | `add_null_columns` | 7 |
| `__combineTSDF` | 449-454 | `align_dataframes_for_union` | 2 |
| `__checkPartitionCols` | 393-398 | `validate_partition_columns_match` | 4 |
| `__validateTsColMatch` | 400-407 | `validate_column_types_match` | 6 |
| **Total** | | | **38 LOC** |

---

## Implementation Checklist

### Week 1: Foundation ✅
- [ ] Create `tempo/utils/dataframe_ops.py`
- [ ] Implement all 24 utility functions
- [ ] Write 100+ unit tests
- [ ] Add comprehensive documentation
- [ ] Code review and approval

### Week 2: Strategies ✅
- [ ] Refactor `AsOfJoiner._prefixColumns`
- [ ] Refactor `AsOfJoiner._prefixOverlappingColumns`
- [ ] Refactor `UnionSortFilterAsOfJoiner._appendNullColumns`
- [ ] Refactor `SkewAsOfJoiner._skewSeparatedJoin` (remove local import)
- [ ] Run full test suite (38/38 tests)
- [ ] Performance benchmarks

### Week 3: TSDF ✅
- [ ] Refactor `TSDF.__addPrefixToColumns`
- [ ] Refactor `TSDF.__addColumnsFromOtherDF`
- [ ] Refactor `TSDF.__combineTSDF`
- [ ] Refactor `TSDF.__checkPartitionCols`
- [ ] Refactor `TSDF.__validateTsColMatch`
- [ ] Run TSDF test suite
- [ ] Integration testing

### Week 4: Cleanup ✅
- [ ] Delete `tempo/as_of_join.py`
- [ ] Remove any remaining duplicate functions
- [ ] Update all documentation
- [ ] Final performance validation
- [ ] Create comprehensive PR

---

## Success Criteria

- ✅ All 38 join tests pass
- ✅ All TSDF tests pass
- ✅ No local TSDF imports in `strategies.py`
- ✅ No duplicate prefix/validation logic anywhere
- ✅ 90%+ test coverage for new utilities
- ✅ No performance regression (within 5%)
- ✅ Net code reduction of 10%+
- ✅ `as_of_join.py` deleted
- ✅ Complete documentation

---

## Impact Summary

### Before
```
3,547 lines across 3 files
❌ Duplicate logic in 3 places
❌ Circular dependencies
❌ Local imports as workaround
❌ Inconsistent behavior
❌ Hard to test
```

### After
```
3,150 lines across 3 files (+ utilities)
✅ Logic in ONE place only
✅ No circular dependencies
✅ No local imports needed
✅ Consistent behavior everywhere
✅ Easy to test independently
```

**Net Result**: -397 lines, cleaner architecture, better testability, consistent behavior

---

## Related Documents

- **Full Proposal**: [REFACTOR_PROPOSAL_SHARED_UTILITIES.md](REFACTOR_PROPOSAL_SHARED_UTILITIES.md)
- **Circular Dependency Pattern**: [CIRCULAR_DEPENDENCY_REFACTOR.md](CIRCULAR_DEPENDENCY_REFACTOR.md)
- **Join Enhancements**: [ASOF_JOIN_ENHANCEMENTS.md](ASOF_JOIN_ENHANCEMENTS.md)

---

**Questions?** See [REFACTOR_PROPOSAL_SHARED_UTILITIES.md](REFACTOR_PROPOSAL_SHARED_UTILITIES.md) for complete implementation details.
