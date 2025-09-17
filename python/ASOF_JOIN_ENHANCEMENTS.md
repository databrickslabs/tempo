# As-Of Join Enhancements Implementation Strategy

## Implementation Instructions
**IMPORTANT**: This document serves as the source of truth for all as-of join improvements.

### Key Guidelines:
1. **Update this document for every change and todo**
2. **Keep changes scoped and complete** - Each commit should represent a fully working incremental change
3. **Run tests between each commit** to verify no breaking changes have been introduced
4. **Commit frequently** for easy rollbacks if issues arise

### Testing Commands:
- **Run all tests for current DBR version (154)**: `make test` or `hatch run dbr154:test`
- **Run specific test file**: `hatch run dbr154:test tests/as_of_join_tests.py`
- **Run all DBR versions**: `make test-all`
- **Run linting**: `make lint` or `hatch run lint:runLint`
- **Run type checking**: `make type-check` or `hatch run type-check:check`
- **Generate coverage report**: `make coverage-report`

## Overview
This document provides a comprehensive implementation strategy for enhancing as-of join features in the Tempo library. It consolidates experimental work from the `as_of_join_refactor` branch with the existing v0.2-integration codebase, addressing all known issues and incomplete implementations.

## Critical Issues from Experimental Branch

### Known Bugs and TODOs
1. **Timezone Handling Issues** (from commit c21deed)
   - BroadcastAsOfJoiner: Composite timestamp indexes with nanosecond precision have timezone inconsistencies
   - UnionSortFilterAsOfJoiner: Produces timestamps with different timezone offsets compared to broadcast join
   - Test failures noted in as_of_join_tests.py for "nanos" test case

2. **Incomplete Implementations**
   - `_toleranceFilter()` method returns unchanged TSDF (line 274-278 in experimental branch)
   - `SkewAsOfJoiner._join()` simply calls parent class without skew-specific logic (line 313-314)
   - Missing comparable expression handling for composite indexes in BroadcastAsOfJoiner

3. **Test Discrepancies**
   - Broadcast join returns different row counts than union join for certain test cases
   - Simple_ts test: union join returns 4 rows (left join behavior) vs broadcast's 3 rows (inner join behavior)
   - Nanos test: skipped due to timezone handling issues

## Features to Implement

### 1. Strategy Pattern for Join Types
**Goal**: Implement a flexible strategy pattern that allows selection of different join algorithms based on data characteristics.

#### Current State
- The v0.2-integration branch has `asofJoin` as a method in the TSDF class (lines 931-1133 in tsdf.py)
- Current implementation mixes broadcast join and union-sort-filter logic in a single method
- Uses conditional logic based on `sql_join_opt` and `tsPartitionVal` parameters

#### Implementation Strategy
1. **Create Abstract Strategy Interface**
   - Move the experimental `AsOfJoiner` abstract class to a new module `tempo/joins/strategies.py`
   - Define the contract: `_join()`, `_checkAreJoinable()`, `_prefixOverlappingColumns()`
   - Ensure compatibility with existing TSDF class structure

2. **Implement Concrete Strategies**
   - `BroadcastAsOfJoiner`: For small datasets that fit in memory
   - `UnionSortFilterAsOfJoiner`: Default strategy for general cases
   - `SkewAsOfJoiner`: For handling skewed data distributions

3. **Integration with TSDF**
   - Modify `TSDF.asofJoin()` to use strategy pattern internally
   - Maintain backward compatibility with existing API
   - Add optional `strategy` parameter to allow manual strategy selection

#### Implementation Steps
- [ ] Create `tempo/joins/` package with `__init__.py`
- [ ] Extract and refactor strategy classes from experimental branch
- [ ] Fix timezone handling in BroadcastAsOfJoiner for composite indexes
- [ ] Update TSDF.asofJoin to delegate to strategy objects
- [ ] Add comprehensive tests for each strategy
- [ ] Ensure consistent join behavior (left vs inner join semantics)

#### Detailed Code Structure
```python
# tempo/joins/strategies.py
from abc import ABC, abstractmethod

class AsOfJoiner(ABC):
    def __init__(self, left_prefix: str = "left", right_prefix: str = "right"):
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix

    def __call__(self, left: TSDF, right: TSDF) -> TSDF:
        self._checkAreJoinable(left, right)
        left, right = self._prefixOverlappingColumns(left, right)
        return self._join(left, right)

    @abstractmethod
    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        pass

class BroadcastAsOfJoiner(AsOfJoiner):
    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        # FIX: Handle composite indexes with comparableExpr()
        right_comparable_expr = right.ts_index.comparableExpr()
        if isinstance(right_comparable_expr, list):
            right_comparable_expr = right_comparable_expr[0] if right_comparable_expr else sfn.col(right.ts_index.colname)
        # ... rest of implementation
```

### 2. Automatic Strategy Selection
**Goal**: Automatically choose the optimal join strategy based on data characteristics.

#### Current State
- No automatic optimization in current implementation
- Experimental branch has `choose_as_of_join_strategy()` function with size-based selection

#### Implementation Strategy
1. **Size-based Selection**
   - Analyze DataFrame sizes using Spark plan metadata
   - Use broadcast join for datasets below threshold (30MB default)
   - Make threshold configurable via Spark config

2. **Skew Detection**
   - Analyze data distribution across partitions
   - Automatically use SkewAsOfJoiner when significant skew is detected
   - Add metrics collection for monitoring strategy selection

3. **Configuration Options**
   - Add `spark.tempo.asof.join.auto_optimize` config (default: true)
   - Add `spark.tempo.asof.join.broadcast_threshold` config
   - Allow override via method parameter

#### Implementation Steps
- [ ] Port `get_spark_plan()` and `get_bytes_from_plan()` utilities from experimental branch
- [ ] Fix regex pattern in `get_bytes_from_plan()` for size extraction
- [ ] Implement skew detection logic using partition statistics
- [ ] Create strategy selection heuristics
- [ ] Add logging for strategy selection decisions
- [ ] Create performance benchmarks

#### Detailed Implementation
```python
def choose_as_of_join_strategy(
    left_tsdf: TSDF,
    right_tsdf: TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    tolerance: Optional[int] = None,
) -> AsOfJoiner:
    """
    Returns an AsOfJoiner based on data characteristics.

    Decision Tree:
    1. If sql_join_opt and data < 30MB: BroadcastAsOfJoiner
    2. If tsPartitionVal is set: SkewAsOfJoiner
    3. Default: UnionSortFilterAsOfJoiner
    """
    if sql_join_opt:
        spark = SparkSession.builder.getOrCreate()
        left_bytes = get_bytes_from_plan(left_tsdf.df, spark)
        right_bytes = get_bytes_from_plan(right_tsdf.df, spark)

        # 30MB threshold for broadcast
        bytes_threshold = 30 * 1024 * 1024
        if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
            logger.info(f"Using BroadcastAsOfJoiner (left: {left_bytes/1024/1024:.2f}MB, right: {right_bytes/1024/1024:.2f}MB)")
            return BroadcastAsOfJoiner(spark, left_prefix or "left", right_prefix)

    if tsPartitionVal is not None:
        logger.info(f"Using SkewAsOfJoiner with partition value {tsPartitionVal}")
        return SkewAsOfJoiner(
            left_prefix=left_prefix or "left",
            right_prefix=right_prefix,
            skipNulls=skipNulls,
            tolerance=tolerance,
            tsPartitionVal=tsPartitionVal,
            fraction=fraction,
        )

    logger.info("Using default UnionSortFilterAsOfJoiner")
    return UnionSortFilterAsOfJoiner(left_prefix or "left", right_prefix, skipNulls, tolerance)
```

### 3. Skew-Aware Join Handling
**Goal**: Optimize performance for datasets with skewed distributions.

#### Current State
- Current implementation has basic skew handling with `tsPartitionVal` parameter
- Uses `__getTimePartitions()` method in TSDF class for time-based partitioning
- Experimental branch has incomplete `SkewAsOfJoiner` class that needs completion

#### Implementation Strategy
1. **Skew Detection**
   - Calculate partition size variance
   - Identify hot partitions (>2x average size)
   - Log skew metrics for monitoring

2. **Adaptive Partitioning**
   - Implement salting for hot partitions
   - Use configurable split threshold
   - Maintain result correctness with proper aggregation

3. **Optimization Techniques**
   - Implement partial broadcast for small lookup tables
   - Use range partitioning for time-based splits
   - Add adaptive execution hints for Spark 3.x

#### Implementation Steps
- [ ] Complete SkewAsOfJoiner implementation based on current TSDF.__getTimePartitions() logic
- [ ] Add partition analysis utilities
- [ ] Implement salting mechanism for extreme skew
- [ ] Add configuration for skew thresholds
- [ ] Create tests with skewed datasets
- [ ] Handle overlapped data removal (is_original flag)

#### Detailed Implementation
```python
class SkewAsOfJoiner(UnionSortFilterAsOfJoiner):
    """
    Implements the as-of join strategy for skewed data.
    Uses time-based partitioning to handle skew.
    """

    def __init__(
        self,
        left_prefix: str = "left",
        right_prefix: str = "right",
        skipNulls: bool = True,
        tolerance: Optional[int] = None,
        tsPartitionVal: Optional[int] = None,
        fraction: float = 0.5,
    ):
        super().__init__(left_prefix, right_prefix, skipNulls, tolerance)
        self.tsPartitionVal = tsPartitionVal
        self.fraction = fraction

    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        # Combine left and right TSDFs
        combined = self._combine(left, right)

        # Apply time-based partitioning for skew handling
        if self.tsPartitionVal:
            # This mimics the current TSDF.__getTimePartitions() logic
            # Creates overlapping time windows to handle edge cases
            combined = self._applyTimePartitions(combined)

        # Filter to get last right row for each left row
        result = self._filterLastRightRow(combined, right_cols, left.ts_schema)

        # Remove overlapped data if time partitioning was used
        if self.tsPartitionVal:
            result.df = result.df.filter(sfn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

        # Apply tolerance filter if specified
        if self.tolerance is not None:
            result = self._toleranceFilter(result)

        return result

    def _applyTimePartitions(self, tsdf: TSDF) -> TSDF:
        """
        Apply time-based partitioning with overlap to handle skew.
        Based on TSDF.__getTimePartitions() implementation.
        """
        # Implementation would mirror __getTimePartitions from TSDF class
        pass
```

### 4. Tolerance Parameter
**Goal**: Filter join results based on time proximity between matched records.

#### Current State
- Current v0.2-integration implementation HAS tolerance support (lines 1105-1131 in tsdf.py)
- Filters results where time difference exceeds tolerance (in seconds)
- Experimental branch has tolerance parameter but `_toleranceFilter()` is not implemented

#### Implementation Strategy
1. **Time-based Filtering**
   - Add `tolerance` parameter (in seconds) to asofJoin method
   - Filter results where time difference exceeds tolerance
   - Support both forward and backward tolerance

2. **API Design**
   ```python
   # Example usage
   result = left_tsdf.asofJoin(
       right_tsdf,
       tolerance=3600,  # 1 hour tolerance
       tolerance_mode="backward"  # or "forward", "both"
   )
   ```

3. **Performance Optimization**
   - Apply tolerance filter during join when possible
   - Use partition pruning for large tolerances
   - Add index hints for time-based filtering

#### Implementation Steps
- [ ] Port existing tolerance logic from TSDF.asofJoin to strategy classes
- [ ] Implement `_toleranceFilter()` method in UnionSortFilterAsOfJoiner
- [ ] Add tolerance_mode parameter for forward/backward/both filtering
- [ ] Add validation for tolerance values
- [ ] Create comprehensive test cases

#### Detailed Implementation (Based on Current TSDF Implementation)
```python
def _toleranceFilter(self, as_of: TSDF, left_ts_col: str, right_ts_col: str,
                     tolerance: int, right_columns: list) -> TSDF:
    """
    Filters out rows from the as_of TSDF that are outside the tolerance.
    Based on current implementation in TSDF.asofJoin (lines 1105-1131).
    """
    df = as_of.df

    # Calculate time difference (in seconds)
    tolerance_condition = (
        df[left_ts_col].cast("double") - df[right_ts_col].cast("double") > tolerance
    )

    # Set right columns to null for rows outside tolerance
    for right_col in right_columns:
        if right_col != right_ts_col:
            df = df.withColumn(
                right_col,
                sfn.when(tolerance_condition, sfn.lit(None)).otherwise(df[right_col])
            )

    # Finally, set right timestamp column to null
    df = df.withColumn(
        right_ts_col,
        sfn.when(tolerance_condition, sfn.lit(None)).otherwise(df[right_ts_col])
    )

    as_of.df = df
    return as_of
```

### 5. Skip Nulls Option
**Goal**: Control null handling behavior in join results.

#### Current State
- Current v0.2-integration implementation has `skipNulls` parameter
- Used in `__getLastRightRow()` method in TSDF class
- Experimental branch implements it in `_filterLastRightRow()` method

#### Implementation Strategy
1. **Null Handling Options**
   - `skipNulls=True`: Ignore null values when finding last matching record
   - `skipNulls=False`: Include nulls in the join result
   - Add `null_fill` parameter for custom null replacement

2. **Implementation Details**
   - Use Spark's `last(ignoreNulls=True/False)` function
   - Handle nulls consistently across all strategies
   - Document behavior clearly in API docs

3. **API Enhancement**
   ```python
   result = left_tsdf.asofJoin(
       right_tsdf,
       skipNulls=True,  # Skip null values
       null_fill={"column1": 0, "column2": "default"}  # Optional fill values
   )
   ```

#### Implementation Steps
- [ ] Maintain skipNulls parameter compatibility
- [ ] Ensure consistent null handling across all strategies
- [ ] Add null_fill functionality for custom replacements
- [ ] Create tests for various null scenarios
- [ ] Update documentation

#### Detailed Implementation (From Experimental Branch)
```python
def _filterLastRightRow(self, combined: TSDF, right_cols: set[str],
                        last_left_tsschema: TSSchema) -> TSDF:
    """
    Filters out the last right-hand row for each left-hand row.
    Handles skipNulls parameter for null value treatment.
    """
    w = combined.allBeforeWindow()
    right_row_field = combined.ts_index.fieldPath(_DEFAULT_RIGHT_ROW_COLNAME)

    if self.skipNulls:
        # Use Spark's last() with ignoreNulls=True
        last_right_cols = [
            sfn.last(col, True).over(w).alias(col) for col in right_cols
        ]
    else:
        # Include nulls in the window calculation
        last_right_cols = [
            sfn.last(
                sfn.when(
                    sfn.col(right_row_field) == _RIGHT_HAND_ROW_INDICATOR,
                    sfn.struct(col)
                ).otherwise(None),
                True
            ).over(w).alias(col)
            for col in right_cols
        ]

    # Apply the window functions and filter
    non_right_cols = list(set(combined.columns) - right_cols)
    last_right_vals = combined.select(*(non_right_cols + last_right_cols))

    as_of_df = last_right_vals.df.where(
        sfn.col(right_row_field) == _LEFT_HAND_ROW_INDICATOR
    ).drop(_DEFAULT_COMBINED_TS_COLNAME)

    return TSDF(as_of_df, ts_schema=copy.deepcopy(last_left_tsschema))
```

## Complete Integration with TSDF Class

### Modified TSDF.asofJoin Method
```python
def asofJoin(
    self,
    right_tsdf: TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    strategy: Optional[str] = None,  # NEW: manual strategy override
) -> TSDF:
    """
    Enhanced as-of join with strategy pattern.

    Parameters:
    ... (existing parameters) ...
    :param strategy: Optional manual strategy selection ("broadcast", "union_sort_filter", "skew", "auto")
    """

    # Import the strategy module
    from tempo.joins.strategies import choose_as_of_join_strategy, BroadcastAsOfJoiner, UnionSortFilterAsOfJoiner, SkewAsOfJoiner

    # Manual strategy override
    if strategy:
        if strategy == "broadcast":
            joiner = BroadcastAsOfJoiner(SparkSession.builder.getOrCreate(), left_prefix or "", right_prefix)
        elif strategy == "union_sort_filter":
            joiner = UnionSortFilterAsOfJoiner(left_prefix or "", right_prefix, skipNulls, tolerance)
        elif strategy == "skew":
            joiner = SkewAsOfJoiner(left_prefix or "", right_prefix, skipNulls, tolerance, tsPartitionVal, fraction)
        else:  # auto or invalid
            joiner = choose_as_of_join_strategy(
                self, right_tsdf, left_prefix, right_prefix,
                tsPartitionVal, fraction, skipNulls, sql_join_opt, tolerance
            )
    else:
        # Automatic strategy selection
        joiner = choose_as_of_join_strategy(
            self, right_tsdf, left_prefix, right_prefix,
            tsPartitionVal, fraction, skipNulls, sql_join_opt, tolerance
        )

    # Execute the join
    return joiner(self, right_tsdf)
```

## Critical Implementation Details from Experimental Branch

### 1. Composite Index Handling for Nanosecond Precision
```python
# From BroadcastAsOfJoiner._join() - MUST be fixed
def _join(self, left: TSDF, right: TSDF) -> TSDF:
    # CRITICAL FIX for composite indexes with nanosecond precision
    # The experimental branch had issues with timezone handling

    # Get comparable expression for the timestamp index
    right_comparable_expr = right.ts_index.comparableExpr()

    # For composite indexes, comparableExpr() returns a list
    # We need the first element (double_ts for nanos)
    if isinstance(right_comparable_expr, list):
        if len(right_comparable_expr) > 0:
            right_comparable_expr = right_comparable_expr[0]
        else:
            # Fallback to column name
            right_comparable_expr = sfn.col(right.ts_index.colname)

    # Use comparable expression for lead calculation
    lead_colname = "lead_" + right.ts_index.colname
    right_with_lead = right.withColumn(
        lead_colname, sfn.lead(right_comparable_expr).over(w)
    )

    # ... rest of join logic
```

### 2. Combined Timestamp Column Creation
```python
# From UnionSortFilterAsOfJoiner._combine()
def _combine(self, left: TSDF, right: TSDF) -> TSDF:
    """
    CRITICAL: Properly handle composite timestamp indexes
    """
    # Union the dataframes
    unioned = left.unionByName(right)

    # Get comparable expressions
    left_comps = left.ts_index.comparableExpr()
    right_comps = right.ts_index.comparableExpr()

    # Ensure list format
    if not isinstance(left_comps, list):
        left_comps = [left_comps]
    if not isinstance(right_comps, list):
        right_comps = [right_comps]

    # Get component names
    if isinstance(left.ts_index, CompositeTSIndex):
        comp_names = left.ts_index.component_fields
    else:
        comp_names = [left.ts_index.colname]

    # Coalesce components
    combined_comps = [
        sfn.coalesce(lc, rc).alias(cn)
        for lc, rc, cn in zip(left_comps, right_comps, comp_names)
    ]

    # Add row indicator
    right_ind = (
        sfn.when(sfn.col(left.ts_index.colname).isNotNull(), _LEFT_HAND_ROW_INDICATOR)
        .otherwise(_RIGHT_HAND_ROW_INDICATOR)
        .alias(_DEFAULT_RIGHT_ROW_COLNAME)
    )

    # Create combined struct column
    with_combined_ts = unioned.withColumn(
        _DEFAULT_COMBINED_TS_COLNAME,
        sfn.struct(*combined_comps, right_ind)
    )

    # Create CompositeTSIndex for the combined column
    combined_ts_col = with_combined_ts.df.schema[_DEFAULT_COMBINED_TS_COLNAME]
    combined_comp_names = comp_names + [_DEFAULT_RIGHT_ROW_COLNAME]
    combined_tsidx = _AsOfJoinCompositeTSIndex(combined_ts_col, *combined_comp_names)

    return TSDF(with_combined_ts.df, ts_schema=TSSchema(combined_tsidx, left.series_ids))
```

### 3. Helper Class for Composite Index
```python
class _AsOfJoinCompositeTSIndex(CompositeTSIndex):
    """
    Special CompositeTSIndex subclass for as-of joins.
    IMPORTANT: Does not implement rangeExpr as it's not needed for join operations.
    """

    @property
    def unit(self) -> Optional[TimeUnit]:
        return None

    def rangeExpr(self, reverse: bool = False) -> Column:
        # This is intentionally not implemented
        raise NotImplementedError(
            "rangeExpr is not defined for as-of join composite ts index"
        )
```

## Comprehensive Testing Strategy for 100% Coverage

### Unit Test Specifications

#### 1. AsOfJoiner Base Class Tests
```python
# tests/test_asof_joiner_base.py
class TestAsOfJoinerBase:
    """Test the abstract base class and common functionality."""

    def test_init_default_prefixes(self):
        """Test default prefix initialization."""
        # Assert left_prefix="left", right_prefix="right"

    def test_init_custom_prefixes(self):
        """Test custom prefix initialization."""
        # Test with empty strings, None values, special characters

    def test_call_method_flow(self):
        """Test __call__ method calls correct sequence."""
        # Mock _checkAreJoinable, _prefixOverlappingColumns, _join
        # Verify call order and parameters

    def test_common_series_ids(self):
        """Test commonSeriesIDs method."""
        # Test with matching series IDs
        # Test with partial overlap
        # Test with no overlap
        # Test with empty series IDs

    def test_prefixable_columns(self):
        """Test _prefixableColumns identification."""
        # Test with overlapping columns
        # Test with series ID columns (should be excluded)
        # Test with no overlapping columns

    def test_prefix_columns(self):
        """Test _prefixColumns with various scenarios."""
        # Test with valid prefix
        # Test with empty prefix (should return unchanged)
        # Test with None prefix
        # Test column renaming accuracy

    def test_prefix_overlapping_columns(self):
        """Test _prefixOverlappingColumns."""
        # Test both left and right prefixing
        # Test with no overlapping columns
        # Test with all columns overlapping

    def test_check_are_joinable_valid(self):
        """Test _checkAreJoinable with valid TSDFs."""
        # Same schema, same series IDs

    def test_check_are_joinable_invalid(self):
        """Test _checkAreJoinable with invalid TSDFs."""
        # Different schemas
        # Different series IDs
        # Different timestamp types
```

#### 2. BroadcastAsOfJoiner Tests
```python
# tests/test_broadcast_asof_joiner.py
class TestBroadcastAsOfJoiner:
    """Test broadcast join strategy."""

    def test_init_parameters(self):
        """Test initialization with various parameters."""
        # Test with SparkSession
        # Test range_join_bin_size parameter
        # Test prefix parameters

    def test_join_simple_timestamp(self):
        """Test join with simple timestamp column."""
        # Create test data with TimestampType
        # Verify join results
        # Check row count matches expected

    def test_join_composite_timestamp(self):
        """Test join with composite timestamp index (nanos)."""
        # Create test data with struct timestamp
        # Verify comparableExpr() handling
        # Check timezone consistency

    def test_join_with_series_ids(self):
        """Test join with multiple series IDs."""
        # Test with single series ID
        # Test with multiple series IDs
        # Test with no series IDs

    def test_lead_column_generation(self):
        """Test lead column creation and cleanup."""
        # Verify lead column is created
        # Verify lead column is dropped after join
        # Test with custom column names

    def test_range_join_bin_size_config(self):
        """Test Spark configuration setting."""
        # Verify config is set correctly
        # Test with different bin sizes

    def test_between_condition(self):
        """Test the between condition for joining."""
        # Test inclusive boundaries
        # Test with null values in lead column

    def test_comparable_expr_handling(self):
        """Test comparableExpr() for different index types."""
        # Test when comparableExpr returns list
        # Test when comparableExpr returns single expression
        # Test when comparableExpr returns empty list
        # Test fallback to column name

    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        # Empty left DataFrame
        # Empty right DataFrame
        # Both empty

    def test_timezone_handling(self):
        """Test timezone consistency (CRITICAL)."""
        # Test with different timezones
        # Test with naive timestamps
        # Test UTC conversion
```

#### 3. UnionSortFilterAsOfJoiner Tests
```python
# tests/test_union_sort_filter_joiner.py
class TestUnionSortFilterAsOfJoiner:
    """Test union-sort-filter join strategy."""

    def test_init_with_skip_nulls(self):
        """Test initialization with skipNulls parameter."""
        # Test True/False values

    def test_init_with_tolerance(self):
        """Test initialization with tolerance parameter."""
        # Test various tolerance values
        # Test None tolerance

    def test_append_null_columns(self):
        """Test _appendNullColumns method."""
        # Test adding single column
        # Test adding multiple columns
        # Test with empty column set

    def test_combine_method(self):
        """Test _combine method creates correct structure."""
        # Test union of DataFrames
        # Test combined timestamp column creation
        # Test row indicator field
        # Test CompositeTSIndex creation

    def test_combine_with_simple_index(self):
        """Test _combine with simple timestamp index."""
        # Verify coalesce logic
        # Verify struct column creation

    def test_combine_with_composite_index(self):
        """Test _combine with composite timestamp index."""
        # Verify component fields handling
        # Verify multiple comparable expressions

    def test_filter_last_right_row_skip_nulls_true(self):
        """Test _filterLastRightRow with skipNulls=True."""
        # Verify last() with ignoreNulls=True
        # Test with null values in data

    def test_filter_last_right_row_skip_nulls_false(self):
        """Test _filterLastRightRow with skipNulls=False."""
        # Verify struct wrapping logic
        # Test null preservation

    def test_row_indicator_logic(self):
        """Test left/right row indicators."""
        # Verify _LEFT_HAND_ROW_INDICATOR = 1
        # Verify _RIGHT_HAND_ROW_INDICATOR = -1
        # Test filtering based on indicators

    def test_window_functions(self):
        """Test allBeforeWindow usage."""
        # Test window partitioning
        # Test window ordering

    def test_tolerance_filter_implementation(self):
        """Test _toleranceFilter method."""
        # Test with valid tolerance
        # Test with zero tolerance
        # Test with None tolerance (no filtering)

    def test_join_complete_flow(self):
        """Test complete _join method flow."""
        # Test column differentiation
        # Test null column addition
        # Test combine operation
        # Test filtering
        # Test tolerance application

    def test_schema_preservation(self):
        """Test schema copying and preservation."""
        # Verify deep copy of schema
        # Test ts_schema consistency
```

#### 4. SkewAsOfJoiner Tests
```python
# tests/test_skew_asof_joiner.py
class TestSkewAsOfJoiner:
    """Test skew-aware join strategy."""

    def test_init_with_partition_val(self):
        """Test initialization with tsPartitionVal."""
        # Test various partition values
        # Test None value

    def test_init_with_fraction(self):
        """Test initialization with fraction parameter."""
        # Test various fraction values (0.0 to 1.0)

    def test_time_partitions_creation(self):
        """Test _applyTimePartitions method."""
        # Test partition boundaries
        # Test overlap handling
        # Test is_original flag

    def test_overlapped_data_removal(self):
        """Test removal of overlapped data."""
        # Verify is_original == 1 filter
        # Verify ts_partition column drop

    def test_skew_handling_large_partitions(self):
        """Test with heavily skewed data."""
        # Create data with 90% in one partition
        # Verify performance improvement

    def test_skew_handling_uniform_data(self):
        """Test with uniform distribution."""
        # Should behave like regular join
```

#### 5. Strategy Selection Tests
```python
# tests/test_strategy_selection.py
class TestStrategySelection:
    """Test choose_as_of_join_strategy function."""

    def test_broadcast_selection_small_data(self):
        """Test broadcast selection for small DataFrames."""
        # Mock get_bytes_from_plan to return < 30MB
        # Verify BroadcastAsOfJoiner selected

    def test_broadcast_selection_large_data(self):
        """Test broadcast not selected for large DataFrames."""
        # Mock get_bytes_from_plan to return > 30MB
        # Verify different strategy selected

    def test_skew_selection_with_partition_val(self):
        """Test skew strategy selection."""
        # Provide tsPartitionVal
        # Verify SkewAsOfJoiner selected

    def test_default_selection(self):
        """Test default strategy selection."""
        # No special conditions
        # Verify UnionSortFilterAsOfJoiner selected

    def test_sql_join_opt_flag(self):
        """Test sql_join_opt parameter effect."""
        # Test with True/False values

    def test_size_estimation_error_handling(self):
        """Test error handling in size estimation."""
        # Mock get_bytes_from_plan to raise exception
        # Verify fallback to default strategy

    def test_spark_plan_parsing(self):
        """Test get_spark_plan function."""
        # Test with valid DataFrame
        # Test plan string extraction

    def test_bytes_from_plan_parsing(self):
        """Test get_bytes_from_plan function."""
        # Test GiB parsing
        # Test MiB parsing
        # Test KiB parsing
        # Test bytes parsing
        # Test regex failure handling
```

#### 6. Helper Class Tests
```python
# tests/test_helper_classes.py
class TestAsOfJoinCompositeTSIndex:
    """Test _AsOfJoinCompositeTSIndex helper class."""

    def test_unit_property(self):
        """Test unit property returns None."""

    def test_range_expr_not_implemented(self):
        """Test rangeExpr raises NotImplementedError."""
        # Verify exception message

    def test_field_path_access(self):
        """Test fieldPath method for row indicator."""
        # Test accessing _DEFAULT_RIGHT_ROW_COLNAME
```

#### 7. Edge Cases and Error Tests
```python
# tests/test_edge_cases.py
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframes(self):
        """Test all strategies with empty DataFrames."""
        # Empty left, non-empty right
        # Non-empty left, empty right
        # Both empty

    def test_single_row_dataframes(self):
        """Test with single row in each DataFrame."""

    def test_duplicate_timestamps(self):
        """Test with duplicate timestamp values."""

    def test_null_timestamps(self):
        """Test with null values in timestamp columns."""

    def test_null_series_ids(self):
        """Test with null values in series ID columns."""

    def test_special_characters_in_columns(self):
        """Test with special characters in column names."""
        # Spaces, dots, underscores, etc.

    def test_very_long_column_names(self):
        """Test with extremely long column names."""

    def test_many_columns(self):
        """Test with hundreds of columns."""

    def test_many_series_ids(self):
        """Test with many series ID columns."""

    def test_extreme_tolerance_values(self):
        """Test with very large/small tolerance values."""
        # Integer overflow scenarios
        # Negative tolerance

    def test_mismatched_schemas(self):
        """Test error handling for incompatible schemas."""
        # Different timestamp types
        # Different series IDs
        # Different column types
```

#### 8. Integration Tests
```python
# tests/test_integration.py
class TestIntegration:
    """End-to-end integration tests."""

    def test_backward_compatibility(self):
        """Ensure existing code continues to work."""
        # Test with all existing test cases
        # Verify same results as current implementation

    def test_strategy_parameter_in_tsdf(self):
        """Test strategy parameter in TSDF.asofJoin."""
        # Test each strategy value
        # Test invalid strategy value

    def test_configuration_options(self):
        """Test Spark configuration integration."""
        # Test spark.tempo.asof.join.auto_optimize
        # Test spark.tempo.asof.join.broadcast_threshold
        # Test spark.tempo.asof.use_legacy

    def test_performance_benchmarks(self):
        """Benchmark against current implementation."""
        # Small data (< 1MB)
        # Medium data (1MB - 100MB)
        # Large data (> 100MB)
        # Skewed data
```

### Test Data Fixtures
```python
# tests/fixtures/asof_join_fixtures.py

@pytest.fixture
def simple_timestamp_data(spark):
    """Create simple timestamp test data."""
    return {
        "left": [...],
        "right": [...],
        "expected": [...]
    }

@pytest.fixture
def composite_timestamp_data(spark):
    """Create composite timestamp test data (nanos)."""
    return {
        "left": [...],
        "right": [...],
        "expected": [...]
    }

@pytest.fixture
def skewed_data(spark):
    """Create heavily skewed test data."""
    # 90% of data in one partition
    return {...}

@pytest.fixture
def null_heavy_data(spark):
    """Create data with many null values."""
    return {...}

@pytest.fixture
def timezone_test_data(spark):
    """Create data with various timezone scenarios."""
    return {...}
```

### Mock Testing Examples
```python
# tests/test_with_mocks.py
from unittest.mock import Mock, patch, MagicMock

class TestWithMocks:
    """Examples of mock usage for unit testing."""

    @patch('tempo.joins.strategies.SparkSession')
    def test_broadcast_joiner_with_mock_spark(self, mock_spark):
        """Test BroadcastAsOfJoiner with mocked SparkSession."""
        mock_spark_instance = Mock()
        mock_spark.builder.getOrCreate.return_value = mock_spark_instance

        joiner = BroadcastAsOfJoiner(mock_spark_instance)

        # Verify Spark config is set
        mock_spark_instance.conf.set.assert_called_with(
            "spark.databricks.optimizer.rangeJoin.binSize", "60"
        )

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_strategy_selection_with_mock_size(self, mock_get_bytes):
        """Test strategy selection with mocked size estimation."""
        # Mock small DataFrames
        mock_get_bytes.side_effect = [10 * 1024 * 1024, 20 * 1024 * 1024]  # 10MB, 20MB

        strategy = choose_as_of_join_strategy(
            left_tsdf, right_tsdf, sql_join_opt=True
        )

        assert isinstance(strategy, BroadcastAsOfJoiner)

        # Mock large DataFrames
        mock_get_bytes.side_effect = [100 * 1024 * 1024, 200 * 1024 * 1024]  # 100MB, 200MB

        strategy = choose_as_of_join_strategy(
            left_tsdf, right_tsdf, sql_join_opt=True
        )

        assert isinstance(strategy, UnionSortFilterAsOfJoiner)

    def test_tsdf_mock_for_unit_tests(self):
        """Test with mocked TSDF objects."""
        mock_left_tsdf = Mock(spec=TSDF)
        mock_left_tsdf.series_ids = ["id1", "id2"]
        mock_left_tsdf.ts_col = "timestamp"
        mock_left_tsdf.columns = ["timestamp", "id1", "id2", "value1"]
        mock_left_tsdf.ts_index = Mock()
        mock_left_tsdf.ts_index.comparableExpr.return_value = Mock()

        mock_right_tsdf = Mock(spec=TSDF)
        mock_right_tsdf.series_ids = ["id1", "id2"]
        mock_right_tsdf.ts_col = "timestamp"
        mock_right_tsdf.columns = ["timestamp", "id1", "id2", "value2"]

        joiner = UnionSortFilterAsOfJoiner()
        prefixable = joiner._prefixableColumns(mock_left_tsdf, mock_right_tsdf)

        assert prefixable == {"timestamp"}  # Only timestamp overlaps (series IDs excluded)
```

### Property-Based Testing
```python
# tests/test_property_based.py
from hypothesis import given, strategies as st, assume
from hypothesis.extra.spark import data_frames

class TestPropertyBased:
    """Property-based testing for as-of joins."""

    @given(
        num_rows_left=st.integers(min_value=0, max_value=1000),
        num_rows_right=st.integers(min_value=0, max_value=1000),
        tolerance=st.one_of(st.none(), st.integers(min_value=0, max_value=86400))
    )
    def test_join_row_count_properties(self, num_rows_left, num_rows_right, tolerance):
        """Test join result row count properties."""
        # Property: Result rows <= left rows (for left join semantics)
        # Property: If right is empty, result should equal left
        # Property: Tolerance should only reduce matches, not increase

    @given(
        prefix_left=st.text(min_size=0, max_size=10),
        prefix_right=st.text(min_size=0, max_size=10)
    )
    def test_prefix_properties(self, prefix_left, prefix_right):
        """Test column prefixing properties."""
        # Property: No column name collisions after prefixing
        # Property: Series IDs never get prefixed
        # Property: Empty prefix means no change

    @given(
        skipNulls=st.booleans(),
        null_percentage=st.floats(min_value=0, max_value=1)
    )
    def test_null_handling_properties(self, skipNulls, null_percentage):
        """Test null handling properties."""
        # Property: skipNulls=True means no nulls in matched values
        # Property: skipNulls=False preserves nulls

    @given(data_frames(
        columns=[
            st.column(name="timestamp", dtype=st.timestamps()),
            st.column(name="id", dtype=st.text()),
            st.column(name="value", dtype=st.floats())
        ]
    ))
    def test_schema_preservation(self, df):
        """Test schema preservation properties."""
        # Property: Output schema contains all left columns
        # Property: Timestamp column type is preserved
```

### Performance Test Specifications
```python
# tests/test_performance.py
import time
import pytest

class TestPerformance:
    """Performance benchmarks for as-of joins."""

    @pytest.mark.benchmark
    def test_broadcast_join_performance(self, benchmark):
        """Benchmark broadcast join strategy."""
        def run_join():
            joiner = BroadcastAsOfJoiner(spark)
            return joiner(left_tsdf, right_tsdf)

        result = benchmark(run_join)
        assert result.df.count() > 0

    @pytest.mark.benchmark
    @pytest.mark.parametrize("data_size", [1000, 10000, 100000, 1000000])
    def test_scalability(self, benchmark, data_size):
        """Test scalability with increasing data sizes."""
        # Generate data of specified size
        left_df = generate_test_data(data_size)
        right_df = generate_test_data(data_size // 2)

        def run_join():
            return left_tsdf.asofJoin(right_tsdf)

        result = benchmark(run_join)
        # Performance should scale sub-linearly

    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage for different strategies."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run join
        result = left_tsdf.asofJoin(right_tsdf, strategy="broadcast")
        result.df.count()  # Force evaluation

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        # Broadcast should use less memory than union for small data
        assert mem_after - mem_before < 100  # Less than 100MB increase
```

### Test Execution Matrix
```yaml
# tox.ini
[tox]
envlist = py{38,39,310,311}-spark{31,32,33}

[testenv]
deps =
    pytest>=7.0
    pytest-cov>=4.0
    pytest-mock>=3.10
    pytest-benchmark>=4.0
    hypothesis>=6.50
    hypothesis-spark>=0.1
    pyspark=={env:SPARK_VERSION:3.3.0}

commands =
    # Unit tests with coverage
    pytest tests/test_asof_joiner_base.py -v --cov=tempo.joins --cov-report=term-missing
    pytest tests/test_broadcast_asof_joiner.py -v --cov-append
    pytest tests/test_union_sort_filter_joiner.py -v --cov-append
    pytest tests/test_skew_asof_joiner.py -v --cov-append
    pytest tests/test_strategy_selection.py -v --cov-append
    pytest tests/test_helper_classes.py -v --cov-append
    pytest tests/test_edge_cases.py -v --cov-append
    pytest tests/test_integration.py -v --cov-append

    # Property-based tests
    pytest tests/test_property_based.py -v --hypothesis-show-statistics

    # Performance tests (optional)
    pytest tests/test_performance.py -v -m benchmark --benchmark-only

    # Generate coverage report
    coverage html
    coverage xml

[testenv:integration]
commands =
    pytest tests/test_integration.py -v --slow
```

### Coverage Metrics
```yaml
# .coveragerc
[run]
source = tempo/joins/
branch = True
parallel = True
omit =
    */tests/*
    */test_*.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstractmethod
    pass

precision = 2
skip_covered = False
show_missing = True
fail_under = 100

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

### Continuous Integration Coverage Check
```yaml
# .github/workflows/coverage.yml
name: Test Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e .[test]
        pip install coverage pytest-cov

    - name: Run tests with coverage
      run: |
        pytest --cov=tempo.joins --cov-report=xml --cov-report=term

    - name: Check coverage threshold
      run: |
        coverage report --fail-under=100

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
        verbose: true
```

## Migration Plan

### Phase 1: Foundation (Week 1-2)
- Set up joins package structure
- Implement base strategy classes
- Create initial tests

### Phase 2: Core Features (Week 3-4)
- Implement all join strategies
- Add automatic strategy selection
- Implement tolerance and null handling

### Phase 3: Optimization (Week 5)
- Complete skew handling
- Performance tuning
- Add monitoring and logging

### Phase 4: Integration (Week 6)
- Update TSDF class integration
- Ensure backward compatibility
- Documentation and examples

## Backward Compatibility

### API Compatibility
- Existing `asofJoin()` calls must work without changes
- New parameters should have sensible defaults
- Deprecation warnings for any breaking changes

### Behavior Compatibility
- Default behavior should match current implementation
- Strategy pattern should be transparent to users
- Results must be identical for same inputs

## Documentation Requirements

### User Documentation
- Update asofJoin method docstring
- Add examples for each new parameter
- Create performance tuning guide

### Developer Documentation
- Document strategy pattern architecture
- Explain strategy selection algorithm
- Provide extension guidelines for custom strategies

## Configuration Reference

### Spark Configuration Options
```properties
# Enable automatic strategy optimization
spark.tempo.asof.join.auto_optimize=true

# Broadcast join threshold in bytes
spark.tempo.asof.join.broadcast_threshold=31457280

# Skew detection threshold (variance ratio)
spark.tempo.asof.join.skew_threshold=2.0

# Default tolerance in seconds (0 = disabled)
spark.tempo.asof.join.default_tolerance=0
```

### Method Parameters
```python
def asofJoin(
    self,
    right_tsdf: TSDF,
    left_prefix: str = None,
    right_prefix: str = "right",
    tolerance: Optional[int] = None,
    tolerance_mode: str = "backward",
    skipNulls: bool = True,
    null_fill: Optional[Dict[str, Any]] = None,
    strategy: Optional[str] = None,  # "broadcast", "union_sort_filter", "skew", "auto"
    sql_join_opt: bool = True,  # Enable SQL optimization
) -> TSDF:
    """
    Enhanced as-of join with strategy selection and advanced options.
    """
```

## Success Criteria

1. **Performance**: 20-50% improvement for skewed datasets
2. **Compatibility**: Zero breaking changes for existing code
3. **Reliability**: All existing tests pass, 90%+ code coverage for new code
4. **Usability**: Clear documentation and intuitive API
5. **Monitoring**: Ability to track strategy selection and performance

## Known Issues and Resolutions

### From Experimental Branch Testing

1. **Timezone Handling in Composite Indexes**
   - **Issue**: Test failures in `as_of_join_tests.py` for nanos test case
   - **Root Cause**: Composite indexes with nanosecond precision have timezone offset issues
   - **Resolution**: Use `comparableExpr()` properly and ensure timezone consistency
   ```python
   # Fix: Ensure timezone-aware comparison
   if isinstance(ts_index, CompositeTSIndex):
       # Use the double_ts field for comparison (first element)
       comparable = ts_index.comparableExpr()[0]
   ```

2. **Join Semantics Inconsistency**
   - **Issue**: Broadcast join returns inner join results, union join returns left join results
   - **Root Cause**: Different implementations have different default behaviors
   - **Resolution**: Add `join_type` parameter to control behavior explicitly
   ```python
   def asofJoin(..., join_type: str = "left"):  # "left" or "inner"
   ```

3. **Incomplete Tolerance Implementation**
   - **Issue**: `_toleranceFilter()` not implemented in experimental branch
   - **Resolution**: Port existing working implementation from TSDF class

4. **SkewAsOfJoiner Not Implemented**
   - **Issue**: Method just calls parent class
   - **Resolution**: Implement using existing `__getTimePartitions()` logic

## Error Handling and Validation

### Input Validation
```python
def _checkAreJoinable(self, left: TSDF, right: TSDF) -> None:
    """
    Comprehensive validation for joinable TSDFs.
    """
    # Schema equivalence
    if left.ts_schema != right.ts_schema:
        raise ValueError(
            f"Timestamp schemas must match. "
            f"Left: {left.ts_schema}, Right: {right.ts_schema}"
        )

    # Series IDs compatibility
    if set(left.series_ids) != set(right.series_ids):
        raise ValueError(
            f"Series IDs must match. "
            f"Left: {left.series_ids}, Right: {right.series_ids}"
        )

    # Timestamp data type compatibility
    left_ts_type = left.df.schema[left.ts_col].dataType
    right_ts_type = right.df.schema[right.ts_col].dataType
    if left_ts_type != right_ts_type:
        raise ValueError(
            f"Timestamp column types must match. "
            f"Left: {left_ts_type}, Right: {right_ts_type}"
        )
```

### Strategy Selection Validation
```python
def choose_as_of_join_strategy(...) -> AsOfJoiner:
    try:
        # Size estimation with error handling
        if sql_join_opt:
            try:
                left_bytes = get_bytes_from_plan(left_tsdf.df, spark)
                right_bytes = get_bytes_from_plan(right_tsdf.df, spark)
            except Exception as e:
                logger.warning(f"Could not estimate DataFrame size: {e}")
                # Fall back to default strategy
                return UnionSortFilterAsOfJoiner(...)

        # ... rest of selection logic
    except Exception as e:
        logger.error(f"Strategy selection failed: {e}")
        # Always fall back to safe default
        return UnionSortFilterAsOfJoiner(...)
```

## Risks and Mitigations

### Risk 1: Breaking Existing Functionality
- **Mitigation**:
  - Comprehensive test coverage including all existing test cases
  - Feature flag: `spark.conf.set("spark.tempo.asof.use_legacy", "true")`
  - Gradual rollout with monitoring

### Risk 2: Performance Regression
- **Mitigation**:
  - Benchmark suite comparing all strategies
  - Performance monitoring in production
  - Quick disable switch via configuration

### Risk 3: Complex API
- **Mitigation**:
  - Keep existing API unchanged by default
  - Progressive disclosure of new features
  - Clear documentation with examples

### Risk 4: Timezone Issues with Composite Indexes
- **Mitigation**:
  - Thorough testing with different timezone configurations
  - Explicit timezone handling in composite index operations
  - Clear documentation of timezone requirements

## Next Steps

1. Review and approve this strategy document
2. Create detailed technical design for each component
3. Set up development environment and test data
4. Begin Phase 1 implementation
5. Regular progress reviews and adjustments

## Implementation Checklist

### Essential Fixes from Experimental Branch
- [ ] Fix timezone handling in composite indexes (BroadcastAsOfJoiner)
- [ ] Fix timezone handling in union sort filter (UnionSortFilterAsOfJoiner)
- [ ] Implement `_toleranceFilter()` method properly
- [ ] Complete `SkewAsOfJoiner._join()` implementation
- [ ] Add proper comparable expression handling
- [ ] Ensure consistent join semantics (left vs inner)

### New Functionality
- [ ] Create tempo/joins/strategies.py module
- [ ] Implement strategy pattern with AsOfJoiner base class
- [ ] Port and fix BroadcastAsOfJoiner
- [ ] Port and fix UnionSortFilterAsOfJoiner
- [ ] Complete SkewAsOfJoiner implementation
- [ ] Implement choose_as_of_join_strategy function
- [ ] Add strategy parameter to TSDF.asofJoin
- [ ] Create comprehensive test suite

### Integration Tasks
- [ ] Update TSDF.asofJoin to use strategy pattern
- [ ] Maintain backward compatibility
- [ ] Add configuration options
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Add performance benchmarks

## Summary

This document provides a comprehensive strategy for implementing enhanced as-of join features in Tempo. It addresses all known issues from the experimental `as_of_join_refactor` branch, including:

1. **Critical timezone handling bugs** in composite timestamp indexes that cause test failures
2. **Incomplete implementations** of tolerance filtering and skew handling
3. **Inconsistent join semantics** between different strategies
4. **Missing comparable expression handling** for nanosecond precision timestamps

The implementation follows a strategy pattern that allows:
- Flexible selection of join algorithms based on data characteristics
- Automatic optimization for broadcast joins when data is small
- Special handling for skewed data distributions
- Configurable tolerance windows for temporal proximity
- Consistent null handling across all strategies

All code examples, implementation details, and known issues from the experimental branch have been documented to ensure a successful implementation.