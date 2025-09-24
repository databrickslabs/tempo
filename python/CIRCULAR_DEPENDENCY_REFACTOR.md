# Circular Dependency Refactoring Documentation

## Summary
This document outlines the circular dependency fix implemented for the AS-OF join strategies and establishes the tuple-based approach `(DataFrame, TSSchema)` as the recommended pattern for data transformations in Tempo. This approach promotes loose coupling, better testability, and improved modularity.

## Completed Work: AS-OF Join Strategies

### Problem
- `strategies.py` imported `TSDF` to create and return TSDF objects
- `tsdf.py` imported strategies to use the join implementations
- This created a circular dependency requiring TYPE_CHECKING workarounds

### Solution Implemented
- Changed strategy methods to return `Tuple[DataFrame, TSSchema]` instead of `TSDF`
- TSDF.asofJoin now wraps the tuple results into TSDF objects
- Removed TYPE_CHECKING workarounds and string annotations

### Benefits Achieved
- Clean separation of concerns
- Strategies are now pure DataFrame transformers
- Better testability (can test without TSDF dependency)
- No runtime imports needed
- Improved modularity and reusability

## The Tuple Pattern: Best Practice for Tempo

### Recommended Approach: Return `(DataFrame, TSSchema)` Tuples

The tuple-based approach should be the standard for all data transformation operations in Tempo. This pattern separates the data (DataFrame) from its metadata (TSSchema), allowing for more flexible and maintainable code.

### Benefits of the Tuple Pattern

#### 1. **Loose Coupling**
- Components don't need to know about wrapper classes (TSDF, IntervalsDF)
- Transformations can work with raw DataFrames
- Reduced interdependencies between modules

#### 2. **Better Testability**
- Can test transformations without instantiating wrapper objects
- Easier to mock and stub in unit tests
- Can test DataFrame operations in isolation

#### 3. **Improved Reusability**
- Functions can be used in non-Tempo contexts
- Easy integration with other Spark-based libraries
- Transformations become composable building blocks

#### 4. **Performance Benefits**
- Avoid unnecessary object creation in intermediate steps
- Can chain transformations without wrapper overhead
- More efficient for multi-step pipelines

#### 5. **Cleaner Architecture**
- Clear separation between data and operations
- Follows functional programming principles
- Easier to understand data flow

### Downsides of Tight Coupling (Current Anti-Pattern)

#### 1. **Circular Dependencies**
- Classes that need each other create import cycles
- Requires workarounds like TYPE_CHECKING and string annotations
- Runtime imports make code harder to follow

#### 2. **Reduced Flexibility**
- Functions locked into specific return types
- Hard to compose with other libraries
- Difficult to extend or modify behavior

#### 3. **Testing Challenges**
- Need to set up entire object hierarchies for simple tests
- Mocking becomes complex with tightly coupled classes
- Integration tests required even for simple logic

#### 4. **Maintenance Burden**
- Changes ripple through tightly coupled systems
- Refactoring becomes risky and time-consuming
- Hard to understand component boundaries

#### 5. **Limited Reusability**
- Functions tied to specific wrapper classes
- Can't easily extract and reuse core logic
- Duplicated code across similar operations

## Future Refactoring Opportunities

### 1. External Processing Modules

#### tempo/stats.py
Several statistical functions create TSDF internally but could return tuples:
- `EMA()` - Exponential moving average calculation
- `make_ohlc_bars()` - OHLC bar creation
- Other statistical transformations

**Potential Refactor:**
```python
# Current
def EMA(tsdf: TSDF, colName: str, window: int = 30, exp_factor: float = 0.2) -> TSDF:
    # ... processing ...
    return TSDF(df, ts_schema=copy.deepcopy(tsdf.ts_schema))

# Proposed
def compute_ema(df: DataFrame, ts_schema: TSSchema, colName: str, window: int = 30, exp_factor: float = 0.2) -> Tuple[DataFrame, TSSchema]:
    # ... processing ...
    return df, copy.deepcopy(ts_schema)
```

#### tempo/interpol.py
Interpolation functions that create TSDF:
- `Interpolate()` - Main interpolation function

**Benefits:**
- Could be used independently of TSDF
- Easier to test interpolation logic
- Could be integrated into other DataFrame processing pipelines

#### tempo/resample.py
Resampling operations that return TSDF:
- `resample()` - Time series resampling

**Benefits:**
- Resampling logic could be reused in other contexts
- Better separation of resampling algorithm from TSDF wrapper

### 2. Internal TSDF Methods

#### Private Helper Methods
Methods that could benefit from tuple approach:
- `__withTransformedDF()` - Internal DataFrame transformation wrapper
- `__combineTSDF()` - Combines two TSDFs
- `__getTimePartitions()` - Creates time-based partitions

**Potential Refactor:**
These could be refactored to work with DataFrames and schemas directly, making the code more modular.

### 3. AS-OF Join Module (tempo/as_of_join.py)

This module appears to be a duplicate/older version of the join implementation. It should either:
- Be removed if deprecated
- Be refactored similarly to strategies.py if still needed

**Files to consider:**
- `tempo/as_of_join.py` - Contains duplicate AsOfJoiner implementations
- `tempo/tsdf_asof_join_patch.py` - Appears to be a patch file

### 4. Intervals Module (tempo/intervals/core/intervals_df.py)

The IntervalsDF class is similar to TSDF - it wraps DataFrames with interval-specific functionality. Opportunities for refactoring:

#### Current State
- `make_disjoint()` returns IntervalsDF
- `union()` and `unionByName()` return IntervalsDF
- Static factory methods like `fromStackedMetrics()` create IntervalsDF

**Potential Refactor:**
```python
# Current
def make_disjoint(self) -> "IntervalsDF":
    # ... processing ...
    return IntervalsDF(result_df, self.start_ts, self.end_ts, self.series)

# Proposed - return components
def compute_disjoint_intervals(df: DataFrame, start_ts: str, end_ts: str, series: List[str]) -> Tuple[DataFrame, Dict[str, Any]]:
    # ... processing ...
    metadata = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "series": series,
        "metrics": computed_metrics
    }
    return result_df, metadata
```

#### Benefits
- Interval operations could be used independently
- Better composability with other DataFrame operations
- Easier testing of interval logic without IntervalsDF wrapper
- Could integrate with TSDF without circular dependencies

### 5. TSDF-IntervalsDF Integration

The TSDF class currently imports IntervalsDF for aggregation operations:
- `TSDF.group_by()` returns IntervalsDF in certain cases
- `TSDF.applyInPandasByWindow()` returns IntervalsDF

**Integration Points:**
```python
# In tsdf.py
return IntervalsDF.fromNestedBoundariesDF(agged_df, "window")
```

This creates a dependency from TSDF to IntervalsDF. Using the tuple approach could decouple these modules.

### 6. Factory Functions and Builders

#### Opportunities for Factory Pattern
Instead of having functions that directly return TSDF, create factory functions that return components:

```python
# Current approach
def create_time_series(data: List, ts_col: str) -> TSDF:
    df = spark.createDataFrame(data)
    return TSDF(df, ts_col=ts_col)

# Proposed approach
def create_time_series_components(data: List, ts_col: str) -> Tuple[DataFrame, TSSchema]:
    df = spark.createDataFrame(data)
    schema = TSSchema.fromDFSchema(df.schema, ts_col)
    return df, schema

# Usage
df, schema = create_time_series_components(data, "timestamp")
tsdf = TSDF(df, ts_schema=schema)
```

## Implementation Priority

### High Priority (Immediate Benefits)
1. **Remove duplicate as_of_join.py** - Cleanup and consistency
2. **Refactor IntervalsDF core operations** - Heavy cross-dependency with TSDF
3. **Refactor stats.py functions** - Heavily used, would benefit from modularity

### Medium Priority (Good Value)
1. **TSDF-IntervalsDF integration points** - Decouple the two main data structures
2. **interpol.py refactoring** - Clean separation of interpolation logic
3. **resample.py refactoring** - Reusable resampling algorithms

### Low Priority (Nice to Have)
1. **Internal TSDF helper methods** - Less urgent, mostly internal
2. **Factory pattern adoption** - Can be done gradually
3. **IntervalsDF factory methods** - Can be refactored as needed

## Refactoring Guidelines

When refactoring to use the tuple approach:

1. **Identify the core logic** - Separate DataFrame transformation from TSDF wrapping
2. **Extract pure functions** - Create functions that work with DataFrames and return tuples
3. **Keep TSDF methods as wrappers** - TSDF methods should call pure functions and wrap results
4. **Maintain backward compatibility** - Keep existing APIs, refactor internals
5. **Add comprehensive tests** - Test both pure functions and TSDF wrappers

## Benefits of Widespread Adoption

1. **Modularity** - Components can be used independently
2. **Testability** - Pure functions are easier to test
3. **Performance** - Avoid unnecessary TSDF creation in intermediate steps
4. **Flexibility** - Users can work with DataFrames directly when needed
5. **Maintainability** - Cleaner separation of concerns

## Migration Strategy

1. **Phase 1**: Complete AS-OF join refactoring (DONE)
2. **Phase 2**: Remove duplicate code (as_of_join.py)
3. **Phase 3**: Refactor external modules (stats, interpol, resample)
4. **Phase 4**: Gradually refactor internal methods as needed

## Testing Strategy

For each refactored module:
1. Create tests for pure functions (DataFrame -> DataFrame)
2. Verify TSDF wrapper compatibility
3. Add integration tests
4. Benchmark performance (should be same or better)

## Implementation Example

### Before (Tightly Coupled)
```python
class AsOfJoiner:
    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        # Circular dependency - needs to import TSDF
        from tempo.tsdf import TSDF
        result_df = perform_join(left.df, right.df)
        return TSDF(result_df, ts_schema=left.ts_schema)
```

### After (Loosely Coupled)
```python
class AsOfJoiner:
    def _join(self, left, right) -> Tuple[DataFrame, TSSchema]:
        # No dependency on TSDF - pure transformation
        result_df = perform_join(left.df, right.df)
        return result_df, left.ts_schema

# TSDF wraps the result
class TSDF:
    def asofJoin(self, right: TSDF) -> TSDF:
        df, schema = joiner(self, right)
        return TSDF(df, ts_schema=schema)
```

## Conclusion

The tuple approach `(DataFrame, TSSchema)` successfully eliminated circular dependencies in the AS-OF join implementation and should be adopted as the standard pattern throughout Tempo. This approach:

1. **Promotes loose coupling** between components
2. **Improves testability** by allowing isolated unit tests
3. **Increases reusability** of transformation functions
4. **Simplifies maintenance** through clearer boundaries
5. **Enhances performance** by reducing object creation overhead

The refactoring should be done incrementally, starting with high-value targets like the IntervalsDF and stats modules. Each refactoring should maintain backward compatibility while internally adopting the tuple pattern for better architecture.