import pyspark.sql.functions as f
from pyspark.sql.window import Window
import tempo.resample as rs
import tempo.io as tio

class TSDF:

  def __init__(self, df, ts_col="event_ts", partition_cols=None, sequence_col = None):
    """
    Constructor
    :param df:
    :param ts_col:
    :param partitionCols:
    :sequence_col every tsdf allows for a tie-breaker secondary sort key
    """
    self.ts_col = self.__validated_column(df, ts_col)
    self.partitionCols = [] if partition_cols is None else self.__validated_columns(df, partition_cols)

    self.df = df
    self.sequence_col = '' if sequence_col is None else sequence_col
    """
    Make sure DF is ordered by its respective ts_col and partition columns.
    """
  ##
  ## Helper functions
  ##

  def __validated_column(self,df,colname):
    if type(colname) != str:
      raise TypeError(f"Column names must be of type str; found {type(colname)} instead!")
    if colname.lower() not in [col.lower() for col in df.columns]:
      raise ValueError(f"Column {colname} not found in Dataframe")
    return colname

  def __validated_columns(self,df,colnames):
    # if provided a string, treat it as a single column
    if type(colnames) == str:
      colnames = [ colnames ]
    # otherwise we really should have a list or None
    if colnames is None:
      colnames = []
    elif type(colnames) != list:
      raise TypeError(f"Columns must be of type list, str, or None; found {type(colnames)} instead!")
    # validate each column
    for col in colnames:
      self.__validated_column(df,col)
    return colnames

  def __checkPartitionCols(self,tsdf_right):
    for left_col, right_col in zip(self.partitionCols, tsdf_right.partitionCols):
        if left_col != right_col:
            raise ValueError("left and right dataframe partition columns should have same name in same order")

  def __validateTsColMatch(self, right_tsdf):
      left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
      right_ts_datatype = right_tsdf.df.select(self.ts_col).dtypes[0][1]
      if left_ts_datatype != right_ts_datatype:
          raise ValueError("left and right dataframe timestamp index columns should have same type")

  def __addPrefixToColumns(self,col_list,prefix):
    """
    Add prefix to all specified columns.
    """
    from functools import reduce

    df = reduce(lambda df, idx: df.withColumnRenamed(col_list[idx], '_'.join([prefix,col_list[idx]])),
                range(len(col_list)), self.df)

    ts_col = '_'.join([prefix, self.ts_col])
    seq_col = '_'.join([prefix, self.sequence_col]) if self.sequence_col else self.sequence_col
    return TSDF(df, ts_col, self.partitionCols, sequence_col = seq_col)

  def __addColumnsFromOtherDF(self, other_cols):
    """
    Add columns from some other DF as lit(None), as pre-step before union.
    """
    from functools import reduce
    new_df = reduce(lambda df, idx: df.withColumn(other_cols[idx], f.lit(None)), range(len(other_cols)), self.df)

    return TSDF(new_df, self.ts_col, self.partitionCols)

  def __combineTSDF(self, ts_df_right, combined_ts_col):
    combined_df = (self.df
                   .unionByName(ts_df_right.df)
                   .withColumn(combined_ts_col,f.coalesce(self.ts_col, ts_df_right.ts_col)))

    return TSDF(combined_df, combined_ts_col, self.partitionCols)

  def __getLastRightRow(self, left_ts_col, right_cols, sequence_col, tsPartitionVal):
    from functools import reduce
    """Get last right value of each right column (inc. right timestamp) for each self.ts_col value
    
    self.ts_col, which is the combined time-stamp column of both left and right dataframe, is dropped at the end
    since it is no longer used in subsequent methods.
    """
    ptntl_sort_keys = [self.ts_col, sequence_col]
    sort_keys = [f.col(col_name) for col_name in ptntl_sort_keys if col_name != '']
    sort_keys.append('rec_ind')

    window_spec = Window.partitionBy(self.partitionCols).orderBy(sort_keys).rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # splitting off the condition as we want different columns in the reduce if we are implementing the skew AS OF join
    if tsPartitionVal is None:
        df = reduce(lambda df, idx: df.withColumn(right_cols[idx], f.last(right_cols[idx], True).over(window_spec)),
                     range(len(right_cols)), self.df)
    else:
        df = reduce(
            lambda df, idx: df.withColumn(right_cols[idx], f.last(right_cols[idx], True).over(window_spec)).withColumn(
                'non_null_ct' + right_cols[idx], f.count(right_cols[idx]).over(window_spec)),
            range(len(right_cols)), self.df)

    df = (df.filter(f.col(left_ts_col).isNotNull()).drop(self.ts_col)).drop('rec_ind')

    # remove the null_ct stats used to record missing values in partitioned as of join
    if tsPartitionVal is not None:
      for column in df.columns:
        if (column.startswith("non_null")):
          any_blank_vals = (df.agg({column: 'min'}).collect()[0][0] == 0)
          newCol = column.replace("non_null_ct", "")
          if any_blank_vals:
            print("Column " + newCol + " had no values within the lookback window. Consider using a larger window to avoid missing values. If this is the first record in the data frame, this warning can be ignored.")
          df = df.drop(column)


    return TSDF(df, left_ts_col, self.partitionCols)

  def __getTimePartitions(self, tsPartitionVal, fraction=0.1):
    """
    Create time-partitions for our data-set. We put our time-stamps into brackets of <tsPartitionVal>. Timestamps
    are rounded down to the nearest <tsPartitionVal> seconds.

    We cast our timestamp column to double instead of using f.unix_timestamp, since it provides more precision.
    
    Additionally, we make these partitions overlapping by adding a remainder df. This way when calculating the
    last right timestamp we will not end up with nulls for the first left timestamp in each partition.

    TODO: change ts_partition to accomodate for higher precision than seconds.
    """
    partition_df = (
        self.df
        .withColumn("ts_col_double", f.col(self.ts_col).cast("double")) # double is preferred over unix_timestamp
        .withColumn('ts_partition',f.lit(tsPartitionVal) * (f.col("ts_col_double") / f.lit(tsPartitionVal)).cast('integer'))
        .withColumn("partition_remainder",(f.col("ts_col_double") - f.col("ts_partition"))/f.lit(tsPartitionVal))
        .withColumn("is_original", f.lit(1))).cache() # cache it because it's used twice.

    # add [1 - fraction] of previous time partition to the next partition.
    remainder_df = (
      partition_df.filter(f.col("partition_remainder") >= f.lit(1 - fraction))
      .withColumn("ts_partition", f.col("ts_partition") + f.lit(tsPartitionVal))
      .withColumn("is_original",f.lit(0)))

    df = partition_df.union(remainder_df).drop("partition_remainder","ts_col_double")
    return TSDF(df, self.ts_col, self.partitionCols + ['ts_partition'])

  def describe(self):
    """
         Describe a TSDF object using a global summary across all time series (anywhere from 10 to millions) as well as the standard Spark data frame stats. Missing vals
         Summary
         global - unique time series based on partition columns, min/max times, granularity - lowest precision in the time series timestamp column
         count / mean / stddev / min / max - standard Spark data frame describe() output
         missing_vals_pct - percentage (from 0 to 100) of missing values.
    """
    #extract the double version of the timestamp column to summarize
    double_ts_col = self.ts_col + "_dbl"

    this_df = self.df.withColumn(double_ts_col, f.col(self.ts_col).cast("double"))

    #summary missing value percentages
    missing_vals = this_df.select(
          [(100 * f.count(f.when(f.col(c[0]).isNull(), c[0])) / f.count(f.lit(1))).alias(c[0]) for c in this_df.dtypes if
           c[1] != 'timestamp']).select(f.lit('missing_vals_pct').alias("summary"), "*")


    # describe stats
    desc_stats = this_df.describe().union(missing_vals)
    unique_ts = this_df.select(*self.partitionCols).distinct().count()

    max_ts = this_df.select(f.max(f.col(self.ts_col)).alias("max_ts")).collect()[0][0]
    min_ts = this_df.select(f.min(f.col(self.ts_col)).alias("max_ts")).collect()[0][0]
    gran = this_df.selectExpr("""min(case when {0} - cast({0} as integer) > 0 then '1-millis'
                  when {0} % 60 != 0 then '2-seconds'
                  when {0} % 3600 != 0 then '3-minutes'
                  when {0} % 86400 != 0 then '4-hours' 
                  else '5-days' end) granularity""".format(double_ts_col)).collect()[0][0][2:]

    non_summary_cols = [c for c in desc_stats.columns if c != 'summary']


    desc_stats = desc_stats.select(f.col("summary"), f.lit(" ").alias("unique_ts_count"), f.lit(" ").alias("min_ts"),
                                   f.lit(" ").alias("max_ts"), f.lit(" ").alias("granularity"), *non_summary_cols)

    # add in single record with global summary attributes and the previously computed missing value and Spark data frame describe stats
    global_smry_rec = desc_stats.limit(1).select(f.lit('global').alias("summary"),f.lit(unique_ts).alias("unique_ts_count"), f.lit(min_ts).alias("min_ts"), f.lit(max_ts).alias("max_ts"), f.lit(gran).alias("granularity"), *[f.lit(" ").alias(c) for c in non_summary_cols])

    full_smry = global_smry_rec.union(desc_stats)

    try:
        dbutils.fs.ls("/")
        return(full_smry)
    except:
        return(full_smry)
        pass

  def asofJoin(self, right_tsdf, left_prefix=None, right_prefix="right", tsPartitionVal=None, fraction=0.5):
    """
    Performs an as-of join between two time-series. If a tsPartitionVal is specified, it will do this partitioned by
    time brackets, which can help alleviate skew.

    NOTE: partition cols have to be the same for both Dataframes.
    Parameters
    :param right_tsdf - right-hand data frame containing columns to merge in
    :param left_prefix - optional prefix for base data frame
    :param right_prefix - optional prefix for right-hand data frame
    :param tsPartitionVal - value to break up each partition into time brackets
    :param fraction - overlap fraction
    """

    if (tsPartitionVal is not None):
      print("WARNING: You are using the skew version of the AS OF join. This may result in null values if there are any values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum lookback, trading off performance and potential blank AS OF values for sparse keys")

    # Check whether partition columns have same name in both dataframes
    self.__checkPartitionCols(right_tsdf)

    # prefix non-partition columns, to avoid duplicated columns.
    left_df = self.df
    right_df = right_tsdf.df


    # validate timestamp datatypes match
    self.__validateTsColMatch(right_tsdf)

    orig_left_col_diff = list(set(left_df.columns).difference(set(self.partitionCols)))
    orig_right_col_diff = list(set(right_df.columns).difference(set(self.partitionCols)))

    left_tsdf = ((self.__addPrefixToColumns([self.ts_col] + orig_left_col_diff, left_prefix))
                 if left_prefix is not None else self)
    right_tsdf = right_tsdf.__addPrefixToColumns([right_tsdf.ts_col] + orig_right_col_diff, right_prefix)

    left_nonpartition_cols = list(set(left_tsdf.df.columns).difference(set(self.partitionCols)))
    right_nonpartition_cols = list(set(right_tsdf.df.columns).difference(set(self.partitionCols)))

    # For both dataframes get all non-partition columns (including ts_col)
    left_columns = [left_tsdf.ts_col] + left_nonpartition_cols
    right_columns = [right_tsdf.ts_col] + right_nonpartition_cols

    # Union both dataframes, and create a combined TS column
    combined_ts_col = "combined_ts"
    combined_df = (left_tsdf
                   .__addColumnsFromOtherDF(right_columns)
                   .__combineTSDF(right_tsdf.__addColumnsFromOtherDF(left_columns),
                                  combined_ts_col))
    combined_df.df = combined_df.df.withColumn("rec_ind", f.when(f.col(left_tsdf.ts_col).isNotNull(), 1).otherwise(-1))

    # perform asof join.
    if tsPartitionVal is None:
        asofDF = combined_df.__getLastRightRow(left_tsdf.ts_col, right_columns, right_tsdf.sequence_col, tsPartitionVal)
    else:
        tsPartitionDF = combined_df.__getTimePartitions(tsPartitionVal, fraction=fraction)
        asofDF = tsPartitionDF.__getLastRightRow(left_tsdf.ts_col, right_columns, right_tsdf.sequence_col, tsPartitionVal)

        # Get rid of overlapped data and the extra columns generated from timePartitions
        df = asofDF.df.filter(f.col("is_original") == 1).drop("ts_partition","is_original")

        asofDF = TSDF(df, asofDF.ts_col, combined_df.partitionCols)

    return asofDF


  def __baseWindow(self):
    # add all sort keys - time is first, unique sequence number breaks the tie

    ptntl_sort_keys = [self.ts_col, self.sequence_col]
    sort_keys = [f.col(col_name).cast("long") for col_name in ptntl_sort_keys if col_name != '']

    w = Window().orderBy(sort_keys)
    if self.partitionCols:
      w = w.partitionBy([f.col(elem) for elem in self.partitionCols])
    return w


  def __rangeBetweenWindow(self, range_from, range_to):
    return self.__baseWindow().rangeBetween(range_from, range_to)


  def __rowsBetweenWindow(self, rows_from, rows_to):
    return self.__baseWindow().rowsBetween(rows_from, rows_to)


  def withPartitionCols(self, partitionCols):
    """
    Sets certain columns of the TSDF as partition columns. Partition columns are those that differentiate distinct timeseries
    from each other.
    :param partitionCols: a list of columns used to partition distinct timeseries
    :return: a TSDF object with the given partition columns
    """
    return TSDF(self.df, self.ts_col, partitionCols)
  
  def vwap(self, frequency='m',volume_col = "volume", price_col = "price"):
        # set pre_vwap as self or enrich with the frequency
        pre_vwap = self.df
        if frequency == 'm':
            pre_vwap = self.df.withColumn("time_group", f.concat(f.lpad(f.hour(f.col(self.ts_col)), 2, '0'), f.lit(':'),
                                                               f.lpad(f.minute(f.col(self.ts_col)), 2, '0')))
        elif frequency == 'H':
            pre_vwap = self.df.withColumn("time_group", f.concat(f.lpad(f.hour(f.col(self.ts_col)), 2, '0')))
        elif frequency == 'D':
            pre_vwap = self.df.withColumn("time_group", f.concat(f.lpad(f.day(f.col(self.ts_col)), 2, '0')))

        group_cols = ['time_group']
        if self.partitionCols:
          group_cols.extend(self.partitionCols)
        vwapped = ( pre_vwap.withColumn("dllr_value", f.col(price_col) * f.col(volume_col))
                            .groupby(group_cols)
                            .agg( sum('dllr_value').alias("dllr_value"),
                                  sum(volume_col).alias(volume_col),
                                  max(price_col).alias("_".join(["max",price_col])) )
                            .withColumn("vwap", f.col("dllr_value") / f.col(volume_col)) )

        return TSDF( vwapped, self.ts_col, self.partitionCols )
  
  def EMA(self,colName,window=30,exp_factor = 0.2):
    """
    Constructs an approximate EMA in the fashion of:
    EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
    TODO: replace case when statement with coalesce
    TODO: add in time partitions functionality (what is the overlap fraction?)
    """

    emaColName = "_".join(["EMA",colName])
    df = self.df.withColumn(emaColName,f.lit(0)).orderBy(self.ts_col)
    w = self.__baseWindow()
    # Generate all the lag columns:
    for i in range(window):
      lagColName = "_".join(["lag",colName,str(i)])
      weight = exp_factor * (1 - exp_factor)**i
      df = df.withColumn(lagColName, weight * f.lag(f.col(colName),i).over(w))
      df = df.withColumn(emaColName, f.col(emaColName) + f.when(
          f.col(lagColName).isNull(),f.lit(0)).otherwise(f.col(lagColName))).drop(lagColName)
      # Nulls are currently removed
      
    return TSDF(df, self.ts_col, self.partitionCols)

  def withLookbackFeatures(self,
                           featureCols,
                           lookbackWindowSize,
                           exactSize=True,
                           featureColName="features"):
      """
      Creates a 2-D feature tensor suitable for training an ML model to predict current values from the history of
      some set of features. This function creates a new column containing, for each observation, a 2-D array of the values
      of some number of other columns over a trailing "lookback" window from the previous observation up to some maximum
      number of past observations.

      :param featureCols: the names of one or more feature columns to be aggregated into the feature column
      :param lookbackWindowSize: The size of lookback window (in terms of past observations). Must be an integer >= 1
      :param exactSize: If True (the default), then the resulting DataFrame will only include observations where the
        generated feature column contains arrays of length lookbackWindowSize. This implies that it will truncate
        observations that occurred less than lookbackWindowSize from the start of the timeseries. If False, no truncation
        occurs, and the column may contain arrays less than lookbackWindowSize in length.
      :param featureColName: The name of the feature column to be generated. Defaults to "features"
      :return: a DataFrame with a feature column named featureColName containing the lookback feature tensor
      """
      # first, join all featureCols into a single array column
      tempArrayColName = "__TempArrayCol"
      feat_array_tsdf = self.df.withColumn(tempArrayColName, f.array(featureCols))

      # construct a lookback array
      lookback_win = self.__rowsBetweenWindow(-lookbackWindowSize, -1)
      lookback_tsdf = (feat_array_tsdf.withColumn(featureColName,
                                                  f.collect_list(f.col(tempArrayColName)).over(lookback_win))
                                      .drop(tempArrayColName))

      # make sure only windows of exact size are allowed
      if exactSize:
          return lookback_tsdf.where(f.size(featureColName) == lookbackWindowSize)

      return TSDF( lookback_tsdf, self.ts_col, self.partitionCols )

  def withRangeStats(self, type='range', colsToSummarize=[], rangeBackWindowSecs=1000):
          """
          Create a wider set of stats based on all numeric columns by default
          Users can choose which columns they want to summarize also. These stats are:
          mean/count/min/max/sum/std deviation/zscore
          :param type - this is created in case we want to extend these stats to lookback over a fixed number of rows instead of ranging over column values
          :param colsToSummarize - list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
          :param rangeBackWindowSecs - lookback this many seconds in time to summarize all stats. Note this will look back from the floor of the base event timestamp (as opposed to the exact time since we cast to long)
          Assumptions:
               1. The features are summarized over a rolling window that ranges back
               2. The range back window can be specified by the user
               3. Sequence numbers are not yet supported for the sort
               4. There is a cast to long from timestamp so microseconds or more likely breaks down - this could be more easily handled with a string timestamp or sorting the timestamp itself. If using a 'rows preceding' window, this wouldn't be a problem
           """

          # identify columns to summarize if not provided
          # these should include all numeric columns that
          # are not the timestamp column and not any of the partition columns
          if not colsToSummarize:
            # columns we should never summarize
            prohibited_cols = [ self.ts_col.lower() ]
            if self.partitionCols:
              prohibited_cols.extend([ pc.lower() for pc in self.partitionCols])
            # types that can be summarized
            summarizable_types = ['int', 'bigint', 'float', 'double']
            # filter columns to find summarizable columns
            colsToSummarize = [datatype[0] for datatype in self.df.dtypes if
                                ((datatype[1] in summarizable_types) and
                                 (datatype[0].lower() not in prohibited_cols))]

          # build window
          w = self.__rangeBetweenWindow(-1 * rangeBackWindowSecs, 0)

          # compute column summaries
          selectedCols = self.df.columns
          derivedCols = []
          for metric in colsToSummarize:
              selectedCols.append(f.mean(metric).over(w).alias('mean_' + metric))
              selectedCols.append(f.count(metric).over(w).alias('count_' + metric))
              selectedCols.append(f.min(metric).over(w).alias('min_' + metric))
              selectedCols.append(f.max(metric).over(w).alias('max_' + metric))
              selectedCols.append(f.sum(metric).over(w).alias('sum_' + metric))
              selectedCols.append(f.stddev(metric).over(w).alias('stddev_' + metric))
              derivedCols.append(
                      ((f.col(metric) - f.col('mean_' + metric)) / f.col('stddev_' + metric)).alias("zscore_" + metric))
          selected_df = self.df.select(*selectedCols)
          summary_df = selected_df.select(*selected_df.columns, *derivedCols)

          return TSDF(summary_df, self.ts_col, self.partitionCols)

  def write(self, spark, tabName, optimizationCols = None):
    tio.write(self, spark, tabName, optimizationCols)

  def resample(self, freq, func=None, metricCols = None, prefix=None):
    """
    function to upsample based on frequency and aggregate function similar to pandas
    :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
    :param func: function used to aggregate input
    :return: TSDF object with sample data using aggregate function
    """
    rs.validateFuncExists(func)
    enriched_tsdf = rs.aggregate(self, freq, func, metricCols, prefix)
    return(enriched_tsdf)

  def calc_bars(tsdf, freq, func = None, metricCols = None):

      resample_open = tsdf.resample(freq=freq, func='floor', metricCols = metricCols, prefix='open')
      resample_low = tsdf.resample(freq=freq, func='min', metricCols = metricCols, prefix='low')
      resample_high = tsdf.resample(freq=freq, func='max', metricCols = metricCols, prefix='high')
      resample_close = tsdf.resample(freq=freq, func='ceil', metricCols = metricCols, prefix='close')

      join_cols = resample_open.partitionCols + [resample_open.ts_col]
      bars = resample_open.df.join(resample_high.df, join_cols).join(resample_low.df, join_cols).join(resample_close.df, join_cols)
      non_part_cols = set(set(bars.columns) - set(resample_open.partitionCols)) - set([resample_open.ts_col])
      sel_and_sort = resample_open.partitionCols + [resample_open.ts_col] + sorted(non_part_cols)
      bars = bars.select(sel_and_sort)

      return(TSDF(bars, resample_open.ts_col, resample_open.partitionCols))