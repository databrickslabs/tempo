import pyspark.sql.functions as fn

from pyspark.sql.functions import *
from pyspark.sql.window import Window

class TSDF:


  def __init__(self, df, ts_col = "EVENT_TS", seq_nb_col = '', key=''):
        """"
        @param key - TSDF includes an option for a key so we can always join narrow key datasets back to the larger fact table
        @param seq_nb_col - TSDF includes an option for a sequence number for sorting purposes - especially useful for AS OF joins which will require this
        """
        self.df = df
        self.ts_col = ts_col
        self.seq_nb_col = seq_nb_col
        self.key = key
        
  def __createTimeSeriesDF(self, tsdf, ts_select_cols, seq_select_col, fillLeft = True, partitionCols = []):
    df = tsdf.df

    left_ts_val, right_ts_val = (col(self.ts_col), lit(None)) if fillLeft else (lit(None),col(self.ts_col))
    left_seq_val, right_seq_val = (col(seq_select_col), lit(None)) if fillLeft else (lit(None), col(seq_select_col))

    left_seq_val, right_seq_val = (lit(1), lit(1)) if len(seq_select_col) == 0 else (left_seq_val, right_seq_val)

    selectedCols = ts_select_cols + partitionCols
    selectedCols = selectedCols + [self.key] if len(self.key) > 0 else selectedCols
    selectedCols = selectedCols + [seq_select_col] if len(seq_select_col) > 0 else selectedCols


    res = df.withColumn(ts_select_cols[0],left_ts_val) \
            .withColumn(ts_select_cols[1], right_ts_val) \
            .withColumn(ts_select_cols[2], col(self.ts_col).cast("double")) \
            .select(selectedCols)

    full_cols = ts_select_cols + partitionCols
    full_cols = full_cols + [self.key] if len(self.key) > 0  else full_cols
    full_cols = full_cols + [seq_select_col] if len(seq_select_col) > 0 else full_cols

    res = res.select(full_cols)

    return res
  
  def __getUnionDF(self, tsdf_right, ts_select_cols, seq_select_cols, partitionCols):
    df_right = tsdf_right.df
    seq_select_col = tsdf_right.seq_nb_col
    df_left = self.__createTimeSeriesDF(self, ts_select_cols, seq_select_col, partitionCols=partitionCols)
    df_right = self.__createTimeSeriesDF(tsdf_right, ts_select_cols, seq_select_col,
                                       fillLeft = False, partitionCols = partitionCols) # df_right ts col should have same ts


    #  column name, else error
    return df_left.select(sorted(df_left.columns)).union(df_right.select(sorted(df_right.columns))).withColumn("is_original",lit(1))
    
  def __getTimePartitions(self,UnionDF, ts_select_cols, seq_select_cols, tsPartitionVal, fraction = 0.5):
    """
    Create time-partitions for our data-set. We put our time-stamps into brackets of <tsPartitionVal>. Timestamps
    are rounded down to the nearest <tsPartitionVal> seconds.
    
    Additionally, we make these partitions overlapping by adding a remainder df. This way when calculating the
    last right timestamp we will not end up with nulls for the first left timestamp in each partition.
    """
    partition_df = (
        UnionDF
        .withColumn('ts_partition',lit(tsPartitionVal) * (col(ts_select_cols[2]) / lit(tsPartitionVal)).cast('integer'))
        .withColumn("partition_remainder",(col(ts_select_cols[2]) - col("ts_partition"))/lit(tsPartitionVal))
        .withColumn("is_original",lit(1))).cache() # cache it because it's used twice.
    
    # TODO: Is there a smarter way of figuring out the remainders? We only really need to add max(ts_right) to the next partition,
    # but this induces an extra groupBy. 
    remainder_df = (
      partition_df.filter(col("partition_remainder") >= lit(1 - fraction))
      .withColumn("ts_partition", col("ts_partition") + lit(tsPartitionVal)) # add [1 - fraction] of previous time partition to the next.
      .withColumn("is_original",lit(0)))
    return partition_df.union(remainder_df)
  
  def __getLastRightTs(self,UnionDF, ts_select_cols, seq_select_col, partitionCols = []):
    """Calculates the last observed timestamp of the right dataframe for each timestamp in the left dataframe"""
    has_seq_nb = (len(seq_select_col) > 0)
    windowSpec = Window.partitionBy(partitionCols).orderBy(ts_select_cols[2], seq_select_col) if has_seq_nb else Window.partitionBy(partitionCols).orderBy(ts_select_cols[2])

    final_selected_cols = partitionCols + [self.key, ts_select_cols[0],ts_select_cols[1]] if len(self.key) > 0 else partitionCols + [ts_select_cols[0],ts_select_cols[1]]
    final_selected_cols = final_selected_cols+[seq_select_col] if has_seq_nb else final_selected_cols

    full_df_with_seq = UnionDF.withColumn(ts_select_cols[1], last(col(ts_select_cols[1]), True).over(windowSpec))
    full_df_with_seq = full_df_with_seq.withColumn(seq_select_col, last(col(seq_select_col), True).over(windowSpec)) if has_seq_nb else full_df_with_seq

    return (full_df_with_seq
            .filter((col(ts_select_cols[2])== col(ts_select_cols[0]).cast("double")) & 
                    (col("is_original") == lit(1))) # Remove the overlapping partition parts in case we made use of time-partitions.
            .select(final_selected_cols))
  
  def asofJoin(self, right_TSDF, partitionCols = [], tsPartitionVal = None, fraction = 0.5, asof_prefix = None):

    right_DF = right_TSDF.df
    right_ts_col_name = right_TSDF.ts_col
    right_ts_col_name = self.ts_col if right_ts_col_name is None else right_ts_col_name
    #right_seq_col_name = self.seq_nb_col if right_seq_col_name is None else right_seq_col_name



    
    # Define timeColumns that we will use throughout the calculation of the asof Join
    ts_select_cols = ["_".join([self.ts_col,val]) for val in ["left","right","ms"]]
    seq_select_col = right_TSDF.seq_nb_col
    right_DF = right_DF.withColumnRenamed(right_ts_col_name,right_TSDF.ts_col)

    if (right_TSDF.seq_nb_col):
        right_DF = right_DF.withColumnRenamed(seq_select_col, right_TSDF.seq_nb_col)
   
    # in order to avoid duplicate output columns from the join, we'll supply a prefix for asof fields only
    prefix = asof_prefix + '_' if asof_prefix is not None else ''

    right_DF = right_DF.withColumn(self.key, lit(None)) if len(self.key) > 0 else right_DF
    right_DF = right_DF.toDF(*(prefix + c if c not in list(
        set().union(partitionCols, self.key, ts_select_cols, seq_select_col,
                    [self.ts_col, right_TSDF.seq_nb_col, right_ts_col_name])) else c for c in
                               right_DF.columns))

    right_TSDF.df = right_DF
    self.df = self.df.withColumn(seq_select_col, lit(None)) if len(right_TSDF.seq_nb_col) > 0 else self.df
    unionDF = self.__getUnionDF(right_TSDF, ts_select_cols, seq_select_col, partitionCols)

 
    # Only make use of  time partitions if tsPartitionVal was supplied
    if tsPartitionVal is None:
      asofDF = self.__getLastRightTs(unionDF,ts_select_cols,seq_select_col,partitionCols)
    else:
      tsPartitionDF = self.__getTimePartitions(unionDF,ts_select_cols,[seq_select_col],tsPartitionVal, fraction = fraction)
      asofDF = self.__getLastRightTs(tsPartitionDF,ts_select_cols,seq_select_col,partitionCols = partitionCols + ["ts_partition"])

    # Now need to join asofDF to self_df and right_df to get all the columns from the original dataframes.
    joinedDF = asofDF

    left_df_join_keys = [ts_select_cols[0], self.key] if len(self.key) > 0 else [ts_select_cols[0]]
    right_df_join_keys = [ts_select_cols[1]] + partitionCols
    left_df_join_keys = left_df_join_keys + partitionCols

    right_df_join_keys = right_df_join_keys + [right_TSDF.seq_nb_col] if len(right_TSDF.seq_nb_col) > 0 else right_df_join_keys

    joinedDF = (joinedDF.join(self.df.withColumnRenamed(self.ts_col,ts_select_cols[0]).drop(right_TSDF.seq_nb_col),left_df_join_keys) \
                .join(right_DF.withColumnRenamed(right_ts_col_name, ts_select_cols[1]).drop(self.key),right_df_join_keys)
                )
    return TSDF( joinedDF, self.ts_col , right_TSDF.seq_nb_col, self.key)
  
  def vwap(self, frequency='m',volume_col = "volume", price_col = "price", partitionCols = ['symbol']):
        # set pre_vwap as self or enrich with the frequency
        pre_vwap = self.df
        if frequency == 'm':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col(self.ts_col)), 2, '0'), lit(':'),
                                                               lpad(minute(col(self.ts_col)), 2, '0')))
        elif frequency == 'H':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col(self.ts_col)), 2, '0')))
        elif frequency == 'D':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(day(col(self.ts_col)), 2, '0')))

        vwapped = pre_vwap.withColumn("dllr_value", col(price_col) * col(volume_col)).groupby(partitionCols + ['time_group']).agg(
            sum('dllr_value').alias("dllr_value"), sum(volume_col).alias(volume_col),
            max(price_col).alias("_".join(["max",price_col]))).withColumn("vwap", col("dllr_value") / col(volume_col))
        return TSDF( vwapped, self.ts_col )
  
  def EMA(self,colName,window=30,exp_factor = 0.2,partitionCols = []):
    # Constructs an approximate EMA in the fashion of:
    # EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
    
    # Initialise EMA column:
    emaColName = "_".join(["EMA",colName])
    df = self.df.withColumn(emaColName,lit(0)).orderBy(self.ts_col)
    # Generate all the lag columns:
    for i in range(window):
      lagColName = "_".join(["lag",colName,str(i)])
      weight = exp_factor * (1 - exp_factor)**i
      df = df.withColumn(lagColName, weight * (lag(col(colName),i).over(Window.partitionBy(partitionCols).orderBy(lit(1)))))
      df = df.withColumn(emaColName,col(emaColName) + when(col(lagColName).isNull(),lit(0)).otherwise(col(lagColName))).drop(lagColName) # Nulls are currently removed
      
    return TSDF( df, self.ts_col )

  def withLookbackFeatures(self,
                           featureCols,
                           lookbackWindowSize,
                           exactSize=True,
                           featureColName="features",
                           partitionCols=[]):
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
      :param partitionCols: The names of any partition columns (columns whose values partition the DataFrame into
        independent timeseries)
      :return: a DataFrame with a feature column named featureColName containing the lookback feature tensor
      """
      # first, join all featureCols into a single array column
      tempArrayColName = "__TempArrayCol"
      feat_array_tsdf = self.df.withColumn(tempArrayColName, fn.array(featureCols))

      # construct a lookback array
      lookback_win = Window.orderBy(self.ts_col).rowsBetween(-lookbackWindowSize, -1)
      if partitionCols:
          lookback_win = lookback_win.partitionBy(partitionCols)
      lookback_tsdf = (feat_array_tsdf.withColumn(featureColName,
                                                  fn.collect_list(fn.col(tempArrayColName)).over(lookback_win))
                                      .drop(tempArrayColName))

      # make sure only windows of exact size are allowed
      if exactSize:
          return lookback_tsdf.where(fn.size(featureColName) == lookbackWindowSize)

      return TSDF( lookback_tsdf, self.ts_col )

  def withRangeStats(self, type='range', partitionCols=[], colsToSummarize=[], rangeBackWindowSecs=1000):
          """
          Create a wider set of stats based on all numeric columns by default
          Users can choose which columns they want to summarize also. These stats are:
          mean/count/min/max/sum/std deviation/zscore
          :param type - this is created in case we want to extend these stats to lookback over a fixed number of rows instead of ranging over column values
          :param partitionCols - list of partitions columns to be used for the range windowing
          :param colsToSummarize - list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
          :param rangeBackWindowSecs - lookback this many seconds in time to summarize all stats. Note this will look back from the floor of the base event timestamp (as opposed to the exact time since we cast to long)
          Assumptions:
               1. The features are summarized over a rolling window that ranges back
               2. The range back window can be specified by the user
               3. Sequence numbers are not yet supported for the sort
               4. There is a cast to long from timestamp so microseconds or more likely breaks down - this could be more easily handled with a string timestamp or sorting the timestamp itself. If using a 'rows preceding' window, this wouldn't be a problem
           """

          # build window
          w = ( Window().orderBy(col(self.ts_col).cast("long"))
                        .rangeBetween(-1 * rangeBackWindowSecs, 0) )
          if partitionCols:
            w = w.partitionBy([col(elem) for elem in partitionCols])

          # identify columns to summarize if not provided
          # these should include all numeric columns that
          # are not the timestamp column and not any of the partition columns
          if not colsToSummarize:
            # columns we should never summarize
            prohibited_cols = [ self.ts_col.lower() ]
            if partitionCols:
              prohibited_cols.extend([ pc.lower() for pc in partitionCols])
            # types that can be summarized
            summarizable_types = ['int', 'bigint', 'float', 'double']
            # filter columns to find summarizable columns
            colsToSummarize = [datatype[0] for datatype in self.df.dtypes if
                                ((datatype[1] in summarizable_types) and
                                 (datatype[0].lower() not in prohibited_cols))]

          # compute column summaries
          selectedCols = self.df.columns
          derivedCols = []
          for metric in colsToSummarize:
              selectedCols.append(mean(metric).over(w).alias('mean_' + metric))
              selectedCols.append(count(metric).over(w).alias('count_' + metric))
              selectedCols.append(min(metric).over(w).alias('min_' + metric))
              selectedCols.append(max(metric).over(w).alias('max_' + metric))
              selectedCols.append(sum(metric).over(w).alias('sum_' + metric))
              selectedCols.append(stddev(metric).over(w).alias('stddev_' + metric))
              derivedCols.append(
                      ((col(metric) - col('mean_' + metric)) / col('stddev_' + metric)).alias("zscore_" + metric))
          selected_df = self.df.select(*selectedCols)

          summary_df = selected_df.select(*selected_df.columns, *derivedCols)

          return TSDF( summary_df, self.ts_col )