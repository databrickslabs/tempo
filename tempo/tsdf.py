import pyspark.sql.functions as fn

from pyspark.sql.functions import *
from pyspark.sql.window import Window

class TSDF:
  
  def __init__(self, df, ts_col = "EVENT_TS", partitionCols = []):
    """
    Constructor
    :param df:
    :param ts_col:
    :param partitionCols:
    """
    self.df = df
    self.ts_col = self.__validated_column(ts_col)
    self.partitionCols = self.__validated_columns(partitionCols)

  ##
  ## Helper functions
  ##

  def __validated_column(self,colname):
    if type(colname) != str:
      raise TypeError(f"Column names must be of type str; found {type(colname)} instead!")
    if colname.lower() not in [col.lower() for col in self.df.columns]:
      raise ValueError(f"Column {colname} not found in Dataframe")
    return colname

  def __validated_columns(self,colnames):
    # if provided a string, treat it as a single column
    if type(colnames) == str:
      colnames = [ colnames ]
    # otherwise we really should have a list or None
    if colnames is None:
      colnames = []
    elif type(colnames) != list:
      raise TypeError(f"Columns must be of type list, str, or None; found {type(colname)} instead!")
    # validate each column
    for col in colnames:
      self.__validated_column(col)
    return colnames

  def __createTimeSeriesDF(self, df, ts_select_cols, fillLeft = True, partitionCols = []):
    left_ts_val, right_ts_val = (col(self.ts_col), lit(None)) if fillLeft else (lit(None),col(self.ts_col))
    
    return (df
            .withColumn(ts_select_cols[0],left_ts_val)
            .withColumn(ts_select_cols[1],right_ts_val)
            .withColumn(ts_select_cols[2],col(self.ts_col).cast("double"))
            .select(ts_select_cols + partitionCols))
  
  def __getUnionDF(self,df_right, ts_select_cols, partitionCols):
    df_left = self.__createTimeSeriesDF(self.df, ts_select_cols, partitionCols=partitionCols)
    df_right = self.__createTimeSeriesDF(df_right, ts_select_cols,
                                       fillLeft = False, partitionCols = partitionCols) # df_right ts col should have same ts column name, else error
    return df_left.select(sorted(df_left.columns)).union(df_right.select(sorted(df_right.columns))).withColumn("is_original",lit(1))
    
  def __getTimePartitions(self,UnionDF, ts_select_cols,tsPartitionVal, fraction = 0.5):
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
  
  def __getLastRightTs(self,UnionDF, ts_select_cols, partitionCols = []):
    """Calculates the last observed timestamp of the right dataframe for each timestamp in the left dataframe"""
    windowSpec = Window.partitionBy(partitionCols).orderBy(ts_select_cols[2])
    
    return (UnionDF
            .withColumn(ts_select_cols[1], last(col(ts_select_cols[1]), True).over(windowSpec))
            .filter((col(ts_select_cols[2])== col(ts_select_cols[0]).cast("double")) & 
                    (col("is_original") == lit(1))) # Remove the overlapping partition parts in case we made use of time-partitions.
            .select(partitionCols + [ts_select_cols[0],ts_select_cols[1]]))

  def __baseWindow(self):
    w = Window().orderBy(fn.col(self.ts_col).cast("long"))
    if self.partitionCols:
      w = w.partitionBy([fn.col(elem) for elem in self.partitionCols])
    return w

  def __rangeBetweenWindow(self, range_from, range_to):
    return self.__baseWindow().rangeBetween(range_from, range_to)

  def __rowsBetweenWindow(self, rows_from, rows_to):
    return self.__baseWindow().rowsBetween(rows_from, rows_to)

  ##
  ## TSDF API functions
  ##

  def withPartitionCols(self, partitionCols):
    """
    Sets certain columns of the TSDF as partition columns. Partition columns are those that differentiate distinct timeseries
    from each other.
    :param partitionCols: a list of columns used to partition distinct timeseries
    :return: a TSDF object with the given partition columns
    """
    return TSDF( self.df, self.ts_col, partitionCols )

  def asofJoin(self, right_DF, right_ts_col_name = None, tsPartitionVal = None, fraction = 0.5, asof_prefix = None):
    
    right_ts_col_name = self.ts_col if right_ts_col_name is None else right_ts_col_name
    
    # Define timeColumns that we will use throughout the calculation of the asof Join
    ts_select_cols = ["_".join([self.ts_col,val]) for val in ["left","right","ms"]] 
    right_DF = right_DF.withColumnRenamed(right_ts_col_name,self.ts_col)
   
    # in order to avoid duplicate output columns from the join, we'll supply a prefix for asof fields only
    prefix = asof_prefix + '_' if asof_prefix is not None else ''
    right_DF = right_DF.toDF(*(prefix+c if c not in list(set().union(self.partitionCols,ts_select_cols, [self.ts_col, right_ts_col_name])) else c for c in right_DF.columns))
    unionDF = self.__getUnionDF(right_DF.withColumnRenamed(right_ts_col_name,self.ts_col),ts_select_cols,self.partitionCols)
 
    # Only make use of  time partitions if tsPartitionVal was supplied
    if tsPartitionVal is None:
      asofDF = self.__getLastRightTs(unionDF,ts_select_cols,self.partitionCols)
    else:
      tsPartitionDF = self.__getTimePartitions(unionDF,ts_select_cols,tsPartitionVal, fraction = fraction)
      asofDF = self.__getLastRightTs(tsPartitionDF,ts_select_cols,partitionCols = self.partitionCols + ["ts_partition"])
      
    # Now need to join asofDF to self_df and right_df to get all the columns from the original dataframes.
    joinedDF = (asofDF
                .join(self.df.withColumnRenamed(self.ts_col,ts_select_cols[0]),[ts_select_cols[0]]+ self.partitionCols)
                .join(right_DF.withColumnRenamed(right_ts_col_name, ts_select_cols[1]),[ts_select_cols[1]] + self.partitionCols)
                )

    return TSDF(joinedDF, ts_select_cols[0], self.partitionCols)
  
  def vwap(self, frequency='m',volume_col = "volume", price_col = "price"):
        # set pre_vwap as self or enrich with the frequency
        pre_vwap = self.df
        print('input schema: ', pre_vwap.printSchema())
        if frequency == 'm':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col(self.ts_col)), 2, '0'), lit(':'),
                                                               lpad(minute(col(self.ts_col)), 2, '0')))
        elif frequency == 'H':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col(self.ts_col)), 2, '0')))
        elif frequency == 'D':
            pre_vwap = self.df.withColumn("time_group", concat(lpad(day(col(self.ts_col)), 2, '0')))

        group_cols = ['time_group']
        if self.partitionCols:
          group_cols.extend(self.partitionCols)
        vwapped = ( pre_vwap.withColumn("dllr_value", col(price_col) * col(volume_col))
                            .groupby(group_cols)
                            .agg( sum('dllr_value').alias("dllr_value"),
                                  sum(volume_col).alias(volume_col),
                                  max(price_col).alias("_".join(["max",price_col])) )
                            .withColumn("vwap", col("dllr_value") / col(volume_col)) )

        return TSDF( vwapped, self.ts_col, self.partitionCols )
  
  def EMA(self,colName,window=30,exp_factor = 0.2):
    # Constructs an approximate EMA in the fashion of:
    # EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
    
    # Initialise EMA column:
    emaColName = "_".join(["EMA",colName])
    df = self.df.withColumn(emaColName,lit(0)).orderBy(self.ts_col)
    w = self.__baseWindow()
    # Generate all the lag columns:
    for i in range(window):
      lagColName = "_".join(["lag",colName,str(i)])
      weight = exp_factor * (1 - exp_factor)**i
      df = df.withColumn(lagColName, weight * lag(col(colName),i).over(w) )
      df = df.withColumn(emaColName,col(emaColName) + when(col(lagColName).isNull(),lit(0)).otherwise(col(lagColName))).drop(lagColName) # Nulls are currently removed
      
    return TSDF( df, self.ts_col, self.partitionCols )

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
      feat_array_tsdf = self.df.withColumn(tempArrayColName, fn.array(featureCols))

      # construct a lookback array
      lookback_win = self.__rowsBetweenWindow(-lookbackWindowSize, -1)
      lookback_tsdf = (feat_array_tsdf.withColumn(featureColName,
                                                  fn.collect_list(fn.col(tempArrayColName)).over(lookback_win))
                                      .drop(tempArrayColName))

      # make sure only windows of exact size are allowed
      if exactSize:
          return lookback_tsdf.where(fn.size(featureColName) == lookbackWindowSize)

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
              selectedCols.append(mean(metric).over(w).alias('mean_' + metric))
              selectedCols.append(count(metric).over(w).alias('count_' + metric))
              selectedCols.append(min(metric).over(w).alias('min_' + metric))
              selectedCols.append(max(metric).over(w).alias('max_' + metric))
              selectedCols.append(sum(metric).over(w).alias('sum_' + metric))
              selectedCols.append(stddev(metric).over(w).alias('stddev_' + metric))
              derivedCols.append(
                      ((col(metric) - col('mean_' + metric)) / col('stddev_' + metric)).alias("zscore_" + metric))
          selected_df = self.df.select(*selectedCols)
          #print(derivedCols)
          summary_df = selected_df.select(*selected_df.columns, *derivedCols)

          return TSDF(summary_df, self.ts_col, self.partitionCols)