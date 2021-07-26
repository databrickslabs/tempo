import pyspark.sql.functions as f

# define global frequency options
import tempo

SEC = 'sec'
MIN = 'min'
HR = 'hr'
DAY = 'day'

# define global aggregate function options for downsampling
floor = "floor"
min = "min"
max = "max"
average = "mean"
ceiling = "ceil"

freq_dict = {'sec' : 'seconds', 'min' : 'minutes', 'hr' : 'hours', 'day' : 'days'}

allowableFreqs = [SEC, MIN, HR, DAY]
allowableFuncs = [floor, min, max, average, ceiling]

def __appendAggKey(tsdf, freq = None):
    """
    :param tsdf: TSDF object as input
    :param freq: frequency at which to upsample
    :return: return a TSDF with a new aggregate key (called agg_key)
    """
    df = tsdf.df
    parsed_freq = checkAllowableFreq(tsdf, freq)
    agg_window = f.window(f.col(tsdf.ts_col), "{} {}".format(parsed_freq[0], freq_dict[parsed_freq[1]]))

    df = df.withColumn("agg_key", agg_window)
    return tempo.TSDF(df, tsdf.ts_col, partition_cols = tsdf.partitionCols)

def aggregate(tsdf, freq, func, metricCols = None, prefix = None):
    """
    aggregate a data frame by a coarser timestamp than the initial TSDF ts_col
    :param tsdf: input TSDF object
    :param func: aggregate function
    :param metricCols: columns used for aggregates
    :return: TSDF object with newly aggregated timestamp as ts_col with aggregated values
    """
    tsdf = __appendAggKey(tsdf, freq)
    df = tsdf.df

    groupingCols = tsdf.partitionCols + ['agg_key']
    if metricCols is None:
        metricCols = list(set(df.columns).difference(set(groupingCols + [tsdf.ts_col])))
    if prefix is None:
        prefix = ''
    else:
        prefix = prefix+'_'

    groupingCols = [f.col(column) for column in groupingCols]

    if func == floor:
        metricCol = f.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(f.min('struct_cols').alias("closest_data")).select(*groupingCols, f.col("closest_data.*"))
        new_cols = [f.col(tsdf.ts_col)] + [f.col(c).alias("{}".format(prefix) + c) for c in metricCols]
        res = res.select(*groupingCols, *new_cols)
    elif func == average:
        exprs = {x: "avg" for x in metricCols}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}'.format(prefix) + (c.split("avg(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == min:
        exprs = {x: "min" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}'.format(prefix) + (c.split("min(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == max:
        exprs = {x: "max" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}'.format(prefix) + (c.split("max(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == ceiling:
        metricCol = f.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(f.max('struct_cols').alias("ceil_data")).select(*groupingCols, f.col("ceil_data.*"))
        new_cols = [f.col(tsdf.ts_col)] + [f.col(c).alias("{}".format(prefix) + c) for c in metricCols]
        res = res.select(*groupingCols, *new_cols)

    # aggregate by the window and drop the end time (use start time as new ts_col)
    res = res.drop(tsdf.ts_col).withColumnRenamed('agg_key', tsdf.ts_col).withColumn(tsdf.ts_col, f.col(tsdf.ts_col).start)

    # sort columns so they are consistent
    non_part_cols = set(set(res.columns) - set(tsdf.partitionCols)) - set([tsdf.ts_col])
    sel_and_sort = tsdf.partitionCols + [tsdf.ts_col] + sorted(non_part_cols)
    res = res.select(sel_and_sort)
    return(tempo.TSDF(res, ts_col = tsdf.ts_col, partition_cols = tsdf.partitionCols))


def checkAllowableFreq(tsdf, freq):
    if freq not in allowableFreqs:
      try:
          periods = freq.lower().split(" ")[0].strip()
          units = freq.lower().split(" ")[1].strip()
      except:
          raise ValueError("Allowable grouping frequencies are sec (second), min (minute), hr (hour), day. Reformat your frequency as <integer> <day/hour/minute/second>")
      if units.startswith(SEC):
          return (periods, SEC)
      elif units.startswith(MIN):
          return (periods, MIN)
      elif units.startswith("hour"):
          return (periods, "hour")
      elif units.startswith(DAY):
          return (periods, DAY)
    elif freq in allowableFreqs:
      return (1, freq)


def validateFuncExists(func):
  if ( (func is None)):
      raise ValueError("Aggregate function missing. Provide one of the allowable functions: " + ", ".join(allowableFuncs))
  elif (func not in allowableFuncs):
      raise ValueError("Aggregate function is not in the valid list. Provide one of the allowable functions: " + ", ".join(allowableFuncs))

