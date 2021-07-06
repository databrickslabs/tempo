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

allowableFreqs = [SEC, MIN, HR, DAY]

def __appendAggKey(tsdf, freq = None):
    """
    :param tsdf: TSDF object as input
    :param freq: frequency at which to upsample
    :return: return a TSDF with a new aggregate key (called agg_key)
    """
    df = tsdf.df
    checkAllowableFreq(freq)

    # compute timestamp columns
    sec_col = f.second(f.col(tsdf.ts_col))
    min_col = f.minute(f.col(tsdf.ts_col))
    hour_col = f.hour(f.col(tsdf.ts_col))

    if (freq == SEC):
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(" "), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lpad(min_col, 2, '0'), f.lit(':'), f.lpad(sec_col, 2, '0')).cast("timestamp")
    elif (freq == MIN):
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(' '), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lpad(min_col, 2, '0'), f.lit(':'), f.lit('00')).cast("timestamp")
    elif (freq == HR):
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(' '), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lit('00'), f.lit(':'), f.lit('00')).cast("timestamp")
    elif (freq == DAY):
        agg_key = f.col(tsdf.ts_col).cast("date").cast("timestamp")

    df = df.withColumn("agg_key", agg_key)
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

    groupingCols = [f.col(column) for column in groupingCols]

    if func == floor:
        if prefix is None:
            prefix = floor
        metricCol = f.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(f.min('struct_cols').alias("closest_data")).select(*groupingCols, f.col("closest_data.*"))
        new_cols = [f.col(tsdf.ts_col)] + [f.col(c).alias("{}_".format(prefix) + c) for c in metricCols]
        res = res.select(*groupingCols, *new_cols)
    elif func == average:
        if prefix is None:
          prefix = average
        exprs = {x: "avg" for x in metricCols}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}_'.format(prefix) + (c.split("avg(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == min:
        if prefix is None:
          prefix = min
        exprs = {x: "min" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}_'.format(prefix) + (c.split("min(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == max:
        if prefix is None:
            prefix = max
        exprs = {x: "max" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(set(res.columns).difference(set(tsdf.partitionCols + [tsdf.ts_col, 'agg_key'])))
        new_cols = [f.col(c).alias('{}_'.format(prefix) + (c.split("max(")[1]).replace(')', '')) for c in agg_metric_cls]
        res = res.select(*groupingCols, *new_cols)
    elif func == ceiling:
        if prefix is None:
            prefix = ceiling
        metricCol = f.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(f.max('struct_cols').alias("ceil_data")).select(*groupingCols, f.col("ceil_data.*"))
        new_cols = [f.col(tsdf.ts_col)] + [f.col(c).alias("{}_".format(prefix) + c) for c in metricCols]
        res = res.select(*groupingCols, *new_cols)

    res = res.drop(tsdf.ts_col).withColumnRenamed('agg_key', tsdf.ts_col)
    return(tempo.TSDF(res, ts_col = tsdf.ts_col, partition_cols = tsdf.partitionCols))


def checkAllowableFreq(freq):
    if freq not in allowableFreqs:
      raise ValueError("Allowable grouping frequencies are sec (second), min (minute), hr (hour)")

def validateFuncExists(func):
  if func is None:
      raise ValueError("Aggregate function missing. Provide one of the allowable functions: " + ", ".join([CLOSEST_LEAD, MIN_LEAD, MAX_LEAD, MEAN_LEAD]))
