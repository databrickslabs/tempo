import pyspark.sql.functions as f
import tempo.tsdf as ts

# define global frequency options
SEC = 'sec'
MIN = 'min'
HR = 'hr'

# define global aggregate function options for downsampling
CLOSEST_LEAD = "closest_lead"
MIN_LEAD = "min_lead"
MAX_LEAD = "max_lead"
MEAN_LEAD = "mean_lead"

allowableFreqs = [SEC, MIN, HR]

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
        #agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lpad(hour_col, 2, '0'), f.lpad(min_col, 2, '0'), f.lpad(sec_col, 2, '0'))
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(" "), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lpad(min_col, 2, '0'), f.lit(':'), f.lpad(sec_col, 2, '0')).cast("timestamp")
    elif (freq == MIN):
        #agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lpad(hour_col, 2, '0'), f.lpad(min_col, 2, '0'))
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(' '), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lpad(min_col, 2, '0'), f.lit(':'), f.lit('00')).cast("timestamp")
    elif (freq == HR):
        #agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lpad(hour_col, 2, '0'))
        agg_key = f.concat(f.col(tsdf.ts_col).cast("date"), f.lit(' '), f.lpad(hour_col, 2, '0'), f.lit(':'), f.lit('00'), f.lit(':'), f.lit('00')).cast("timestamp")

    df = df.withColumn("agg_key", agg_key)
    return ts.TSDF(df, tsdf.ts_col, partition_cols = tsdf.partitionCols)

def aggregate(tsdf, freq, func, metricCols = None):
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

    if func == CLOSEST_LEAD:
        #exprs = {x: "min" for x in metricCols}
        metricCol = f.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(f.min('struct_cols').alias("closest_data")).select(*groupingCols, f.col("closest_data.*"))
    elif func == MEAN_LEAD:
        exprs = {x: "avg" for x in metricCols}
        res= df.groupBy(groupingCols).agg(exprs)
    elif func == MIN_LEAD:
        exprs = {x: "min" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)
    elif func == MAX_LEAD:
        exprs = {x: "max" for x in metricCols}
        summaryCols = metricCols
        res = df.groupBy(groupingCols).agg(exprs)

    res = res.drop(tsdf.ts_col).withColumnRenamed('agg_key', tsdf.ts_col)
    return(ts.TSDF(res, ts_col = tsdf.ts_col, partition_cols = tsdf.partitionCols))


def checkAllowableFreq(freq):
    if freq not in allowableFreqs:
      raise ValueError("Allowable grouping frequencies are sec (second), min (minute), hr (hour)")

def validateFuncExists(func):
  if func is None:
      raise ValueError("Aggregate function missing. Provide one of the allowable functions: " + ", ".join([CLOSEST_LEAD, MIN_LEAD, MAX_LEAD, MEAN_LEAD]))