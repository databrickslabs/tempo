from tempo.tsdf import TSDF
import pyspark.sql.functions as F

def calc_anomalies(spark, yaml_file):

    import os
    import yaml

    yaml_path = ''

    if (yaml_file.startswith("s3")):
        import boto3

        s3 = boto3.client('s3')
        s3_bucket = yaml_file.split("s3://")[1]
        bucket_file_tuple = s3_bucket.split("/")
        bucket_name_only = bucket_file_tuple[0]
        file_only = "/".join(bucket_file_tuple[1:])

        s3.download_file(bucket_name_only, file_only, 'ad.yaml')
        yaml_path = '/databricks/driver/ad.yaml'
    elif yaml_file.startswith("dbfs:/") & (os.getenv('DATABRICKS_RUNTIME_VERSION') != None):
        new_dbfs_path = "/" + yaml_file.replace(":", "")
        yaml_path = new_dbfs_path
    else:
        yaml_path = yaml_file

    print(yaml_path)

    with open(yaml_path) as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
        print('data type is ' + str(type(data)))

    import json
    for d in data.keys():
        print(d)
        print(data[d])
        table = data[d]['database'] + '.' + data[d]['name']
        tgt_table = 'tempo.' + data[d]['name']
        df = spark.table(table)
        partition_cols = data[d]['partition_cols']
        ts_col = data[d]['ts_col']
        mode = data[d]['mode']
        metrics = data[d]['metrics']
        lkbck_window = data[d]['lookback_window']
        #tsdf = TSDF(df, partition_cols = partition_cols, ts_col = ts_col)

        # logic to stack metrics instead of individual columns
        l = []
        sep = ', '
        sql_list = sep.join(metrics).split(",")
        n = len(metrics)
        for a in range(n):
            l.append("'{}'".format(metrics[a]) + "," + sql_list[a])
        # partition_col string
        partition_cols_str = ", ".join(partition_cols)
        metrics_cols_str = ", ".join(metrics)
        k = sep.join(l)
        for metric_col in metrics:
            df = df.withColumn(metric_col, F.col(metric_col).cast("double"))

        df.createOrReplaceTempView("tsdf_view")
        stacked = spark.sql("select {}, {}, {}, stack({}, {}) as (metric, value) from tsdf_view".format(ts_col, partition_cols_str, metrics_cols_str, n, k))

        part_cols_w_metrics = partition_cols + metrics

        tsdf = TSDF(stacked, partition_cols = part_cols_w_metrics, ts_col = ts_col)
        moving_avg = tsdf.withRangeStats(['value'], rangeBackWindowSecs=int(lkbck_window)).df
        anomalies = moving_avg.select(ts_col, *partition_cols, 'metric', 'zscore_' + 'value').withColumn("anomaly_fl", F.when(F.col('zscore_' + 'value') > 2.5, 1).otherwise(0))

        # class 1 - 2.5 standard deviations outside mean
        # brand new table
        if mode == 'new':
            spark.sql("create database if not exists tempo")
            anomalies.write.mode('overwrite').option("overwriteSchema", "true").format("delta").saveAsTable(tgt_table + "_class1")
        # append to existing table without DLT
        elif mode == 'append':
            # incremental append with DLT
            print('hey')
        elif (mode == 'incremental') & (os.getenv('DATABRICKS_RUNTIME_VERSION') != None):
            import dlt
            @dlt.view
            def taxi_raw():
              return spark.read.json("/databricks-datasets/nyctaxi/sample/json/")

            # Use the function name as the table name
            @dlt.table
            def filtered_data():
              return dlt.read("taxi_raw").where(...)

            # Use the name parameter as the table name
            @dlt.table(name="filtered_data")
            def create_filtered_data():
              return dlt.read("taxi_raw").where(...)
            #anomalies.write.format("delta").saveAsTable("class1")

