import pandas as pd
from random import random
from time import time

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

def build_test_as_of_join(num_cols: int, num_rows: int = 1000) -> DataFrame:
    # Create 1000 feature columns of doubles
    features = {f"feature_{n}" : [random() for i in range(num_rows)] for n in range(num_cols)}

    users = pd.DataFrame({
      "user" :  [i for i in range(num_rows)],
      **features
    })
    desired_order = ["ts"] + list(users.columns)
    spark = SparkSession.builder.getOrCreate()
    feature_df = spark.createDataFrame(users).withColumn("ts", F.current_timestamp())
    feature_df = feature_df.select(*desired_order)

    user_subset = users[["user"]]
    users_df = spark.createDataFrame(user_subset).withColumn("ts", F.current_timestamp())

    users_df.join(feature_df, on='user', how='left')

    from tempo.tsdf import TSDF

    df_tsdf = TSDF(users_df, ts_col='ts')
    ft_tsdf = TSDF(
        feature_df,
        ts_col='ts',
    )

    start_ts = time()
    joined_df = df_tsdf.asofJoin(
            ft_tsdf,
            left_prefix="left",
            right_prefix="right",
            skipNulls=True,
            tolerance=None
        ).df
    end_ts = time()
    print(f"Time to construct join for {num_cols} columns: {end_ts - start_ts}")

    return joined_df

# run for several different numbers of columns
# for test_col in [10, 20, 50, 100, 150, 200, 250, 300, 400, 500]:
#      build_test_as_of_join(test_col)

build_test_as_of_join(500)