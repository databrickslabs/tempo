# tempo - Time Series Utilities for Data Teams Using Databricks

<p align="center">
  <img src="tempo - black background.svg" width="300px"/>
</p>


## Project Description
The purpose of this project is to make time series manipulation with Spark simpler. Operations covered under this package include AS OF joins, rolling statistics with user-specified window lengths, featurization of time series using lagged values, and Delta Lake optimization on time and partition fields.

[![codecov](https://codecov.io/gh/databrickslabs/tempo/branch/master/graph/badge.svg)](https://codecov.io/gh/databrickslabs/tempo)

## Using the Project

### Starting Point: TSDF object, a wrapper over a Spark data frame
The entry point into all features for time series analysis in tempo is a TSDF object which wraps the Spark data frame. At a high level, a TSDF contains a data frame which contains many smaller time series, one per partition key. In order to create a TSDF object, a distinguished timestamp column much be provided in order for sorting purposes for public methods. Optionally, a sequence number and partition columns can be provided as the assumptive columns on which to create new features from. Below are the public methods available for TSDF transformation and enrichment.

#### Sample Reference Architecture for Capital Markets

<p align="center">
  <img src="ts_in_fs.png" width="700px"/>
</p>

#### 1. asofJoin - AS OF Join to Paste Latest AS OF Information onto Fact Table

##### This join uses windowing in order to select the latest record from a source table and merges this onto the base Fact table

Parameters: 

ts_col = timestamp column on which to sort fact and source table
partition_cols - columns to use for partitioning the TSDF into more granular time series for windowing and sorting

```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
normal_asof_result = base_trades.asofJoin(skewQuotes, partition_cols = ["symbol"], right_prefix = 'asof').df
```

#### 2. Skew Join Optimized AS OF Join

The purpose of the skew optimized as of join is to bucket each set of `partition_cols` to get the latest source record merged onto the fact table

Parameters: 

ts_col = timestamp column for sorting 
partition_cols = partition columns for defining granular time series for windowing and sorting
tsPartitionVal = value to break up each partition into time brackets
fraction = overlap fraction
right_prefix = prefix used for source columns when merged into fact table

```
from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
partitioned_asof_result = base_trades.asofJoin(skewQuotes, partition_cols = ["symbol"], tsPartitionVal = 1200, fraction = 0.1, right_prefix='asof').df
```

#### 3 - Approximate Exponential Moving Average

The approximate exponential moving average uses an approximation of the form `EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) ` to define a rolling moving average based on exponential decay.

Parameters: 

ts_col = timestamp on which to sort for computing previous `n` terms where `n` is the size of the window
window = number of lagged values to compute for moving average

```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
ema_trades = base_trades.EMA("trade_pr", window = 180, partitionCols = ["symbol"]).df
```

#### 4 - Volume-weighted average point (VWAP) Calculation

This calculation computes a volume-weighted average point, where point can be any feature, e.g. a price, a temperature reading, etc.

Parameters: 

ts_col = column on which to bin for VWAP calculation (default to minute unit)
price_col = feature column on which to aggregate

```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
vwap_res = base_trades.vwap(price_col = "trade_pr").df
```

#### 5 - Time Series Lookback Feature Generation

Method for placing lagged values into an array for traditional ML methods

Parameters: 

ts_col = timestamp column used for sorting and computing lagged values per partition 
partitionCols = columns to use for more granular time series calculation
lookbackWindowSize = cardinality of computed feature vector
featureCols = features to aggregate into array

```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
res_df = base_trades.withLookbackFeatures(featureCols = ['trade_pr'] , lookbackWindowSize = 20, partitionCols=['symbol']).df
```

#### 6 - Range Stats Lookback Append

Method for computing rolling statistics based on the distinguished timestamp column 

Parameters: 

ts_col = timestamp column used for sorting values to get rolling values
partitionCols = partition columns used for the range stats windowing in Spark
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
res_stats = base_trades.withRangeStats(partitionCols=['symbol']).df
```


## Project Support
Please note that all projects in the /databrickslabs github account are provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs).  They are provided AS-IS and we do not make any guarantees of any kind.  Please do not submit a support ticket relating to any issues arising from the use of these projects.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo.  They will be reviewed as time permits, but there are no formal SLAs for support.

## Unit Test Coverage (as of 11/13/20)

```
coverage report -m
Name                Stmts   Miss  Cover   Missing
-------------------------------------------------
tempo/__init__.py       1      0   100%
tempo/io.py            23      5    78%   17, 34-37
tempo/resample.py      54     16    70%   34, 38-40, 67-77, 85, 89
tempo/tsdf.py         150     37    75%   30, 32, 38, 41, 43, 52, 202, 212, 216-236, 246-258, 281-294
tests/__init__.py       0      0   100%
tests/tests.py        117      2    98%   81, 412
-------------------------------------------------
TOTAL                 345     60    83%
```

## Project Setup
After cloning the repo, it is highly advised that you create a [virtual environment](https://docs.python.org/3/library/venv.html) to isolate and manage
packages for this project, like so:

`python -m venv <path to project root>/venv`

You can then install the required modules via pip:

`pip install requirements.txt`

## Building the Project
Once in the main project folder, build into a wheel using the following command: 

`python setup.py bdist_wheel`

## Deploying / Installing the Project
For installation in a Databricks notebook (using Databricks Runtime for ML), you'll need to upload to the FileStore via UI (or directly). If uploading via the UI, you may need to rename with the commands below. Also below is the command to install the wheel into the notebook scope:

`%fs cp /FileStore/tables/tempo_0_1_py3_none_any-1f645.whl /FileStore/tables/tempo-0.1-py3-none-any.whl`

`%pip install /FileStore/tables/tempo-0.1-py3-none-any.whl`

## Releasing the Project
Instructions for how to release a version of the project
