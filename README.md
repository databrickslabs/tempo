# Tempo - Time Series Utilities for Data Teams Using Databricks
Standard Project Template for Databricks Labs Projects

## Project Description
The purpose of this project is to provide easier ways to perform machine learning experiments, ETL, and ad-hoc analytics on time series within Databricks using Apache Spark. This includes data parallel and model parallel use cases encountered across the field. 

## Project Support
Please note that all projects in the /databrickslabs github account are provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs).  They are provided AS-IS and we do not make any guarantees of any kind.  Please do not submit a support ticket relating to any issues arising from the use of these projects.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo.  They will be reviewed as time permits, but there are no formal SLAs for support.

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

## Using the Project

#### Example 1 - AS OF Join to Paste Latest Quote Information onto Trade

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
```

#### Example 1 - AS OF Join to Paste Latest Quote Information onto Trade
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
normal_asof_result = base_trades.asofJoin(skewQuotes,partitionCols = ["symbol"], asof_prefix = 'asof').df
```

#### Example 2 - AS OF Join - Skew Join Optimized
```
from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
partitioned_asof_result = base_trades.asofJoin(skewQuotes, partitionCols = ["symbol"], tsPartitionVal = 1200, fraction = 0.1, asof_prefix='asof').df
```

#### Example 3 - Exponential Moving Average Approximated
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
ema_trades = base_trades.EMA("trade_pr", window = 180, partitionCols = ["symbol"]).df
```

#### Example 4 - VWAP Calculation
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
vwap_res = base_trades.vwap(price_col = "trade_pr").df
```

#### Example 5 - Time Series Lookback Feature Generation
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
res_df = base_trades.withLookbackFeatures(featureCols = ['trade_pr'] , lookbackWindowSize = 20, partitionCols=['symbol']).df
```

#### Example 6 - Range Stats Lookback Append
```

from tempo import *

base_trades = TSDF(skewTrades, ts_col = 'event_ts')
res_stats = base_trades.withRangeStats(partitionCols=['symbol']).df
```