# Tempo - Time Series Utilities for Data Teams Using Databricks
Standard Project Template for Databricks Labs Projects

## Project Description
The purpose of this project is to provide easier ways to perform machine learning experiments, ETL, and ad-hoc analytics on time series within Databricks using Apache Spark. This includes data parallel and model parallel use cases encountered across the field. 

## Project Support
Please note that all projects in the /databrickslabs github account are provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs).  They are provided AS-IS and we do not make any guarantees of any kind.  Please do not submit a support ticket relating to any issues arising from the use of these projects.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo.  They will be reviewed as time permits, but there are no formal SLAs for support.


## Building the Project
Once in the main project folder, build into a wheel using the following command: 

`python setup.py bdist_wheel`

## Deploying / Installing the Project
For installation in a Databricks notebook, you'll need to upload to the FileStore via UI (or directly). If uploading via the UI, you may need to rename with the commands below. Also below is the command to install the wheel into the notebook scope: 

`%fs cp /FileStore/tables/tca_0_1_py3_none_any-1f645.whl /FileStore/tables/tca-0.1-py3-none-any.whl`

`dbutils.library.install("/FileStore/tables/tca-0.1-py3-none-any.whl") #  Library
dbutils.library.restartPython()`

## Releasing the Project
Instructions for how to release a version of the project

## Using the Project

```from tca.base import newBaseTs 

base_trades = newBaseTs(skewTrades)
normal_asof_result = base_trades.asofJoin(skewQuotes,partitionCols = ["symbol"])
normal_asof_result.select("EVENT_TS_left").distinct().count()```
