package com.databrickslabs.tempo.utils

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.collection.JavaConverters._
/**
  * Enables access to the Spark variable.
  * Additional logic can be added to the if statement to enable different types of spark environments
  * Common uses include DBRemote and local, driver only spark, and local docker configured spark
  */
trait SparkSessionWrapper extends Serializable {

  /**
    * Access to spark
    * If testing locally or using DBConnect, the System variable "OVERWATCH" is set to "LOCAL" to make the code base
    * behavior differently to work in remote execution AND/OR local only mode but local only mode
    * requires some additional setup.
    */
  lazy val spark: SparkSession = if (!(sys.env.get("DATABRICKS_RUNTIME_VERSION") == None)){
    println("Using Databricks SparkSession")
    SparkSession
      .builder().appName("Tempo")
      .getOrCreate()
  } else {
    println("Using Custom, local SparkSession")
    SparkSession.builder()
      .master("local")
      .appName("Tempo")
      //      .config("spark.driver.bindAddress", "0.0.0.0")
      //      .enableHiveSupport()
      //      .config("spark.warehouse.dir", "metastore")
      .getOrCreate()
  }

  lazy val sc: SparkContext = spark.sparkContext

}