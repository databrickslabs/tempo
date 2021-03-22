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
    * Init environment. This structure alows for multiple calls to "reinit" the environment. Important in the case of
    * autoscaling. When the cluster scales up/down envInit and then check for current cluster cores.
    */
  @transient
  lazy protected val _envInit: Boolean = envInit()

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
  //  sc.setLogLevel("DEBUG")

  def getCoresPerWorker: Int = sc.parallelize("1", 1)
    .map(_ => java.lang.Runtime.getRuntime.availableProcessors).collect()(0)

  def getNumberOfWorkerNodes: Int = sc.statusTracker.getExecutorInfos.length - 1

  def getTotalCores: Int = getCoresPerWorker * getNumberOfWorkerNodes

  def getCoresPerTask: Int = {
    try {
      spark.conf.get("spark.task.cpus").toInt
    }
    catch {
      case _: java.util.NoSuchElementException => 1
    }
  }

  def getParTasks: Int = scala.math.floor(getTotalCores / getCoresPerTask).toInt

  def getDriverCores: Int = java.lang.Runtime.getRuntime.availableProcessors

  /**
    * Set global, cluster details such as cluster cores, driver cores, logLevel, etc.
    * This also provides a simple way to change the logging level throuhgout the package
    * @param logLevel log4j log level
    * @return
    */
  def envInit(logLevel: String = "INFO"): Boolean = {
    sc.setLogLevel(logLevel)
    true
  }
}