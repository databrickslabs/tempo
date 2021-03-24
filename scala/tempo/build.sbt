name := "tempo"

version := "0.1"

scalaVersion := "2.12.4"

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.11
  "org.apache.spark" %% "spark-core" % "3.0.0",
  // https://mvnrepository.com/artifact/org.apache.spark/spark-sql_2.11
  "org.apache.spark" %% "spark-sql" % "3.0.0",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "io.delta" %% "delta-core" % "0.8.0"
)