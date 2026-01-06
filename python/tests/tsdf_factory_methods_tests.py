"""
Tests for TSDF factory methods to improve code coverage.
Targets buildEmptyLattice and other factory method code paths.
"""

import unittest
from datetime import datetime, timedelta

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFFactoryMethodsTests(SparkTest):
    """Test TSDF factory methods for better coverage."""

    def test_build_empty_lattice_basic(self):
        """Test buildEmptyLattice with minimal parameters"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 11, 0, 0)
        step = timedelta(minutes=15)

        lattice = TSDF.buildEmptyLattice(
            self.spark, start_time=start, end_time=end, step_size=step
        )

        self.assertEqual(lattice.df.count(), 4)
        self.assertIn("ts_idx", lattice.df.columns)

    def test_build_empty_lattice_with_custom_ts_col(self):
        """Test buildEmptyLattice with custom timestamp column name"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=10)
        num_intervals = 5

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            ts_col="custom_ts",
        )

        self.assertEqual(lattice.df.count(), 5)
        self.assertIn("custom_ts", lattice.df.columns)
        self.assertEqual(lattice.ts_col, "custom_ts")

    def test_build_empty_lattice_with_series_ids_dict(self):
        """Test buildEmptyLattice with series_ids as dict"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=30)
        num_intervals = 3

        series_dict = {"symbol": ["A", "B", "C"]}

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            series_ids=series_dict,
        )

        self.assertEqual(lattice.df.count(), 9)
        self.assertIn("symbol", lattice.df.columns)
        self.assertEqual(lattice.series_ids, ["symbol"])

    def test_build_empty_lattice_with_series_ids_dataframe(self):
        """Test buildEmptyLattice with series_ids as DataFrame"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(hours=1)
        num_intervals = 2

        series_data = [("A",), ("B",)]
        series_df = self.spark.createDataFrame(series_data, ["ticker"])

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            series_ids=series_df,
        )

        self.assertEqual(lattice.df.count(), 4)
        self.assertIn("ticker", lattice.df.columns)
        self.assertEqual(lattice.series_ids, ["ticker"])

    def test_build_empty_lattice_with_observation_cols_list(self):
        """Test buildEmptyLattice with observation_cols as list"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=20)
        num_intervals = 3

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            observation_cols=["price", "volume"],
        )

        self.assertEqual(lattice.df.count(), 3)
        self.assertIn("price", lattice.df.columns)
        self.assertIn("volume", lattice.df.columns)

        first_row = lattice.df.first()
        self.assertIsNone(first_row["price"])
        self.assertIsNone(first_row["volume"])

    def test_build_empty_lattice_with_observation_cols_dict(self):
        """Test buildEmptyLattice with observation_cols as dict with types"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=15)
        num_intervals = 2

        obs_cols = {"price": "double", "count": "int"}

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            observation_cols=obs_cols,
        )

        self.assertEqual(lattice.df.count(), 2)
        self.assertIn("price", lattice.df.columns)
        self.assertIn("count", lattice.df.columns)

    def test_build_empty_lattice_with_series_and_observations(self):
        """Test buildEmptyLattice with both series_ids and observation_cols"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=30)
        num_intervals = 2

        series_dict = {"symbol": ["X", "Y"]}
        obs_cols = ["price", "volume"]

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            series_ids=series_dict,
            observation_cols=obs_cols,
        )

        self.assertEqual(lattice.df.count(), 4)
        self.assertIn("symbol", lattice.df.columns)
        self.assertIn("price", lattice.df.columns)
        self.assertIn("volume", lattice.df.columns)

    def test_build_empty_lattice_with_num_partitions(self):
        """Test buildEmptyLattice with custom num_partitions"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=10)
        num_intervals = 10

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            num_partitions=2,
        )

        self.assertEqual(lattice.df.count(), 10)
        self.assertEqual(lattice.df.rdd.getNumPartitions(), 2)

    def test_build_empty_lattice_with_series_and_partitions(self):
        """Test buildEmptyLattice with series_ids and num_partitions"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=15)
        num_intervals = 3

        series_dict = {"id": ["A", "B"]}

        lattice = TSDF.buildEmptyLattice(
            self.spark,
            start_time=start,
            step_size=step,
            num_intervals=num_intervals,
            series_ids=series_dict,
            num_partitions=2,
        )

        self.assertEqual(lattice.df.count(), 6)
        self.assertEqual(lattice.df.rdd.getNumPartitions(), 2)


if __name__ == "__main__":
    unittest.main()
