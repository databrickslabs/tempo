"""
Tests for TSDF time-based filtering methods to improve code coverage.
Targets at, before, after, atOrBefore, atOrAfter, between, earliest, latest.
"""

import unittest
from datetime import datetime

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFTimeFilteringTests(SparkTest):
    """Test TSDF time-based filtering methods for better coverage."""

    def test_earliest(self):
        """Test earliest() method to get first n records per series"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.earliest(2)

        collected = result.df.orderBy("timestamp").collect()
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0]["price"], 100.0)
        self.assertEqual(collected[1]["price"], 101.0)

    def test_earliest_single(self):
        """Test earliest() method with n=1"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.earliest(1)

        self.assertEqual(result.df.count(), 1)
        collected = result.df.collect()
        self.assertEqual(collected[0]["price"], 100.0)

    def test_latest(self):
        """Test latest() method to get last n records per series"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.latest(2)

        collected = result.df.orderBy("timestamp").collect()
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0]["price"], 102.0)
        self.assertEqual(collected[1]["price"], 103.0)

    def test_latest_single(self):
        """Test latest() method with n=1"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.latest(1)

        self.assertEqual(result.df.count(), 1)
        collected = result.df.collect()
        self.assertEqual(collected[0]["price"], 103.0)


class TSDFEarliestLatestTests(SparkTest):
    """Test earliest and latest methods with multiple series."""

    def test_earliest_multiple_series(self):
        """Test earliest() with multiple series"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.earliest(2)

        self.assertEqual(result.df.count(), 4)

        x_data = result.df.filter("ticker = 'X'").orderBy("timestamp").collect()
        self.assertEqual(len(x_data), 2)
        self.assertEqual(x_data[0]["value"], 10.0)
        self.assertEqual(x_data[1]["value"], 11.0)

        y_data = result.df.filter("ticker = 'Y'").orderBy("timestamp").collect()
        self.assertEqual(len(y_data), 2)
        self.assertEqual(y_data[0]["value"], 20.0)
        self.assertEqual(y_data[1]["value"], 21.0)

    def test_latest_multiple_series(self):
        """Test latest() with multiple series"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.latest(2)

        self.assertEqual(result.df.count(), 4)

        x_data = result.df.filter("ticker = 'X'").orderBy("timestamp").collect()
        self.assertEqual(len(x_data), 2)
        self.assertEqual(x_data[0]["value"], 13.0)
        self.assertEqual(x_data[1]["value"], 14.0)

        y_data = result.df.filter("ticker = 'Y'").orderBy("timestamp").collect()
        self.assertEqual(len(y_data), 2)
        self.assertEqual(y_data[0]["value"], 21.0)
        self.assertEqual(y_data[1]["value"], 22.0)


if __name__ == "__main__":
    unittest.main()
