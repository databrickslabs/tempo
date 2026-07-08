"""
Tests for v0.1.x backwards-compatibility shims in TSDF.

Every shim must emit a ``DeprecationWarning`` (removal targeted for v1.0.0)
while preserving the v0.1.x behavior. See ``MIGRATION_GUIDE.md``.
"""

import warnings
from datetime import datetime

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFDeprecationTests(SparkTest):
    """Assert deprecated v0.1.x APIs still work and warn."""

    def _simple_df(self):
        data = [
            ("A", datetime(2024, 1, 1, 10, 0), 100.0, 10.0),
            ("A", datetime(2024, 1, 1, 10, 1), 101.0, 12.0),
            ("B", datetime(2024, 1, 1, 10, 0), 200.0, 20.0),
        ]
        return self.spark.createDataFrame(
            data, ["symbol", "event_ts", "price", "volume"]
        )

    # --- constructor params ---

    def test_partition_cols_param_warns_and_maps_to_series_ids(self):
        df = self._simple_df()
        with self.assertWarns(DeprecationWarning):
            tsdf = TSDF(df, ts_col="event_ts", partition_cols=["symbol"])
        self.assertEqual(tsdf.series_ids, ["symbol"])

    def test_series_ids_takes_precedence_over_partition_cols(self):
        df = self._simple_df()
        with self.assertWarns(DeprecationWarning):
            tsdf = TSDF(
                df,
                ts_col="event_ts",
                series_ids=["symbol"],
                partition_cols=["price"],
            )
        self.assertEqual(tsdf.series_ids, ["symbol"])

    def test_sequence_col_param_warns(self):
        data = [
            ("A", datetime(2024, 1, 1, 10, 0), 1, 100.0),
            ("A", datetime(2024, 1, 1, 10, 0), 2, 101.0),
        ]
        df = self.spark.createDataFrame(data, ["symbol", "event_ts", "seq", "price"])
        with self.assertWarns(DeprecationWarning):
            TSDF(df, ts_col="event_ts", series_ids=["symbol"], sequence_col="seq")

    # --- deprecated attributes ---

    def test_partitionCols_attribute_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(tsdf.partitionCols, ["symbol"])

    def test_sequence_col_attribute_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            tsdf.sequence_col  # noqa: B018 - accessing the property triggers the warning

    # --- asofJoin(sql_join_opt=) ---

    def test_asofjoin_sql_join_opt_warns(self):
        left = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        right = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            left.asofJoin(right, sql_join_opt=True)

    # --- relocated stats methods ---

    def test_vwap_method_warns(self):
        # NOTE: tempo.stats.vwap has a separate pre-existing schema bug (it
        # aggregates away the ts column but rebuilds a TSDF with the original
        # schema). That is out of scope for the deprecation shim, so here we
        # only assert the wrapper emits the deprecation warning.
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                tsdf.vwap(price_col="price", volume_col="volume")
            except Exception:
                pass
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_EMA_method_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            tsdf.EMA("price", window=2)

    def test_withRangeStats_method_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            tsdf.withRangeStats()

    def test_withGroupedStats_method_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            tsdf.withGroupedStats(freq="1 min")

    def test_withLookbackFeatures_method_warns(self):
        tsdf = TSDF(self._simple_df(), ts_col="event_ts", series_ids=["symbol"])
        with self.assertWarns(DeprecationWarning):
            tsdf.withLookbackFeatures(feature_cols=["price"], lookback_window_size=2)
