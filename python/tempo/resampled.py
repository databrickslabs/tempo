from __future__ import annotations

from typing import Callable, List, Optional, Union


class ResampledTSDF:
    """
    Restricted wrapper around a TSDF that has been resampled.

    Like Spark's GroupedData, this object only exposes operations that are
    valid after a resample: interpolate() and as_tsdf(). Arbitrary DataFrame
    transformations (filter, withColumn, etc.) are intentionally blocked to
    prevent silent invalidation of resample context.
    """

    def __init__(
        self,
        tsdf: TSDF,
        resample_freq: str,
        resample_func: Union[Callable, str],
    ) -> None:
        self._tsdf = tsdf
        self._resample_freq = resample_freq
        self._resample_func = resample_func

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def df(self):
        """The underlying Spark DataFrame."""
        return self._tsdf.df

    @property
    def ts_col(self) -> str:
        return self._tsdf.ts_col

    @property
    def series_ids(self) -> list:
        return self._tsdf.series_ids

    @property
    def ts_schema(self):
        return self._tsdf.ts_schema

    @property
    def columns(self) -> list:
        return self._tsdf.columns

    @property
    def resample_freq(self) -> str:
        return self._resample_freq

    @property
    def resample_func(self) -> Union[Callable, str]:
        return self._resample_func

    # ------------------------------------------------------------------
    # Valid post-resample operations
    # ------------------------------------------------------------------

    def interpolate(
        self,
        method: str,
        target_cols: Optional[List[str]] = None,
        show_interpolated: bool = False,
    ) -> TSDF:
        """
        Interpolate missing values produced by the resample step.

        :param method: interpolation method — "linear", "zero", "null", "bfill", or "ffill"
        :param target_cols: columns to interpolate (default: all non-key columns)
        :param show_interpolated: if True, add a column indicating interpolated rows
        :return: a plain TSDF with interpolated data
        """
        import logging

        import pandas as pd

        from tempo.interpol import backward_fill, forward_fill, zero_fill
        from tempo.interpol import interpolate as interpol_func

        logger = logging.getLogger(__name__)

        tsdf = self._tsdf

        # Resolve target columns
        if target_cols is None:
            prohibited_cols: List[str] = tsdf.series_ids + [tsdf.ts_col]
            target_cols = [col for col in tsdf.df.columns if col not in prohibited_cols]

        # Map method name to interpolation function
        fn: Union[str, Callable[[pd.Series], pd.Series]]
        if method == "linear":
            fn = "linear"
        elif method == "null":
            return tsdf
        elif method == "zero":
            fn = zero_fill
        elif method == "bfill":
            fn = backward_fill
        elif method == "ffill":
            fn = forward_fill
        else:
            fn = method

        interpolated_tsdf = interpol_func(
            tsdf=tsdf,
            cols=target_cols,
            fn=fn,
            leading_margin=2,
            lagging_margin=2,
        )

        if show_interpolated:
            logger.warning(
                "show_interpolated=True is not yet implemented in the refactored version"
            )

        return interpolated_tsdf

    def as_tsdf(self) -> TSDF:
        """Return the underlying TSDF without resample metadata (explicit escape hatch)."""
        return self._tsdf

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Delegates to the internal TSDF's show()."""
        self._tsdf.df.show(n, truncate)

    def __repr__(self) -> str:
        return (
            f"ResampledTSDF(freq={self._resample_freq!r}, "
            f"func={self._resample_func!r}, "
            f"ts_col={self.ts_col!r}, "
            f"series_ids={self.series_ids!r})"
        )
