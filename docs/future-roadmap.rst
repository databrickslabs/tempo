Future Roadmap
==============

The general plan for Tempo's future development is below. However, it is worth noting that based on the demand the order
of development might change.

.. admonition:: Tempo's future plan
   :class: important

   - *Next generation API version 0.2*
    1. Meta-data cleanup

    2. Simpler constructors

    3. New core TSDF transformation functions

   - *Clear separation of resampling & interpolation functions*
   --------------------------------------------------------------------------
   - *Intervals API*
    1. Extract intervals from TSDFs
    2. Get distinct (non-overlapping intervals)
    3. Interval-TSDF joins

   - *Streaming Support* 🚀
    1. Streaming as-of joins
    2. Streaming resampling

   - *Integration with timeseries forecasting & ML libraries:* 📈
    1. statsmodels
    2. FB Prophet
    3. Tensorflow
    4. Pytorch
    5. tsfresh
    6. sktime

    And many more...

   - *Simpler Windowing API*
    1. Use natural-language expressions (ie "5 mins", "16 days", etc.)


Don't see what you are looking for, reach out to the team directly
at `labs@databricks.com <mailto:labs@databricks.com>`_ with your feature requests and ideas for improvement.