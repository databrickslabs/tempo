Resampling
==========

.. automodule:: tempo.resample
   :members:
   :undoc-members:
   :show-inheritance:

ResampledTSDF
-------------

Calling :meth:`tempo.tsdf.TSDF.resample` returns a :class:`~tempo.resample_result.ResampledTSDF`,
a restricted view that only exposes operations valid immediately after a
resample (``interpolate``, ``as_tsdf``, ``show``).

.. automodule:: tempo.resample_result
   :members:
   :undoc-members:
   :show-inheritance:
