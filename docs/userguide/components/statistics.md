# Statistics and Metrics { #statistics_model_userguide }

Use statistics to reduce data over dimensions (for example, computing means or variances).
Use metrics to compare two inputs (for example, correlation or bias between forecast and
observation).

## Statistics

Statistics are distinct from prognostic and diagnostic models in principle because
we assume that statistics reduce existing dimensions so that the output DataArray
has a coordinate system that is a subset of the input coordinate system. This
makes statistics less flexible than diagnostic models while having fewer API requirements.

In this section, "statistic" refers to a single reduction operation; "statistics" refers
to the class of such operations.

### Statistics Interface

Statistics operate on [xarray](https://docs.xarray.dev) DataArrays, matching the
diagnostic model API. A statistic takes a single `xr.DataArray` and returns a
`xr.DataArray` with the reduced dimensions removed. Dimensions are identified by name,
so the order of dimensions in the input does not matter. Computation uses PyTorch,
with `.e2s.to_torch()` for inputs and `from_torch` for outputs. Outputs follow the
configured e2s backend policy: `auto` returns NumPy on CPU and CuPy on CUDA, while
`torch` preserves tensors and autograd history. The `output_coords` method accepts
an allocation-free `CoordinateSystem` signature and returns the signature of the output.

```python
--8<-- "earth2studio/statistics/base.py:statistic-interface"
```

The base API hints at, and inspection of the `earth2studio.statistics.moments`
examples reveals, the use of a few properties to make statistic handling easier:

* `reduction_dimensions`, which are a list of dimensions that will be reduced over
* `weights`, which must be broadcastable with `reduction_dimensions`
* `batch_update`, which is useful for applying statistics when data comes in streams and batches

Where applicable, specified `reduction_dimensions` set a requirement for the
coordinates passed in the call method.

### Custom Statistics

To integrate your own statistic, satisfy the interface above. We recommend
that you review the custom statistic example in [extension examples](../../examples/index.md#extend).

## Metrics

Like statistics, metrics are reductions across existing dimensions. Unlike statistics,
which are usually defined over a single input, we define metrics to take a pair of
inputs `x` and `y`, both `xr.DataArray`s that must share dimension names and coordinates
(apart from an optional ensemble dimension that only the forecast `x` carries).
Otherwise, the API and requirements are similar to the statistics requirements.

### Metrics Interface

```python
--8<-- "earth2studio/statistics/base.py:metric-interface"
```

## Contributing Statistics and Metrics

Want to add your own statistics or metrics to the package? We are happy to
work with you. At the minimum we expect the statistic or metric to abide by the interfaces defined
above. We can also work with you to ensure that there are `reduction_dimensions`
applicable and, if possible, weight and batching support.
