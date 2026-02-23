(statistics_model_userguide)=

# Statistics

Statistics are distinct from prognostic and diagnostic models in principle because
we assume that statistics reduce existing coordinates so that the output tensors
have a coordinate system that is a subset of the input coordinate system. This
makes statistics less flexible than diagnostic models but have fewer API requirements.

## Statistics Interface

Statistics API only specifies a {func}`__call__` method that matches similar methods
across the package.

```{literalinclude} ../../../earth2studio/statistics/base.py
:lines: 24-43
:language: python
```

The base API hints at, and inspection of the {mod}`earth2studio.statistics.moments`
examples, the use of a few properties to make statistic handling easier:

* `reduction_dimensions`, which are a list of dimensions that will be reduced over
* `weights`, which must be broadcastable with `reduction_dimensions`
* `batch_update`, which is useful for applying statistics when data comes in streams and batches

Where applicable, specified `reduction_dimensions` set a requirement for the
coordinates passed in the call method.

## Custom Statistics

Integrating your own statistics is easy, just satisfy the interface above. We recommend
users look at the custom statistic example in the {ref}`extension_examples` examples.

# Metrics

Like statistics, metrics are reductions across existing dimensions. Unlike statistics,
which are usually defined over a single input, we define metrics to take a pair of
inputs. Otherwise, the API and requirements are similar to the statistics requirements.

## Metrics Interface

```{literalinclude} ../../../earth2studio/statistics/base.py
    :lines: 52-
    :language: python
```

## Contributing Statistics and Metrics

Want to add your own statistics or metrics to the package? Great, we will be happy to
work with you. At the minimum we expect the model to abide by the interfaces defined
above. We may also work with the user to ensure that there are `reduction_dimensions`
applicable and, if possible, weight and batching support possible.
