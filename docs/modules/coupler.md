# Coupler

`earth2studio.coupler`

Run several prognostic models that feed each other (for example a GOES to MRMS
StormScope pair, or an atmosphere and an ocean model with different time steps) as a
single `earth2studio.run` runner. The coupler reads each model's `input_coords`,
`output_coords` and `forcing_coords`, advances all models in coupling windows set by the
slowest model, and publishes one output stream per model. The exchange between models is
a plain function, see `hold` and `window_mean`.

!!! warning
    The coupler is an initial, lightweight implementation. It supports prognostic models
    with one input slot and one output slot with a single lead time; ensembles are one
    member per work item.

<!-- e2s-autosummary
currentmodule: earth2studio.coupler
template: class
output: generated/coupler
-->

{% autosummary %}
earth2studio.coupler.CoupledRunner
earth2studio.coupler.EnsembleCoupledRunner
earth2studio.coupler.RegriddingCoupledRunner
{% endautosummary %}

<!-- e2s-autosummary
currentmodule: earth2studio.coupler
template: function
output: generated/coupler
-->

{% autosummary %}
earth2studio.coupler.hold
earth2studio.coupler.window_mean
{% endautosummary %}
