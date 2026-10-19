# Utilities

## Coordinate Utilities

A collection of utilities to manipulate and check coordinate systems dictionaries.

<!-- e2s-autosummary
currentmodule: earth2studio.utils.coords
template: function
output: generated/utils/coords
-->

{% autosummary %}
earth2studio.utils.coords.handshake_dim
earth2studio.utils.coords.handshake_coords
earth2studio.utils.coords.handshake_size
earth2studio.utils.coords.map_coords
earth2studio.utils.coords.split_coords
{% endautosummary %}

## Grid Interpolation

<!-- e2s-autosummary
currentmodule: earth2studio.utils.interp
template: class
output: generated/utils/interp
-->

{% autosummary %}
earth2studio.utils.interp.LatLonInterpolation
{% endautosummary %}

## Regridders

Implementations of the `earth2studio.grids.Regridder` protocol, which map fields
between grid definitions.

<!-- e2s-autosummary
currentmodule: earth2studio.grids
template: class
output: generated/grids
-->

{% autosummary %}
earth2studio.grids.BilinearRegridder
earth2studio.grids.NearestRegridder
{% endautosummary %}

<!-- e2s-autosummary
currentmodule: earth2studio.grids
template: function
output: generated/grids
-->

{% autosummary %}
earth2studio.grids.list_regridders
earth2studio.grids.register_regridder
earth2studio.grids.resolve_regridder
{% endautosummary %}

## Data Source Wrappers

Compositions that stand in for a data or forecast source.

<!-- e2s-autosummary
currentmodule: earth2studio.data
template: class
output: generated/data
-->

{% autosummary %}
earth2studio.data.CompositeSource
earth2studio.data.RegriddedSource
{% endautosummary %}

## Observation Utilities

<!-- e2s-autosummary
currentmodule: earth2studio.utils.obs
template: class
output: generated/utils/obs
-->

{% autosummary %}
earth2studio.utils.obs.ObsGridMapping
{% endautosummary %}

## Time Utilities

<!-- e2s-autosummary
currentmodule: earth2studio.utils.time
template: function
output: generated/utils/time
-->

{% autosummary %}
earth2studio.utils.time.timearray_to_datetime
earth2studio.utils.time.to_time_array
{% endautosummary %}

## Checkpoint Classes

<!-- e2s-autosummary
currentmodule: earth2studio.utils.checkpoint
template: class
output: generated/utils/checkpoint
-->

{% autosummary %}
earth2studio.utils.checkpoint.Checkpoint
earth2studio.utils.checkpoint.CheckpointSession
earth2studio.utils.checkpoint.CheckpointState
earth2studio.utils.checkpoint.NullCheckpoint
{% endautosummary %}

## Checkpoint Helpers

<!-- e2s-autosummary
currentmodule: earth2studio.utils.checkpoint
template: function
output: generated/utils/checkpoint
-->

{% autosummary %}
earth2studio.utils.checkpoint.bind_checkpoint_state
{% endautosummary %}

## Data Utilities

<!-- e2s-autosummary
currentmodule: earth2studio.data
template: function
output: generated/data
-->

{% autosummary %}
earth2studio.data.datasource_to_file
earth2studio.data.fetch_data
earth2studio.data.prep_data_array
{% endautosummary %}

## Model Utilities

<!-- e2s-autosummary
currentmodule: earth2studio.models.auto
template: class
output: generated/models/auto
-->

{% autosummary %}
earth2studio.models.auto.Package
{% endautosummary %}

<!-- e2s-autosummary
currentmodule: earth2studio.models.batch
template: function
output: generated/models/batch
-->

{% autosummary %}
earth2studio.models.batch.batch_func
{% endautosummary %}
