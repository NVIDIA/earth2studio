# Utilities

## Coordinate Utilities

A collection of utilities to manipulate and check coordinate systems dictionaries.

<!-- e2s-autosummary
currentmodule: earth2studio
template: function
output: generated/utils/1
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
currentmodule: earth2studio
template: class
output: generated/utils/2
-->

{% autosummary %}
earth2studio.utils.interp.LatLonInterpolation
{% endautosummary %}

## Observation Utilities

<!-- e2s-autosummary
currentmodule: earth2studio
template: class
output: generated/utils/3
-->

{% autosummary %}
earth2studio.utils.obs.ObsGridMapping
{% endautosummary %}

## Time Utilities

<!-- e2s-autosummary
currentmodule: earth2studio
template: function
output: generated/utils/4
-->

{% autosummary %}
earth2studio.utils.time.timearray_to_datetime
earth2studio.utils.time.to_time_array
{% endautosummary %}

## Checkpoint Classes

<!-- e2s-autosummary
currentmodule: earth2studio
template: class
output: generated/utils/5
-->

{% autosummary %}
earth2studio.utils.checkpoint.Checkpoint
earth2studio.utils.checkpoint.CheckpointSession
earth2studio.utils.checkpoint.CheckpointState
earth2studio.utils.checkpoint.NullCheckpoint
{% endautosummary %}

## Checkpoint Helpers

<!-- e2s-autosummary
currentmodule: earth2studio
template: function
output: generated/utils/6
-->

{% autosummary %}
earth2studio.utils.checkpoint.bind_checkpoint_state
{% endautosummary %}

## Data Utilities

<!-- e2s-autosummary
currentmodule: earth2studio
template: function
output: generated/utils/7
-->

{% autosummary %}
earth2studio.data.datasource_to_file
earth2studio.data.fetch_data
earth2studio.data.prep_data_array
{% endautosummary %}

## Model Utilities

<!-- e2s-autosummary
currentmodule: earth2studio.models
template: class
output: generated/utils/8
-->

{% autosummary %}
earth2studio.models.auto.Package
earth2studio.models.batch.batch_func
{% endautosummary %}
