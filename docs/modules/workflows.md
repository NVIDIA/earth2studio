# Workflows

`earth2studio.run`

Built in workflows designed to be a catalyst to help accelerate user defined inference
use cases.

!!! warning
    `deterministic`, `diagnostic`, and `ensemble` are legacy workflows limited to
    simple cases: a single unforced prognostic model and one output store. For forced
    or coupled models, multiple output streams, distributed runs, or resuming
    interrupted work, use [`Pipeline` and runners](../userguide/components/execution.md).

<!-- e2s-autosummary
currentmodule: earth2studio.run
template: function
output: generated/workflows
-->

{% autosummary %}
earth2studio.run.deterministic
earth2studio.run.diagnostic
earth2studio.run.ensemble
earth2studio.run.map_field
{% endautosummary %}
