# Adaptive Model Domains

Some models can run on a selected geographic region rather than only their default
domain. Models implementing `DomainAdaptable` provide `with_region(region)`, which
returns a new model view sharing the loaded weights and statics. The original model
and other views retain their own domains.

## Select a Region

The `Region` object describes a geographic request independently of any model grid.
Choose a shape supported by your model:

| Constructor | Geographic request |
| --- | --- |
| `Region.box(west, south, east, north)` | Longitude/latitude bounds in degrees. |
| `Region.band(south, north)` | Latitude band spanning all longitudes. |
| `Region.circle(lat, lon, radius_m)` | Center and geodesic radius in metres. |
| `Region.rectangle(lat, lon, half_width_m, half_height_m, bearing_deg)` | Oriented rectangle. |

Rectangle sizes are half extents in metres; omitting `half_height_m` makes a square.
The bearing is clockwise from north to the rectangle's positive y axis. A box with
`west > east` crosses the antimeridian; use a band for full-longitude coverage.

Given a loaded model that supports box requests:

```python
from earth2studio.grids import Region, infer_grid
from earth2studio.models.base import DomainAdaptable

if not isinstance(model, DomainAdaptable):
    raise TypeError("This model does not support regional views")

region = Region.box(west=-105.0, south=35.0, east=-100.0, north=40.0)
regional_model = model.with_region(region)

# For a single-input model, inspect the selected input domain.
input_signature = regional_model.input_coords()
input_grid = infer_grid(input_signature)
output_signature = regional_model.output_coords(input_signature)
```

## Use the Selected Domain

The region is a request, not the final grid. The model selects compatible input,
forcing, and output grids, which may include padding or a larger footprint. Inspect
the returned view's coordinate signatures for the actual domain, and prepare inputs
against those signatures. For example, pass `input_grid` as `target_grid` to
`fetch_data`; multi-input models require preparing each slot separately.

Unsupported shapes or regions that cannot be covered raise `ValueError`. Coverage
describes geometry, not forecast skill outside a model's training region. Select
the region before initializing a rollout; start a new trajectory when changing
domains rather than reusing incompatible domain-dependent state. Views can run
sequentially; concurrent execution depends on the wrapper's documented support.

See [Data and Metadata](../components/coordinates.md#spatial-grids) for grid objects
and [the model domain contract](../../specs/E2S-06_MODEL_CONTRACT_SPEC.md#domain-selection)
for the full interface requirements.
