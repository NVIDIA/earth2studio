<!-- markdownlint-disable MD013 -->
# StormCast CONUS subregion cropping

StormCast runs on the full 1024x1792 HRRR CONUS grid by default. Cropping inference to a smaller
region was about 6.6x faster in one Omaha measurement, but this is not a generally reproducible
benchmark. Cropping is opt-in because it has only been validated for one case.

## Configuration

The connector automatically centers the crop on the configured sites when `center` is omitted:

```yaml
subregion:
  enabled: true
  center: {lat: 41.26, lon: -95.94}   # omit to use the mean of `sites`
  size_cells: [512, 640]              # [height, width] in ~3 km grid cells
```

The connector maps the center to the nearest HRRR cell, snaps and clamps the patch to the model
grid, and validates it before loading. It raises an error if:

- the center is outside CONUS;
- a configured site falls outside the crop;
- either dimension is below 128 cells; or
- the crop is too near the northern edge.

Each dimension must be a positive multiple of 4 and no larger than the full domain. The tested
`[512, 640]` size covers roughly 1500 x 1900 km.

## Northern-edge limitation

The model uses absolute positional-embedding indices. A crop whose upper latitude index exceeds
about 1024 overflows that embedding and is rejected. Sites within a few hundred kilometers of the
northern CONUS edge may therefore require the full domain. Latitude crops also have a fixed
four-token positional-embedding shift.

## Explicit grid limits

Advanced users can set both `hrrr_lat_lim` and `hrrr_lon_lim` instead of `center` and `size_cells`.
The limits must remain within `lat in [17, 1041]` and `lon in [3, 1795]`. Additionally,
`lat0 - 17`, `lat1 - lat0`, `lon0 - 3`, and `lon1 - lon0` must all be divisible by 4.

## Quality caveat

No material difference from full CONUS was detected in one Omaha comparison, but that is not a
guarantee. Validate paired seeds and multiple initialization times against full-domain forecasts
before relying on cropping at a new site. When unsure, keep `subregion.enabled: false`.
