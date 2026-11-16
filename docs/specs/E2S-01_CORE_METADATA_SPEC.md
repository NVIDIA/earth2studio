<!-- markdownlint-configure-file { "MD013": { "tables": false } } -->

# E2S-01: Earth2Studio Core Metadata and Coordinate Systems

## Purpose

Define the metadata that gives field values and tabular records their meaning.
Coordinate systems and frame schemas let components declare, compare, and plan
inputs and outputs without allocating or reading the data itself.

## Contract

This contract covers coordinate objects, axis names and types, shared attributes,
and tabular schemas. [Grids](E2S-02_GRID_SPEC.md) owns spatial geometry, CRS,
topology, and grid identity. [Time statistics](E2S-03_TIME_STATISTICS_SPEC.md) owns
temporal quantity modifiers, aggregation windows, and their derived metadata.
[Models](E2S-06_MODEL_CONTRACT_SPEC.md) defines model-specific coordinate matching
and execution; [Data fetching](E2S-05_FETCH_DATA_SPEC.md) defines source requests.

### Coordinate object

`CoordinateSystem` is a metadata-only coordinate signature with no field data.
It is constructed by `coord_array()` as an `xarray.DataArray` whose backing array
is a shape-and-dtype-only placeholder. The placeholder stores no field values,
reports zero field-data bytes, and rejects conversion to a NumPy array. Indexing
and transposing it update the declared shape without allocating field values.

Coordinate labels themselves are real arrays and do occupy memory. An execution
DataArray carries actual field values with the same metadata structure. The legacy
`CoordSystem` ordered mapping of axis names to NumPy arrays cannot express the full
signature and is not an equivalent object.

| ID | Requirement |
| --- | --- |
| C1 | A coordinate signature MUST expose ordered dimensions through `.dims`, lengths through `.sizes`, labels through `.coords`, the declared field dtype through `.dtype`, and shared metadata through `.attrs`. Planning MUST NOT read or materialize field values; signature `.values` is unsupported. |
| C2 | Dimension names MUST be unique. Dimension order MUST describe field axis order; consumers MUST NOT silently sort, rename, align, or transpose axes when checking a signature. |
| C3 | A dimension coordinate MUST be one-dimensional on its namesake dimension and match its size. Auxiliary coordinates MAY be scalar or span multiple declared dimensions; their shape MUST agree with those dimensions. An unlabelled dimension MUST still have a declared size. |
| C4 | Dynamic dimensions MUST be explicitly listed in `earth2studio_dynamic_dims`, form a leading prefix in dimension order, and have size zero in the signature. A fixed zero-sized dimension MUST NOT be interpreted as a wildcard. |
| C5 | Resolving dynamic dimensions MUST leave any remaining dynamic dimensions as a leading prefix. Concrete execution inputs MUST satisfy the consuming component's size and nonempty requirements. |
| C6 | Metadata operations MUST preserve unaffected coordinate values, order, attributes, name, and field dtype. Replacing an axis MUST remove or recompute dependent auxiliary coordinates and derived metadata so they cannot describe the old labels. |

### Axis names and data types

Known names have the meanings below wherever they occur. These are coordinate
types, independent of the field dtype. `coord_array()` defaults the field dtype
to `float32`; this does not require every field or coordinate to be `float32`.

| Name | Coordinate dtype | Meaning |
| --- | --- | --- |
| `time` | NumPy `datetime64` | Absolute time; initialization/reference time when paired with forecast leads, otherwise valid time. UTC semantics without a timezone-bearing object dtype |
| `lead_time` | NumPy `timedelta64` | Offset from reference time; may include input history. Negative offsets are permitted where the consuming contract allows history |
| `valid_time` | NumPy `datetime64` | Auxiliary absolute validity time; when derived from a forecast, equals `time + lead_time` with the corresponding coordinate dimensions |
| `variable` | String labels | Physical quantity identifier from the Earth2Studio vocabulary or a documented extension; temporal qualifiers follow the time-statistics contract |
| `ensemble` | Consistent declared dtype, commonly integers or strings | Member identity; labels identify members rather than their array positions |
| `batch` | Declared label dtype, when labelled | Generic collection of independent samples. May be unlabelled; when labelled, retain the declared label dtype and order |
| `lat`, `lon` | Floating-point | Geographic coordinates in degrees, as axes or auxiliaries; dimensionality, ordering, and longitude convention follow the grid contract |
| `y`, `x` | Numeric coordinates or integer indexes | Native spatial axes in the grid's native units, or integer indexes for indexed layouts |
| `hpx`, `face`, `height`, `width` | Integer indexes | HEALPix pixel or layout indexes; representation and ordering follow the grid contract |

| ID | Requirement |
| --- | --- |
| A1 | Known axis names MUST retain the meanings and dtype families above. Custom axes MAY be added with documented meaning, dtype, and ordering. Renamed spatial axes MUST remain identifiable through grid metadata. |
| A2 | Concrete temporal labels used for temporal validation MUST be explicit, nonempty, and free of `NaT`. Numeric offsets MUST NOT substitute for `timedelta64`. Precision conversion MUST preserve values and avoid overflow; nanosecond precision is not mandatory for all coordinate arrays. |
| A3 | `variable` MUST be a one-dimensional coordinate with unique, nonempty string labels. Labels MUST identify the quantity and its units consistently across producers and consumers. |
| A4 | Coordinate order MUST be preserved. Uniqueness or monotonicity beyond `variable` is imposed by the consuming contract, not assumed globally; repeated requested times, for example, may repeat labels. |
| A5 | A source's `member` argument selects the `ensemble` axis; it MUST NOT introduce a second standard axis named `member`. |

### Standard attributes

Signature attributes live on the DataArray; descriptive attributes may also live
on individual coordinates. A name or descriptive attribute does not replace
coordinate labels or establish axis order.

| Attribute | Location | Type and role |
| --- | --- | --- |
| `earth2studio_kind` | Signature | String `"coordinate_array"`; identifies a coordinate signature |
| `earth2studio_schema_version` | Signature | Integer `1`; version of the signature metadata format |
| `earth2studio_dynamic_dims` | Signature | Ordered tuple of dimension names; empty when all dimensions are fixed |
| `units` | Coordinate or homogeneous field | String describing the units of its values |
| `source` | Field or DataFrame | Producer/provenance identifier string; distinct from a DataFrame's per-row `source` column |

| ID | Requirement |
| --- | --- |
| M1 | Signature construction MUST set kind, schema version, and dynamic dimensions consistently with the object. Concrete field outputs MUST omit these three signature-only attributes. |
| M2 | Descriptive attributes are optional. When present, they MUST describe the associated values accurately. A single field-level `units` MUST NOT claim one meaning for a mixed-variable field; variable-dependent descriptions belong on coordinates indexed by `variable`. |
| M3 | Metadata-preserving transformations MUST retain unrelated user attributes and coordinate attributes. Attributes invalidated by a transformation MUST be updated or removed. |
| M4 | Grid identity/CRS attributes and `earth2studio_statistics` MUST obey their owning specs. They MUST NOT be independently inferred from field shape or copied onto incompatible coordinates. |

### DataFrame schema objects

A DataFrame represents rows of observations, forecasts, or analyses rather than
a dense Cartesian product of axes. Column names carry meaning; its row index is
not a replacement for `time`, location, or variable columns. CPU frames use
pandas; GPU-capable interfaces may use cuDF with the same logical schema.

Two declarations serve different roles:

- **`FrameSchema`** is an ordered mapping from column names to one-dimensional
  NumPy arrays. An empty array declares a column's dtype with dynamic values;
  a nonempty array declares acceptable values, such as supported variable labels.
  The arrays are column constraints, not equal-length sample rows.
- **PyArrow `Schema`** describes source fields by name, type, nullability, and
  optional field metadata. A source exposes its schema as `SCHEMA`.
  `E2STUDIO_SCHEMA` supplies the shared field vocabulary and suggested types.

| ID | Requirement |
| --- | --- |
| F1 | Schema column names MUST be unique. A `FrameSchema` MUST declare required columns and their dtype/value constraints; a consumer MUST validate those requirements before using the rows. An empty constraint array MUST NOT require an empty DataFrame. |
| F2 | A source schema MUST declare each supported field's type and nullability. Standard field names MUST retain the meanings below; sources SHOULD use the suggested types, documenting deviations. Additional source-specific fields MAY be declared. |
| F3 | Field selection MUST preserve requested column order. `fields=None` selects the full source schema; a name or list selects those fields; a PyArrow schema selects fields with matching source types. Unknown fields MUST raise `KeyError`; incompatible requested Arrow types MUST raise `TypeError`. |
| F4 | Returned columns MUST satisfy the selected source schema, including on an empty result. Null values are permitted only for nullable fields. A projected subset need not contain all core columns, but MUST be checked against a consuming `FrameSchema` before execution. |
| F5 | Field metadata MAY include `description`, `units`, and `source_name`; `source_name` describes the upstream name and MUST NOT replace the public column name. Source frames MUST identify the producer in `frame.attrs["source"]`. |
| F6 | Forecast frame schemas MUST declare `lead_time` as a duration when exposing forecast offsets, with `time` interpreted as reference time. If `valid_time` is exposed, it MUST equal `time + lead_time`. |

### Standard DataFrame fields

These tables define a vocabulary, not a requirement that every frame contain
every column. The source schema and consuming `FrameSchema` determine presence.
The shared Arrow schema permits nulls in all listed fields; a source or consumer
may impose stricter constraints. Physical units of `observation` follow `variable`.

| Name | Suggested Arrow type | Meaning |
| --- | --- | --- |
| `time` | `timestamp[ns]` | Observation time, or forecast reference time; UTC |
| `lat` | `float32` | Latitude in degrees, within [-90, 90] |
| `lon` | `float32` | Longitude in degrees, within [0, 360] |
| `elev` | `float32` | Observation elevation above the surface, metres |
| `observation` | `float32` | Measurement or predicted value of the identified variable |
| `variable` | `string` | Earth2Studio quantity identifier |
| `pres` | `float32` | Observation pressure level, Pa |
| `type`, `class` | `string` | Observation type identifier and class |
| `source`, `station` | `string` | Per-row source and station identifiers |
| `station_elev` | `float32` | Station elevation, metres |
| `quality` | `uint16` | Quality-control marker; 0 is best, 15 is missing |
| `satellite` | `string` | Satellite platform identifier |
| `scan_angle` | `float32` | Satellite scan angle, degrees |
| `sensor_index` | `uint16` | Sensor channel index |
| `solza`, `solaza` | `float32` | Solar zenith and azimuth angles, degrees |
| `satellite_za`, `satellite_aza` | `float32` | Satellite zenith and azimuth angles, degrees |
| `wavenumber` | `float64` | Channel wavenumber, cm^-1 |
| `track_id`, `storm_name`, `basin` | `string` | Track identifier, storm name, and basin code |
| `season` | `int32` | Tropical cyclone season year |

Forecast extensions to this shared vocabulary are:

| Name | Suggested Arrow type | Meaning |
| --- | --- | --- |
| `lead_time` | `duration[ns]` | Forecast offset from `time` |
| `valid_time` | `timestamp[ns]` | Optional absolute forecast validity time |

### Verification

- Inspect a signature's dimensions, sizes, labels, dtype, and attributes without
  accessing field values (C1); reject duplicate dimensions or mismatched sizes (C2–C3).
- Resolve a dynamic prefix and check that fixed empty axes remain fixed and a
  non-leading wildcard is rejected (C4–C5).
- Check temporal dtype families, `NaT`, variable uniqueness, and preservation of
  label order, including repeated times (A1–A4).
- Replace a variable or lead-time axis and verify dependent metadata is removed
  or recomputed while unrelated metadata survives (C6, M2–M4).
- Convert a signature to a concrete field and check signature-only attrs are
  absent (M1).
- Validate a frame against dynamic and value-constrained columns; select source
  fields in a different order and exercise unknown names, incompatible types,
  nullability, and empty results (F1–F4).
- Check source provenance and forecast validity-time consistency (F5–F6).
