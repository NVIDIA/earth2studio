<!-- markdownlint-disable MD013 -->
# Hyperspectral observation frames

Status: proposal, decision needed before code.

## Problem

`DataFrameSource`s return one row per observation. For hyperspectral IR
sounders, every footprint's metadata (time, position, geometry, platform) is
repeated once per channel. On `gdas.20230101.t06z.mtiasi.tm00.bufr_d` (one
6-hour IASI dump, 648,079 footprints x 616 channels), measured with the
earth2bufr decoder in `NNJAObsSat`:

| Layout | Rows | Bytes per row | One file | 9 h window (~1.5 files) |
| --- | --- | --- | --- | --- |
| Long, Arrow | 399M | 90 | ~36 GB | ~54 GB |
| Long, pandas (dictionary strings) | 399M | 67 | ~27 GB | ~40 GB |
| Per footprint, float32 radiances | 0.65M | ~2.5 kB | ~1.6 GB | ~2.4 GB |

CrIS (431 channels, three platforms) and AIRS (281 channels) are similar in
size. Microwave sensors (5-22 channels) and the thinned GSI diagnostic files
`UFSObsSat` reads are not affected.

So a model that assimilates all IASI channels over a multi-hour window cannot
take its observations as an ordinary `fetch_dataframe` result. Consumers end up
iterating long-format batches and reshaping each one back into per-footprint
arrays, for example to project spectra onto principal components. Every row is
expanded and immediately collapsed again.

## Options

1. **Per-footprint rows (recommended).** One row per footprint, with the
   channel-varying fields as list columns: `sensor_index: list<uint16>`,
   `wavenumber: list<float64>`, `observation: list<float32>`, and `quality`
   where the product has per-channel flags. This is the layout earth2bufr
   decodes natively (`DRF16BIT` is already a `list<struct<CHNM, SCRA>>` per
   footprint), so no long-format rows are ever built.
   - Fits in an ordinary DataFrame for a multi-hour window. No new type for
     runners, `fetch_dataframe`, `FrameSchema` or the coupler; time filtering
     and platform filtering act on footprint columns as they do today.
   - Departs from one observation per row for these sensors. Consumers that
     want long format call `DataFrame.explode` on the list columns; we can ship
     a helper.
   - Arrow and cuDF support list columns; pandas holds them as
     `ArrowDtype(list)` without copying.
2. **Streamed frames.** Keep long format, and add a replayable, closable
   iterator of DataFrames that decodes lazily from downloaded files.
   - Bounded memory without changing the row layout.
   - Adds a new type that `fetch_dataframe`, every runner and every
     DataFrame-consuming model must understand, plus lifetime management
     (temporary files, close, replay). The rows are still expanded, only in
     smaller pieces.

## Recommendation

Option 1, opt-in per request rather than a change of default:

- `NNJAObsSat(layout="footprint")` (name to be decided) returns per-footprint
  rows for the IR sounders. Long format stays the default, so existing
  consumers and the microwave sensors are unchanged. A mixed request returns
  one frame per layout, or rejects mixing; to be decided.
- `sensor_indices` keeps selecting channels, now through earth2bufr's
  `ListValueFilter("DRF16BIT", "CHNM", ...)` so unselected channels are never
  materialized.
- List lengths may vary by footprint, since a channel with a missing radiance
  is dropped today. Prefer aligned lists over the file's channel set, with
  nulls for missing values, so consumers can stack `observation` into a
  `(footprint, channel)` array without matching channel numbers per row. The
  channel set is per product era, so it is read from the file, not hard-coded.
- Models declare the layout in their `FrameSchema` slot, with list-typed
  `observation` and `sensor_index` columns, so a mismatched source fails the
  existing handshake rather than at decode time.

Streamed frames stay out of `earth2studio.data`, the runners and the `Source`
type. A model that needs bounded-memory iteration over long format can do it
privately until a real case appears that option 1 does not cover.

## Related: extra observation columns

Some consumers need columns the shared NCEP schema does not carry, such as
source message and subset indices for deduplication, raw radiance next to
brightness temperature, or footprint-level quality. Add them opt-in through
`fields`, not to every source's default output. earth2bufr's integer
`_source_message_index` and `_source_subset_index` identify a report within a
file and are cheaper than hashed string identifiers.

## Open questions

1. Layout parameter name and whether it belongs on `NNJAObsSat` only or on a
   shared NCEP IR helper used by future sources.
2. Mixed long and per-footprint requests in one call.
3. Whether AIRS and CrIS move to earth2bufr first (the IASI path does today;
   they still decode through pybufrkit).
