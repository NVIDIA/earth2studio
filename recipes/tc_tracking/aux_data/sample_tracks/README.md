# Sample tracks for the Earth2Studio tropical cyclone tracking recipe

Ensemble forecast tracks of Hurricane Melissa and Typhoon Ragasa (2025) from
WeatherNext 2 Cyclones, FourCastNet 3 and AIFS-ENS v1, for the notebook
`wn2_tracker_comparison.py` of
<https://github.com/NVIDIA/earth2studio/tree/main/recipes/tc_tracking>.

## Contents

| Directory | Run | Track files |
| --- | --- | --- |
| `<storm>_wn2/cyclone_tracks_wn2` | WN2 built-in tracker | 32 |
| `<storm>_wn2/cyclone_tracks_te` | TempestExtremes on WN2 | 32 |
| `<storm>_fcn3/cyclone_tracks_te` | TempestExtremes on FourCastNet 3 | 32 |
| `<storm>_aifs/cyclone_tracks_te` | TempestExtremes on AIFS-ENS v1 | 32 |
| `reference_tracks_ifs` | TempestExtremes on the IFS analysis | 1 per storm |

The file format is described in section 6.2 of the recipe's README.

## How They Were Produced

- Forecasts: `cfg/melissa_wn2.yaml` and `cfg/ragasa_wn2.yaml`, for FourCastNet 3
  and AIFS-ENS v1 with the overrides of section 6.1 of the recipe's README.
  32 members, 40 steps of 6 h (10 days), initial conditions from the IFS
  analysis (ECMWF open data), sea surface temperature for WN2 from ERA5.
- Reference tracks: `cfg/extract_ifs.yaml`. The columns `extract_baseline`
  copies from IBTrACS were removed, as IBTrACS is not redistributed here.
- Forecasts run on 2026-11-26 on one node with 8 NVIDIA H100 GPUs, in the
  recipe's container (Earth2Studio 1.1.0a0), with the recipe code of commit
  `d0a184d2`; reference tracks on 2026-11-25, on CPU. Later versions of the
  recipe write the same format.

WN2 runs are not bitwise reproducible on GPU, so a rerun gives slightly
different tracks.

## Credits and Licence

The models, their creators, the input data and the required attributions are
listed in section 6.4 of the recipe's README. Contains no IBTrACS data. Released
under CC BY 4.0. Based on data and products of the European Centre for
Medium-Range Weather Forecasts (ECMWF); contains modified Copernicus Climate
Change Service information.
