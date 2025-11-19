<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0a0] - 2025-12-xx

### Added

- Added general PlanetaryComputerData source for pulling Planetary Computer STAC assets

### Changed

- Removed tp06 field from Graphcast operational model
- Removed static fields from Graphcast model input / outputs
- Moved StormCast and DLESyM checkpoints to Huggingface

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [0.10.0] - 2025-11-17

### Added

- Added CMIP6 data source
- Added CBottle Tropical Cyclone guidance diagnostic
- Added CBottle Video prognostic model
- Exposed backend arguments of netcdf/zarr to datasource_to_file signature
- Added vertical wind speed support in GFS
- Added ModelOutputDatasetSource to use written model output to start a new model run
- Added FCN3 noise handling routines
- Added ACE2-ERA5 model and data source
- Added tp06 support to CDS data source
- Added IFS_ENS, AIFS, and AIFS_ENS data sources.
- Added Multi-Storage Client integration into ARCO data source
- Ensemble forecasting with downscaling example

### Changed

- Updated CBottle data source to mixture of experts models, added additional parameters
  to load_model to align with other cBottle models.
- Fixed duplicate geo-potential at surface ids in AIFS, IFS data source and orography
  source, orography is denoted by lower case `z`
- Updated package caching default to None, which will default to true for remote
  packages and false for local packages
- Changed IFS to be a forecast data source.
- InferenceOutputSource can now accept an Xarray Dataset directly as an argument
- InferenceOutputSource returns data consistently in `("time", "lead_time", "variable")`
  order
- Added support for ERA5 model levels and additional variables in ARCO data source
- Changed the HRRR_X and HRRR_Y coordinates of the HRRR data source to match the native
  LCC coordinates
- Updated CorrDiffTaiwan model wrapper to use latest PhysicsNeMo APIs.

### Deprecated

- Added depricated warning for IMERG data source. This will be removed in the next
  release.

### Fixed

- Fixed typo: InferenceOuputSource renamed to InferenceOutputSource
- StormCast ensures that conditioning variables are in the correct order
- NetCDFBackend unit change to ensure timedeltas are correctly decoded by xarray
- GFS data source for early 2021 dates
- Updated corrdiff wrapper for newer physicsnemo performance improvements

### Dependencies

- Dropped support for Python 3.10
- Bumped CBottle (and Earth2Grid) versions
- Capped JAX version, due to numpy 2.0 requirement conflicting with NV PyTorch containers
- Temp limit globus-sdk for intake-esgf
- Added multi-storage client into data dependency group

## [0.9.0] - 2025-08-19

### Added

- Async Zarr IO backend with non-blocking write calls
- Different compression codec support in the ZarrBackend with `zarr_codecs` parameter
- IO performance example
- Unified CorrDiff Wrapper
- Added UV script dependencies to all examples
- New metrics: Brier score, fractions skill score, log spectral distance, mean absolute
  error
- Option to compute error of ensemble mean in rmse and mae
- Added FourCastNet 3 model

### Changed

- Zarr IO Backend now uncompressed by default
- Allow HCBV perturbation to handle constant outputs (like land sea mask, or
  geopotential at surface)
- test/models/dx/test_corrdiff.py is now test/models/dx/test_corrdiff_taiwan.py
- Updated APIs for optional dependency managment utils with improved error messages
- Allow Zarr backends to user datetime and timedelta arrays for Zarr 3.0

### Fixed

- Incorrect datetime utc timezone calculation in SFNO wrapper was fixed.
- DLWP output coords lead_time array to have proper shape
- Fixed data sources using GCFS throwing error at end of script from aiohttp session
  clean up
- Fixed HRRR_FX valid lead time check for date times not on 6 hour interval
- Removed time limits for WB2 climatology data source

### Dependencies

- Adding rich to core dependencies
- Changed torch-harmonics to 0.8.0
- Changed makani to 0.2.1

## [0.8.1] - 2025-07-07

### Changed

- Updated default StormCast package version to 1.0.2

### Fixed

- NGC filesystem from API change in version >=3.158.1 of ngcsdk

### Dependencies

- Removed ngcsdk dependency requirement for public NGC packages

## [0.8.0] - 2025-06-13

### Added

- Added GraphCast operational model (0.25 degree resolution)
- Added Graphcast 1 degree model
- Added SolarRadiationAFNO diagnostic model for predicting surface solar radiation
- Added DataArrayPathList for reading local data using glob patterns or explicit file lists
- Added Climate in a Bottle (cBottle) data source
- Added Climate in a Bottle (cBottle) Infilling diagnostic model
- Added Climate in a Bottle (cBottle) Super Resolution diagnostic model
- Added S2S recipe

### Changed

- In recipes, renamed `requirements.txt` -> `recipe-requirements.txt`

### Fixed

- Fixed NCAR data source lat / lon labels and cache reads
- Fixed FuXi tp06 field input to be mm
- Fixed fsspec async filesystem initialization in data sources
- Fixed bug in GFS_FX forecast source which had lead time fixed at 0

### Dependencies

- Moved NGC SDK to optional dependencies due to it causing slow version resolutions
- Removing upper Python restriction on Rapids install for TC trackers

## [0.7.0] - 2025-05-21

### Added

- Added AIFS model wrapper with state caching functionality for improved performance
- Added two cyclone trackers and related utilities
- Added HENS checkpoint example
- Added Earth2Studio recipes folder, documentation and template
- Added DLESyM and DLESyMLatLon atmosphere and ocean prognostic models
- HENS recipe

### Changed

- Hemispheric centred bred vector perturbation now supports single/odd batch sizes
- Refactored NCAR ERA5 source to have async structure
- Refactored GFS and GFS_FX to have async structure
- Refactored GEFS and GEFS_FX to have async structure
- Refactored HRRR and HRRR_FX to have async structure
- Refactored WB2ERA5 and WB2Climatology for async Zarr 3.0
- Expanded the data source protocol to also include async fetch functions for async
  data sources
- Updated StormCast coords to be HRRR index, output coords still provide lat lon
- Interpolation AFNO model load_model now accepts prognostic model

### Removed

- Removed curvilinear from Random data source

### Fixed

- Fixed the asyncio zarr access in the ARCO data source
- Partially fixed multiple tqdm print outs when using the built in workflows
- Generalized CorrelatedSphericalGaussian to support input tensors of higher dims

### Security

- Remove pickle load from Aurora model with direct numpy array loads

### Dependencies

- Default torch version cuda 12.8

## [0.6.0] - 2025-04-15

### Added

- Hemispheric centred bred vector perturbation from HENS
- Add Aurora model to prognostic models
- Added check_extra_imports util for informative errors when optional dependencies are
  not installed
- Added wind gust AFNO diagnostic model
- Added diagnostic for relative humidity from temperature and specific humidity
- Added diagnostic for relative humidity from temperature and dew point
- Added diagnostic for wind speed magnitude
- Added diagnostic for vapour-pressure deficit
- Added PrecipitationAFNOv2 model for predicting tp06
- Added InterpModAFNO model for temporal interpolation of forecasts
- Python 3.13 support

### Fixed

- Bug in Weather Bench 2 climatology data source with Zarr 3.0

### Dependencies

- Migrated repo / package to uv package manager
- Removed physics-nemo, torch harmonics from base packages to enable CPU install
- Added optional dependency groups for all models
- Added optional dependency groups for other submodules
- Added documentation for build, install and package management for developers
- Migrated build system to hatch
- Moved dev and doc optional dependencies to uv dependency groups

## [0.5.0] - 2025-03-26

### Added

- Add StormCast model to prognostic models
- Interpolation between arbitrary lat-lon grids
- Added hybrid level support to HRRR data source
- Added NCAR ERA5 data source
- Added multidim IO support
- Added forecast data source support to `fetch_data`
- Added stormcast deterministic and ensemble examples
- Added Random_FX as a random forecast data source
- Added interpolation support to run functions
- Added fair CRPS metric
- Added basic coordinate roll support in map_coords

### Changed

- Switched HRRR data source back to AWS grib
- Make source an argument for IFS, default of aws
- Changed CorrDiff output coordinates to actual lat/lon instead of ilat/ilon
- Changed the NetCDF4Backend to use proleptic gregorian calendar for time
- Changed the units assigned from the NetCDF4Backend to hours instead of h

### Fixed

- Fixed bug in prep_data_array that implicitly assumed order of coordinates
- Fixed bug in rank_histogram that assumed broadcastable ensemble dimension
- Fixed spread/skill ratio to actually return spread/skill instead of skill/spread
- Fixed NGC download APIs and public API fetching for model files
- Fixed bug when using HRRR datasource in Jupyter notebooks
- Fixed ARCO for Zarr 3.0 and made proper async running with notebook support
- Fixed WB2 data source for Zarr 3.0 support
- Fixed Zarr IO for Zarr 3.0, for Zarr 3.0 datetime and timedeltas stored as int64
- Fixed CorrDiff and Stormcast for Zarr 3.0 support
- Fixed examples for Zarr 3.0 support updates

### Dependencies

- Updates to multiple dependencies for Python 3.12 support
- Added StormCast to optional dependencies
- Update to physicsnemo version 1.0.0
- Added nest asyncio to data dependencies for async data sources

## [0.4.0] - 2024-12-12

### Added

- Added NCEP data store to GFS data source for real-time forecast apps

### Changed

- Set zarr chunks for lead time to size 1 in examples.
- Updated HRRR tp to be hourly accumulated (Grib index 090)
- Added tp to GFS_FX datasource (not supported by GFS)
- Moved HRRR data source to Zarr datastore on S3

### Removed

- Removed `available` function from CDS datasource

### Dependencies

- Moving several ECMWF dependencies to optional
- Adding minimum version for numpy
- Bump minimum CDS API version for new API
- Moving unique data packages to optional deps
- Removed Herbie as dependency

## [0.3.0] - 2024-09-24

### Added

- Forecast datasource API
- GFS forecast datasource
- GEFS (0.5deg and 0.25deg) forecast datasource
- HRRR forecast datasource
- Support for private NGC model packages

### Changed

- Refactored ARCO datasource to use asyncio
- Updated NetCDF4 and Zarr IO to take kwargs for root storage objects allowing better
  control over storage behavior. Breaking changes to NetCDF4 init API.
- Changed the `da` property to DataSetFile and DataArrayFile to no longer be a property
  and moved xr_args to object instantiator.
- Improved map_coords to handle slices and indentical coords more efficiently; removed
  unused ignore_batch argument.

### Removed

- Removed tp06 from ARCO, use WB2 instead

### Fixed

- Fixed caching of data sources to be controlled with `EARTH2STUDIO_CACHE` env var

### Dependencies

- Restrict torch_harmonics version to >=0.5.0, <0.7.1
- Removed specific ONNX version requirement, newer ORT-gpu versions appear to operate
  fine with CUDA 12

## [0.2.0] - 2024-07-23

### Added

- Built in diagnostic workflow
- Basic diagnostic example
- Batch dimension userguide
- Parallel inference example
- Perturbation method section in userguide
- WeatherBench Climatology and ERA5 data source
- Added `datasource_to_file` utility function
- Add lagged ensemble perturbation method
- Add ACC and CRPS metrics
- Added the ability to reload zarr and netcdf backends
- Added the ability to read from an IOBackend
- Add spread/skill ratio
- Added FuXi weather model
- Added rank histogram
- Added reduction_dimensions as required property in statistics
  and metrics API.
- Added Lexicon and Automodel userguide
- Added an 'output_coords' method to Statistics and Metrics.
- Added IMERG data source

### Changed

- Changed utility function `extract_coords` to `split_coords`
- Batched coordinate arrays now use `np.empty(0)` instead of `np.empty(1)`
- Improving user guide layout and developer documentation
- Updated perturbation methods API `PerturbationMethod` -> `Perturbation`.
  These now generate noise and apply it to the input tensor.
- Removed original `Perturbation` class
- Updated SFNO coordinates to include optional batch dimension.
- NetCDF reads are now mode='r+' instead of 'w'.
- Change 'input_coords' and 'output_coords' for models from a property to methods.
  'output_coords' accepts an input coordinates to determine the corresponding outputs.
- Updated Package to use WholeFileCacheFileSystem. Extend package API to open and
  resolve. Deprication warning added to get.

### Fixed

- Enable version switch in documentation site
- Longitude coordinates of precip and climatenet diagnostic models
- Fixed pressure levels of IFS datasource to include all available

### Dependencies

- Bump Modulus required version to 0.6.0
- PyUpgrade pre-commit hook for Python 3.10
- Removed boto3
- Added ruamel.yaml, torch-harmonics, tensorly and tensorly-torch
  as option deps for SFNO

## [0.1.0] - 2024-04-22

### Added

- Initial Release of earth2studio
