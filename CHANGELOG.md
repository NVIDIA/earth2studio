<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-01-03

### Added

- Add StormCast model to prognostic models
- Interpolation between arbitrary lat-lon grids
- Added hybrid level support to HRRR data source
- Added NCAR ERA5 data source

### Changed

- Switched HRRR data source back to AWS grib
- Make source an argument for IFS, default of aws

### Dependencies

- Update NGC SDK dependency to fixed range due to internal API changes
- Updates to multiple dependencies for Python 3.12 support

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
