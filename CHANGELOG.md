<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-xx-xx

### Added

- Built in diagnostic workflow
- Basic diagnostic example
- Batch dimension userguide
- Parallel inference example
- Perturbation method section in userguide
- WeatherBench Climatology data source
- Added `datasource_to_file` utility function
- Add lagged ensemble perturbation method
- Add ACC and CRPS metrics
- Added the ability to reload zarr and netcdf backends
- Added the ability to read from an IOBackend
- Add spread/skill ratio

### Changed

- Changed utility function `extract_coords` to `split_coords`
- Batched coordinate arrays now use `np.empty(0)` instead of `np.empty(1)`
- Improving user guide layout and developer documentation
- Updated perturbation methods API `PerturbationMethod` -> `Perturbation`.
  These now generate noise and apply it to the input tensor.
- Removed original `Perturbation` class
- Updated SFNO coordinates to include optional batch dimension.
- NetCDF reads are now mode='r+' instead of 'w'.

### Deprecated

### Removed

### Fixed

- Enable version switch in documentation site
- Longitude coordinates of precip and climatenet diagnostic models

### Security

### Dependencies

- Bump Modulus required version to 0.6.0
- PyUpgrade pre-commit hook for Python 3.10

## [0.1.0] - 2024-04-22

### Added

- Initial Release of earth2studio
