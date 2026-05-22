# Analysing and Plotting TC Tracks

## Notebooks

- **`plot_tracks_n_fields_notebook.py`** – Plotting tracks and fields for
  individual ensemble members
- **`tracks_slayground_notebook.py`** – Analysing and plotting complete
  tracks from a full ensemble run for a case study on a given storm

Both scripts are [JupyText](https://jupytext.readthedocs.io/) Python files
that can be run directly or converted to Jupyter notebooks:

```bash
# Convert to a Jupyter notebook
jupytext --to notebook plot_tracks_n_fields_notebook.py
jupytext --to notebook tracks_slayground_notebook.py
```

> [!Note]
> The notebooks and the modules in this directory use bare module names
> (`from analyse_n_plot import ...`, `from plotting_helpers import ...`).
> Run them with `plotting/` as the working directory so Python can resolve
> those imports:
>
> ```bash
> cd recipes/tc_tracking/plotting
> jupyter notebook tracks_slayground.ipynb
> ```

## Scripts and Library Modules

- **`analyse_n_plot.py`** – Batch entry point. Drives
  `analyse_individual_storms` (one plot set per storm) and
  `analyse_ensemble_of_storms` (error metrics aggregated across many
  storms). Run with `python analyse_n_plot.py` after editing the storm
  selection and paths near the bottom of the file.
- **`data_handling.py`** – Library: track ingestion, matching against the
  reference, ensemble averaging on the sphere, and lead-time error
  metrics. Imported by the notebooks; not intended to be run directly.
- **`plotting_helpers.py`** – Library: the individual plotting routines
  (spaghetti, intensities over time, histograms, error metrics). Also
  imported by the notebooks; not intended to be run directly.

## Additional Information

- Each notebook specifies at the beginning what data is required and how to
  produce it using the TC tracking pipeline.
- All plotting and analysis routines take a `time_step` keyword argument
  defaulting to 6 h, matching the stock FCN3 and AIFS-ENS configurations.
  Override it if you run the upstream pipeline at a different cadence.
