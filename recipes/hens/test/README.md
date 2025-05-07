# Tests

## Test 1: Reproducibility of Helene Ensemble Members

Run the HENS recipe with first with the `helene.yaml` configuration,
then with the `reproduce_helene_batches.yaml` configuration:

```bash
cd earth2studio/recipes/hens

# run the first recipe
uv run main.py --config-name=helene.yaml

# run the second recipe
uv run main.py --config-name=reproduce_helene_batches.yaml
```

This will produce two directories: `outputs_helene` and `outputs_reprod`.
Compare if tracks produced by `reproduce_helene_batches.yaml` are identical
to tracks produced by the associated ensemble members in `helene.yaml`:

```bash
cd earth2studio/recipes/hens/test
uv run test_reprod.py
```

Note that you can also specify the directories to compare as arguments:

```bash
uv run test_reprod.py --dir1=path_0/to/cyclones --dir2=path_1/to/cyclones
```

To compare individual files, run:

```bash
uv run compare_tracks.py path/to/file0 path/to/file1
```

### Expected Result

When running in default mode, the script will report that all files are identical:

```text
Comparing tracks_pkg_103_2024-09-24T18:00:00_batch_2.nc with tracks_pkg_103_2024-09-24T18:00:00_batch_2.nc:
Files are identical in terms of data values

Comparing tracks_pkg_102_2024-09-24T18:00:00_batch_0.nc with tracks_pkg_102_2024-09-24T18:00:00_batch_0.nc:
Files are identical in terms of data values
```

If the files are not identical, the script will report:

```text
Not all files are identical
```
