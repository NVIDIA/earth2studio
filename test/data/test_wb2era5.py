@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1959, month=1, day=31),
        [
            datetime.datetime(year=1971, month=6, day=1, hour=6),
            datetime.datetime(year=2021, month=11, day=23, hour=12),
        ],
        np.array([np.datetime64("1993-04-05T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["u500", "u200"]])
def test_wb2era5_fetch(time, variable):

    ds = WB2ERA5(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()
    assert WB2ERA5.available(time[0])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
    ],
)
@pytest.mark.parametrize(
    "variable, wb2_variable, wb2_level",
    [("t2m", "2m_temperature", None), ("u200", "u_component_of_wind", 200)],
)
def test_wb2era5_zarr(time, variable, wb2_variable, wb2_level):

    ds = WB2ERA5(cache=False)
    data = ds(time, variable)

    # From https://cloud.google.com/storage/docs/public-datasets/era5
    era5 = xarray.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        chunks={"time": 1},
        consolidated=True,
    )

    if wb2_level:
        xr_data = era5[wb2_variable].sel(time=time, level=wb2_level)
    else:
        xr_data = era5[wb2_variable].sel(time=time)

    assert np.allclose(data.values, xr_data.values)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1993-04-05T00:00")])],
)
@pytest.mark.parametrize("variable", [["z500", "q200"]])
@pytest.mark.parametrize("cache", [True, False])
def test_wb2era5_cache(time, variable, cache):

    ds = WB2ERA5(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime(year=2024, month=1, day=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_wb2era5_available(time, variable):
    assert not WB2ERA5.available(time)
    with pytest.raises(ValueError):
        ds = WB2ERA5()
        ds(time, variable)
