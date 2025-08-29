import pydropsonde.helper as hh
import numpy as np
import xarray as xr

p = np.linspace(100000, 10000, 10)
T = np.linspace(300, 200, 10)
q = np.linspace(0.02, 0.002, 10)
alt = np.linspace(0, 17000, 10)
rh = np.repeat(0.8, 10)
time = np.arange(
    np.datetime64("2024-09-03T04:00:00.500"),
    np.datetime64("2024-09-03T04:10:00.500"),
    np.timedelta64(1, "m"),
)

ds = xr.Dataset(
    data_vars=dict(
        ta=(["alt"], T, {"units": "kelvin"}),
        p=(["alt"], p, {"units": "Pa"}),
        rh=(["alt"], rh),
        q=(["alt"], q),
        time=("alt", time),
    ),
    coords=dict(
        alt=("alt", alt),
    ),
    attrs={
        "example_attribute_(lines_of_code)": "23",
        "launch_time_(UTC)": "2024-09-03T04:00:00.500",
    },
)


def test_q2rh2q():
    rh2q = hh.calc_q_from_rh(ds)
    q2rh = hh.calc_rh_from_q(rh2q)
    assert np.all(np.round(ds.rh.values, 3) == np.round(q2rh.rh.values, 3))


def test_t2theta2t():
    t2theta = hh.calc_theta_from_T(ds)
    theta2t = hh.calc_T_from_theta(t2theta)
    assert np.all(np.round(ds.ta.values, 2) == np.round(theta2t.ta.values, 2))


def test_rh2q_range():
    rh2q = hh.calc_q_from_rh(ds)
    assert rh2q.q.isel(alt=0).values > 0.01
    assert rh2q.q.isel(alt=0).values < 0.02
    assert rh2q.q.isel(alt=-1).values < 1e-4


def test_time_encoding():
    time_encoding = hh.xarray_helper.get_time_encoding(ds.time)
    assert time_encoding["dtype"] == "int64"
    assert "microseconds since" in time_encoding["units"]
    assert np.datetime_as_string(time[0], "s") in time_encoding["units"]


def test_time_encoding_nan():
    time_encoding = hh.xarray_helper.get_time_encoding(
        ds.time.where(ds.time < np.datetime64("2024-09-03T04:09:00"))
    )
    assert np.any(
        np.isnan(ds.time.where(ds.time < np.datetime64("2024-09-03T04:09:00")))
    )
    assert time_encoding["dtype"] == "int64"
    assert "microseconds since" in time_encoding["units"]
    assert np.datetime_as_string(time[0], "s") in time_encoding["units"]
    time_encoding = hh.xarray_helper.get_time_encoding(
        ds.time.where(ds.time < np.datetime64("2024-09-03T03:09:00"))
    )
    assert np.all(
        np.isnan(ds.time.where(ds.time < np.datetime64("2024-09-03T03:09:00")))
    )
    assert time_encoding["dtype"] == "int64"
    assert "microseconds since" in time_encoding["units"]
    assert np.datetime_as_string(time[0], "s") not in time_encoding["units"]
    assert (
        np.datetime_as_string(np.datetime64("1970-01-01T00:00:00"), "s")
        in time_encoding["units"]
    )
