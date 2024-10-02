import pytest
import xarray as xr
import numpy as np
from pydropsonde.processor import Sonde

s_id = "test_this_id"
flight_id = "test_this_flight"
platform_id = "test_this_platform"
launch_time = "2020-02-02 20:22:02"


@pytest.fixture
def sonde():
    sonde = Sonde(serial_id=s_id, launch_time=launch_time)
    sonde.add_flight_id(flight_id)
    sonde.add_platform_id(platform_id)
    return sonde


@pytest.fixture
def sonde_with_prep(sonde):
    altitude = np.arange(0, 1000, 10)[::-1]
    time = np.arange(0, 1000, 10)

    p = np.logspace(1000, 0, 100)
    rh = np.arange(0, 1, 1 / 100)
    ds = xr.Dataset(
        {"p": ("time", p), "rh": ("time", rh), "alt": ("time", altitude)},
        coords={
            "time": time,
            "lat": ("time", time),
            "lon": ("time", time),
            "gpsalt": ("time", time),
        },
    )

    object.__setattr__(sonde, "_prep_l3_ds", ds)
    return sonde


@pytest.fixture
def sonde_interp(sonde_with_prep):
    prep_l3 = sonde_with_prep._prep_l3_ds
    rh = prep_l3.rh.values
    new_rh = np.where((rh > 0.3) & (rh < 0.8), np.nan, rh)
    new_rh[-10:-6] = np.nan
    ds = prep_l3.assign(rh=(prep_l3.dims, new_rh))
    object.__setattr__(sonde_with_prep, "_prep_l3_ds", ds)

    sonde = sonde_with_prep.interpolate_alt(
        alt_var="alt",
        interp_start=0,
        interp_stop=1000,
        interp_step=25,
        max_gap_fill=int(70),
        p_log=True,
        method="bin",
    )

    grid = np.arange(0, 1000, 25)
    grid = (grid[1:] + grid[:-1]) * 0.5
    assert np.all(sonde._prep_l3_ds.alt.values == grid)
    assert np.all(~np.isnan(sonde._prep_l3_ds.rh.values[:8]))
    assert np.all(np.isnan(sonde._prep_l3_ds.rh.values[-31:-12]))
    assert sonde._prep_l3_ds.rh.values[-1] == np.mean(new_rh[2:4])

    return sonde


def test_N_m(sonde_interp):
    sonde = sonde_interp.get_N_m_values()
    mrh = sonde._prep_l3_ds.mrh.values
    mp = sonde._prep_l3_ds.mp.values
    assert np.all(mp == 2)
    assert np.all(mrh[-31:-12] == 0)
    assert np.all(mrh[-12:] == 2)
    assert np.all(mrh[0:1] == 2)
    assert np.all(mrh[3:8] == 2)
    assert mrh[2] == 1
