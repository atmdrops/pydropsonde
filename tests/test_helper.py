import pydropsonde.helper as hh
import pytest
import numpy as np
import xarray as xr

p = np.linspace(10000, 200, 10)
T = np.linspace(30, 20, 10)
q = np.linspace(0.02, 0.002, 10)
alt = np.linspace(0, 12000, 10)
rh = np.repeat(0.8, 10)

ds = xr.Dataset(
    data_vars=dict(
        ta=(["alt"], T),
        p=(["alt"], p),
        rh=(["alt"], rh),
        q=(["alt"], q),
    ),
    coords=dict(
        alt=("alt", alt),
    ),
    attrs={
        "example_attribute_(lines_of_code)": "23",
        "launch_time_(UTC)": "20240903T04:00.000",
    },
)


def test_q2rh2q():
    rh2q = hh.calc_q_from_rh(ds)
    q2rh = hh.calc_rh_from_q(rh2q)
    assert np.all(np.round(ds.rh.values, 3) == np.round(q2rh.rh.values, 3))
