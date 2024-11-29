import numpy as np
from . import physics
import xarray as xr
from configparser import NoSectionError

from moist_thermodynamics import functions as mtf
from moist_thermodynamics import saturation_vapor_pressures as mtsvp

# Keys in l2_variables should be variable names in aspen_ds attribute of Sonde object
l2_variables = {
    "u_wind": {
        "rename_to": "u",
        "attributes": {
            "standard_name": "eastward_wind",
            "long_name": "u component of winds",
            "units": "m s-1",
        },
    },
    "v_wind": {
        "rename_to": "v",
        "attributes": {
            "standard_name": "northward_wind",
            "long_name": "v component of winds",
            "units": "m s-1",
        },
    },
    "tdry": {
        "rename_to": "ta",
        "attributes": {
            "standard_name": "air_temperature",
            "long_name": "air temperature",
            "units": "K",
        },
    },
    "pres": {
        "rename_to": "p",
        "attributes": {
            "standard_name": "air_pressure",
            "long_name": "atmospheric pressure",
            "units": "Pa",
        },
    },
    "rh": {
        "attributes": {
            "standard_name": "relative_humidity",
            "long_name": "relative humidity",
            "units": "",
        }
    },
    "lat": {
        "attributes": {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degree_north",
            "axis": "Y",
        }
    },
    "lon": {
        "attributes": {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degree_east",
            "axis": "X",
        }
    },
    "time": {
        "attributes": {
            "standard_name": "time",
            "long_name": "time of recorded measurement",
            "axis": "T",
        }
    },
    "gpsalt": {
        "attributes": {
            "standard_name": "altitude",
            "long_name": "gps reported altitude above MSL",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        }
    },
    "alt": {
        "attributes": {
            "standard_name": "altitude",
            "long_name": "altitude above MSL",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        }
    },
}


l2_flight_attributes_map = {
    "True Air Speed (m/s)": "true_air_speed_(ms-1)",
    "Ground Speed (m/s)": "ground_speed_(ms-1)",
    "Software Notes": "AVAPS_software_notes",
    "Format Notes": "AVAPS_format_notes",
    "True Heading (deg)": "true_heading_(deg)",
    "Ground Track (deg)": "ground_track_(deg)",
    "Longitude (deg)": "aircraft_longitude_(deg_E)",
    "Latitude (deg)": "aircraft_latitude_(deg_N)",
    "MSL Altitude (m)": "aircraft_msl_altitude_(m)",
    "Geopotential Altitude (m)": "aircraft_geopotential_altitude_(m)",
}

l3_coords = dict(
    launch_time={"long_name": "dropsonde launch time", "time_zone": "UTC"},
    aircraft_longitude={"long_name": "aircraft longitude at launch", "units": "deg_E"},
    aircraft_latitude={"long_name": "aircraft latitude at launch", "units": "deg_N"},
    aircraft_msl_altitude={"long_name": "aircraft altitude at launch", "units": "m"},
)


path_to_flight_ids = "{platform}/Level_0"
path_to_l0_files = "{platform}/Level_0/{flight_id}"

l2_filename_template = "{platform}_{flight_id}_{serial_id}_Level_2.nc"

l3_filename = "Level_3.nc"

es_formular = mtsvp.liq_wagner_pruss
es_name = "Wagner and Pruß 2002 (IAPWS Formulation 1995)"


def get_global_attrs_from_config(config):
    """get global attributes that should be added to each dataset from config
    Input:
        config: configparser
    Returns:
    -------
        global_attrs: dict with global attributes
    """
    try:
        global_attrs = dict(config.items("GLOBAL_ATTRS"))
    except NoSectionError:
        print("No global attributes in config")
        global_attrs = {}

    return global_attrs


l3_vars = [
    "u",
    "v",
    "ta",
    "p",
    "rh",
    "lat",
    "lon",
    "gpsalt",
    "alt",
    "sonde_id",
    "q",
    "iwv",
    "w_dir",
    "w_spd",
]


def get_bool(s):
    if isinstance(s, bool):
        return s
    elif isinstance(s, int):
        return bool(s)
    elif isinstance(s, str):
        lower_s = s.lower()
        if lower_s == "true":
            return True
        elif lower_s == "false":
            return False
        elif lower_s in ["0", "1"]:
            return bool(int(lower_s))
        else:
            raise ValueError(f"Cannot convert {s} to boolean")
    else:
        raise ValueError(f"Cannot convert {s} to boolean")


def convert_rh_to_si(value):
    """convert RH from % to fraction"""
    return value / 100


def convert_pres_to_si(value):
    """convert pressure from hPa to Pa"""
    return value * 100


def convert_tdry_to_si(value):
    """convert temperature from C to K"""
    return value + 273.15


def get_si_converter_function_based_on_var(var_name):
    """get the function to convert a variable to SI units based on its name"""
    func_name = f"convert_{var_name}_to_si"
    func = globals().get(func_name, None)
    if func is None:
        raise ValueError(f"No function named {func_name} found in the module")
    return func


def calc_q_from_rh(ds):
    """
    Input :

        ds : Dataset

    Output :

        ds : Dataset with rh added

    Function to estimate specific humidity from the relative humidity, temperature and pressure in the given dataset.
    """
    e_s = es_formular(ds.ta.values)
    w_s = mtf.partial_pressure_to_mixing_ratio(e_s, ds.p.values)
    w = ds.rh.values * w_s
    q = physics.mr2q(w)
    try:
        q_attrs = ds.q.attrs
        q_attrs.update(
            dict(
                method=f"calculated from RH following {es_name}",
            )
        )
    except AttributeError:
        q_attrs = dict(
            standard_name="specific_humidity",
            long_name="specific humidity",
            units="1",
            method=f"calculated from RH following {es_name}",
        )
    ds = ds.assign(q=(ds.rh.dims, q, q_attrs))
    return ds


def calc_rh_from_q(ds):
    """
    Input :

        ds : Dataset

    Output :

        ds : Dataset with rh added

    Function to estimate relative humidity from the specific humidity, temperature and pressure in the given dataset.
    """
    assert ds.p.attrs["units"] == "Pa"
    e_s = es_formular(ds.ta.values)
    w_s = mtf.partial_pressure_to_mixing_ratio(e_s, ds.p.values)
    w = physics.q2mr(ds.q.values)
    rh = w / w_s

    try:
        rh_attrs = ds.rh.attrs
        rh_attrs.update(
            dict(
                method=f"recalculated from q following {es_name}",
            )
        )
    except AttributeError:
        rh_attrs = dict(
            standard_name="relative_humidity",
            long_name="relative humidity",
            units="1",
            method=f"recalculated from q following {es_name}",
        )
    ds = ds.assign(rh=(ds.q.dims, rh, rh_attrs))

    return ds


def calc_iwv(ds, sonde_dim="sonde_id", alt_dim="alt"):
    """
    Input :

        dataset : Dataset

    Output :

        dataset : Dataset with integrated water vapor

    Function to estimate integrated water vapor in the given dataset.
    """
    pressure = ds.p.values
    temperature = ds.ta.values
    q = ds.q.values
    alt = ds[alt_dim].values

    mask_p = ~np.isnan(pressure)
    mask_t = ~np.isnan(temperature)
    mask_q = ~np.isnan(q)
    mask = mask_p & mask_t & mask_q
    iwv = physics.integrate_water_vapor(
        q=q[mask], p=pressure[mask], T=temperature[mask], z=alt[mask]
    )
    ds_iwv = xr.DataArray([iwv], dims=[sonde_dim], coords={})
    ds_iwv.name = "iwv"
    ds_iwv.attrs = dict(
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
        long_name="integrated water vapour",
        description="vertically integrated water vapour up to aircraft altitude",
    )
    ds = xr.merge([ds, ds_iwv])
    return ds


def calc_theta_from_T(ds):
    """
    Input :

        dataset : Dataset

    Output :

        dataset : Dataset with Potential temperature values

    Function to estimate potential temperature from the temperature and pressure in the given dataset.
    """
    assert ds.p.attrs["units"] == "Pa"
    theta = mtf.theta(ds.ta.values, ds.p.values)
    try:
        theta_attrs = ds.theta.attrs
    except AttributeError:
        theta_attrs = dict(
            standard_name="air_potential_temperature",
            long_name="potential temperature",
            units="kelvin",
        )
    ds = ds.assign(theta=(ds.ta.dims, theta, theta_attrs))

    return ds


def calc_T_from_theta(ds):
    """
    Input :

        dataset : Dataset

    Output :

        dataset: Dataset with temperature calculated from theta

    Function to estimate potential temperature from the temperature and pressure in the given dataset.
    """
    assert ds.p.attrs["units"] == "Pa"
    ta = physics.theta2ta(ds.theta.values, ds.p.values)

    try:
        t_attrs = ds.ta.attrs
    except AttributeError:
        t_attrs = dict(
            standard_name="air_temperature",
            long_name="air temperature",
            units="K",
        )
    ds = ds.assign(ta=(ds.theta.dims, ta, t_attrs))
    return ds


def calc_theta_e(ds):
    """
    Input :

        dataset : Dataset

    Output :

        dataset: Dataset with theta_e added

    Function to estimate theta_e from the temperature, pressure and q in the given dataset.
    """

    assert ds.p.attrs["units"] == "Pa"
    theta_e = mtf.theta_e(T=ds.ta.values, P=ds.p.values, qt=ds.q.values, es=es_formular)

    ds = ds.assign(
        theta_e=(
            ds.ta.dims,
            theta_e,
            dict(
                standard_name="air_equivalent_potential_temperature",
                long_name="equivalent potential temperature",
                units="kelvin",
            ),
        )
    )
    return ds


def calc_wind_dir_and_speed(ds):
    """
    Input :

        dataset : Dataset

    Output :

        dataset: Dataset wind direction and wind speed

    Calculates wind direction between 0 and 360 according to https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

    """
    w_dir = (180 + np.arctan2(ds.u.values, ds.v.values) * 180 / np.pi) % 360
    w_spd = np.sqrt(ds.u.values**2 + ds.v.values**2)

    ds = ds.assign(
        w_dir=(
            ds.u.dims,
            w_dir,
            dict(
                standard_name="wind_from_direction",
                units="degree",
            ),
        )
    )

    ds = ds.assign(
        w_spd=(
            ds.u.dims,
            w_spd,
            dict(
                standard_name="wind_speed",
                units="m s-1",
            ),
        )
    )
    return ds
