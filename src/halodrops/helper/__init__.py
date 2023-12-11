# Keys in l2_variables should be variable names in aspen_ds attribute of Sonde object
l2_variables = {
    "u_wind": {
        "rename_to": "u",
        "attributes": {
            "standard_name": "eastward_wind",
            "long_name": "u component of winds",
            "units": "m s-1",
            "coordinates": "time lon lat alt",
        },
    },
    "v_wind": {
        "rename_to": "v",
        "attributes": {
            "standard_name": "northward_wind",
            "long_name": "v component of winds",
            "units": "m s-1",
            "coordinates": "time lon lat alt",
        },
    },
    "tdry": {
        "rename_to": "ta",
        "attributes": {
            "standard_name": "air_temperature",
            "long_name": "air temperature",
            "units": "K",
            "coordinates": "time lon lat alt",
        },
    },
    "pres": {
        "rename_to": "p",
        "attributes": {
            "standard_name": "air_pressure",
            "long_name": "atmospheric pressure",
            "units": "Pa",
            "coordinates": "time lon lat alt",
        },
    },
    "rh": {
        "attributes": {
            "standard_name": "relative_humidity",
            "long_name": "relative humidity",
            "units": "",
            "coordinates": "time lon lat alt",
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
