# this is used from typhon because poetry does not like typhon
import numpy as np
from numbers import Number
import metpy.constants as mpconst


triple_point_water = 273.16  # Triple point temperature in K


def e_eq_ice_mk(T):
    if np.any(T <= 0):
        raise ValueError("Temperatures must be larger than 0 Kelvin.")

    # Give the natural log of saturation vapor pressure over ice in Pa
    e = 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T

    return np.exp(e)


def e_eq_water_mk(T):
    if np.any(T <= 0):
        raise ValueError("Temperatures must be larger than 0 Kelvin.")

    # Give the natural log of saturation vapor pressure over water in Pa

    e = (
        54.842763
        - 6763.22 / T
        - 4.21 * np.log(T)
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
    )

    return np.exp(e)


def e_eq_mixed_mk(T):
    is_float_input = isinstance(T, Number)
    if is_float_input:
        # Convert float input to ndarray to allow indexing.
        T = np.asarray([T])

    e_eq_water = e_eq_water_mk(T)
    e_eq_ice = e_eq_ice_mk(T)

    is_water = T > triple_point_water

    is_ice = T < (triple_point_water - 23.0)

    e_eq = (
        e_eq_ice + (e_eq_water - e_eq_ice) * ((T - triple_point_water + 23) / 23) ** 2
    )
    e_eq[is_ice] = e_eq_ice[is_ice]
    e_eq[is_water] = e_eq_water[is_water]

    return e_eq[0] if is_float_input else e_eq


def relative_humidity2vmr(
    RH,
    p,
    T,
    e_eq=e_eq_mixed_mk,
):
    return RH * e_eq(T) / p


def vmr2specific_humidity(x):
    Md = mpconst.dry_air_molecular_weight.magnitude
    Mw = mpconst.water_molecular_weight.magnitude * 1e-3
    return x / ((1 - x) * Md / Mw + x)


def specific_humidity2vmr(q):
    Md = mpconst.dry_air_molecular_weight.magnitude
    Mw = mpconst.water_molecular_weight.magnitude * 1e-3

    return q / ((1 - q) * Mw / Md + q)


def vmr2relative_humidity(vmr, p, T, e_eq=e_eq_mixed_mk):
    return vmr * p / e_eq(T)


def density(p, T, R=mpconst.dry_air_molecular_weight.magnitude):
    return p / (R * T)


def integrate_water_vapor(vmr, p, T=None, z=None, axis=0):
    def integrate_column(y, x, axis=0):
        if np.all(x[:-1] >= x[1:]):
            return -np.trapz(y, x, axis=axis)
        else:
            return np.trapz(y, x, axis=axis)

    if T is None and z is None:
        # Calculate IWV assuming hydrostatic equilibrium.
        q = vmr2specific_humidity(vmr)
        g = mpconst.earth_gravity.magnitude
        return -integrate_column(q, p, axis=axis) / g
    elif T is None or z is None:
        raise ValueError(
            "Pass both `T` and `z` for non-hydrostatic calculation of the IWV."
        )
    else:
        # Integrate the water vapor mass density for non-hydrostatic cases.
        R_v = mpconst.water_gas_constant.magnitude
        rho = density(p, T, R=R_v)  # Water vapor density.
        return integrate_column(vmr * rho, z, axis=axis)
