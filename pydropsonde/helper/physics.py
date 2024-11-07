# this is used from typhon because poetry does not like typhon
import numpy as np
from numbers import Number
import metpy.constants as mpconst


triple_point_water = 273.16  # Triple point temperature in K


def q2vmr(q):
    """
    returns the volume mixing ratio from specific humidity
    """
    return q / ((1 - q) * constants.molar_mass_h2o / constants.md + q)


def vmr2q(vmr):
    """
    returns specific humidity from volume mixing ratio
    """
    return vmr / ((1 - vmr) * constants.md / constants.molar_mass_h2o + x)

def density(p, T, R):
    """
    returns density for given pressure, temperature and R
    """
    return p / (R * T) #water vapor density


def integrate_water_vapor(p, q,  T=None, z=None, axis=0):
    """Returns the integrated water vapor for given specific humidity
    Args: 
        p: pressure in Pa
        either: (hydrostatic)
            q: specific humidity
        or: (non-hydrostatic)
            q: specific humidity
            T: temperature
            z: height

    """
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
