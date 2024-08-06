import warnings

import numpy as np
import xarray as xr
from configparser import ConfigParser
import os

import geopy.distance
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm

warnings.filterwarnings("ignore")

config = ConfigParser(inline_comment_prefixes="#")                                          # read config file
config.read("config.cfg")

varname_L4 = config["LEVEL 4"]["varname_L4"].split(", ")                                    # load variable names to process

lat_var    = config["LEVEL 4"]["latitude_varname"]                                          # get variable names from config file
lon_var    = config["LEVEL 4"]["longitude_varname"]
alt_var    = config["LEVEL 4"]["altitude_varname"]
uwind_var  = config["LEVEL 4"]["u_wind_varname"]
vwind_var  = config["LEVEL 4"]["v_wind_varname"]
q_var      = config["LEVEL 4"]["specific_humidity_varname"]
t_var      = config["LEVEL 4"]["air_temperature_varname"]
p_var      = config["LEVEL 4"]["pressure_varname"]

coordsys   = config["REGRESSION"]["coordinate_system"]                                      # get coordinate system specification 
                                                                                            # for regression algoritm

def get_pattern(archive, list_of_sonde_ids):
    """
    Reads out sondes data for all sondes of one flight pattern

    Arguments:
        archive           (string):          data directory where gridded file is stored
        list_of_sonde_ids (list of strings): list of strings of the sonde ids included 
                                             in the flight pattern

    Returns:
        dataset (xr.Dataset): dataset with only the sondes included in the flight pattern
    """

    file_name       = (Datestr + "_" + Platform + "_" + Flightstr +"_L3.nc")                # file name of Level 3 data           
    gridded_dataset = xr.open_dataset(archive + "Level_3/"+ file_name)                      # load output dataset from Level 3

    pattern_list    = [None] * len(list_of_sonde_ids)                                       # define list to store data from
                                                                                            # all sondes of a pattern
    for id_, sonde_id in enumerate(list_of_sonde_ids):  
        pattern_list[id_] = gridded_dataset.sel(sonde_id=sonde_id)                          # add dataset for each sonde to list
    
    dataset = xr.concat(pattern_list, dim="sonde_id")                                       # concatenate to dataset only 
                                                                                            # containing the pattern sondes
    return dataset

def get_circle_center_coords(dataset):
    """
    Calculate center position, coordinates in specified coordinate system and distance
    from pattern center.
    "spherical" and "carthesian" coordinates available.

    For carthesian coordinates:
        x_coord : longitude * 111.32km * cos(latitude)
        y_coord : latitude * 110.54km 
        https://en.wikipedia.org/wiki/Latitude

    For spherical coordinates:
        theta : polar angle     = 90-latitude
        phi   : azimuthal angle = longitude
        https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates

    Arguments:
        dataset  (xr.Dataset): dataset with all sondes included in the flight pattern
        coordsys (string):     string to define coordinate system, choose from
                               "carthesian" and "spherical"
    """
    if coordsys == "carthesian":

        x_coor     = dataset[lon_var] * 111.32*1000 * np.cos(np.radians(dataset[lat_var]))  # convert from latitude, longitude
        y_coor     = dataset[lat_var] * 110.54*1000                                         # to coordinates in meter from 0°N 0°E
        center_x   = x_coor.mean(dim="sonde_id", skipna=True)                               # get pattern center position
        center_y   = y_coor.mean(dim="sonde_id", skipna=True)                               # for each height level
        center_lat = center_x / (111.43 * 1000 * np.cos(np.radians(center_x)))              # convert center position back to 
        center_lon = center_y / (110.54 * 1000)                                             # latitude-longitude
        dx         = x_coor - center_x                                                      # get distance of each sonde from
        dy         = y_coor - center_y                                                      # the pattern center in x and y direction

        dataset["x_coor"]     = x_coor                                                      # saving variables to dataset
        dataset["y_coor"]     = y_coor
        dataset["center_lat"] = center_lat
        dataset["center_lon"] = center_lon
        dataset["dx"]         = dx 
        dataset["dy"]         = dy

        dataset.x_coor.attrs["units"]         = "meter"                                     # adding descriptions and units
        dataset.x_coor.attrs["long_name"]     = "x coordinate, distance in lon-direction from 0°N 0°E"
        dataset.y_coor.attrs["units"]         = "meter"
        dataset.y_coor.attrs["long_name"]     = "y coordinate, distance in lat-direction from 0°N 0°E"
        dataset.center_lat.attrs["units"]     = "degree"
        dataset.center_lat.attrs["long_name"] = "pattern center latitude coordinate"
        dataset.center_lon.attrs["units"]     = "degree"
        dataset.center_lon.attrs["long_name"] = "pattern center longitude coordinate"
        dataset.dx.attrs["units"]             = "meter"
        dataset.dx.attrs["long_name"]         = "difference between sonde and mean (|sonde_x - center_x|)"
        dataset.dy.attrs["units"]             = "meter"
        dataset.dy.attrs["long_name"]         = "difference between sonde and mean (|sonde_y - center_y|)"

    elif coordsys == "spherical":

        theta        = np.pi * (90 - dataset[lat_var]) / 180.                               # convert from latitude, longitude
        phi          = np.pi * dataset[lon_var] / 180.                                      # to spherical coordinates theta, phi
        center_theta = theta.mean(dim="sonde_id", skipna=True)                              # get pattern center position 
        center_phi   = phi.mean(dim="sonde_id", skipna=True)                                # for each height level
        center_lat   = 90 - (center_theta * 180 / np.pi)                                    # convert center position back to
        center_lon   = center_phi * 180 / np.pi                                             # latitude-longitude
        dtheta       = -(theta - center_theta)                                              # get distance of each sonde from
        dphi         = phi - center_phi                                                     # the pattern center in theta and phi
 
        dataset["theta"]        = theta                                                     # saving variables to dataset
        dataset["phi"]          = phi
        dataset["center_theta"] = center_theta
        dataset["center_phi"]   = center_phi
        dataset["center_lat"]   = center_lat
        dataset["center_lon"]   = center_lon
        dataset["dtheta"]       = dtheta 
        dataset["dphi"]         = dphi

        dataset.theta.attrs["units"]            = "radian"                                  # adding descriptions and units
        dataset.theta.attrs["long_name"]        = "theta coordinate (90°-latitude)"
        dataset.phi.attrs["units"]              = "radian"
        dataset.phi.attrs["long_name"]          = "phi coordinate (longitude)"
        dataset.center_theta.attrs["units"]     = "radian"
        dataset.center_theta.attrs["long_name"] = "pattern center theta coordinate"
        dataset.center_phi.attrs["units"]       = "radian"
        dataset.center_phi.attrs["long_name"]   = "pattern center phi coordinate"
        dataset.center_lat.attrs["units"]       = "degree"
        dataset.center_lat.attrs["long_name"]   = "pattern center latitude coordinate"
        dataset.center_lon.attrs["units"]       = "degree"
        dataset.center_lon.attrs["long_name"]   = "pattern center longitude coordinate"
        dataset.dtheta.attrs["units"]           = "radian"
        dataset.dtheta.attrs["long_name"]       = "difference between sonde and mean (|sonde_theta - center_theta|)"
        dataset.dphi.attrs["units"]             = "radian"
        dataset.dphi.attrs["long_name"]         = "difference between sonde and mean (|sonde_phi - center_phi|)"

    else: print("Coordinate system not included, please choose carthesian or spherical.")

    return print("Circles ready for regression")

def fitfunc_pinv(dcoord1, dcoord2, par):
    """
    Estimate a 2D linear model to calculate variable-values from theta-phi coordinates

    Arguments:
        dcoord1 (xr.DataArray): coordinates of datapoints (distance from pattern center)
        dcoord2 (xr.DataArray): coordinates of datapoints (distance from pattern center)
        par     (xr.DataArray): data values

    Returns:
        intercept (xr.DataArray), dpar_dcoord1 (xr.DataArray), dpar_dcoord2 (xr.DataArray),
        par_ (xr.DataArray)
    """

    par_ = par
    par  = np.array(par, copy=True)                                                         # to fix nans, do a copy
    a    = np.stack([np.ones_like(dcoord1), dcoord1, dcoord2], axis=-1)                     # a does not need to be copied as this 
                                                                                            # creates a copy already

    invalid      = np.isnan(par) | np.isnan(dcoord1) | np.isnan(dcoord2)                    # for handling missing values, 
    par[invalid] = 0                                                                        # both par and a are set to 0, 
    a[invalid]   = 0                                                                        # that way these items don't 
    under_constraint = np.sum(~invalid, axis=-1) < float(config["LEVEL 4"]["min_sondes"])   # influence the fit
                                                                                            
    a_inv = np.linalg.pinv(a)                                                               # calculate the 
                                                                                            # Moore-Penrose-Pseudoinverse 
                                                                                            # of the matrix a

    intercept, dpar_dcoord1, dpar_dcoord2 = np.einsum("ijk,ik->ji", a_inv, par)             # perform Einstein summation to solve
                                                                                            # differential equation
    intercept[under_constraint] = np.nan
    dpar_dcoord1[under_constraint] = np.nan
    dpar_dcoord2[under_constraint] = np.nan

    return intercept, dpar_dcoord1, dpar_dcoord2, par_

def fitfunc_phasespace(dcoord1, dcoord2, par):
    """
    calculate variable-values from theta-phi coordinates by minimising the residual
    in p-q-phase space

    Arguments:
        dcoord1 (xr.DataArray): theta coordinates of datapoints (distance from pattern center)
        dcoord2 (xr.DataArray): phi coordinates of datapoints (distance from pattern center)
        par     (xr.DataArray): data values

    Returns:
        intercept (xr.DataArray), dpar_dcoord1 (xr.DataArray), dpar_dcoord2 (xr.DataArray),
        par_ (xr.DataArray)
    """

    par_ = par
    par_av = par.mean(dim=["sonde_id"])                                                     # calculate average of variable between
                                                                                            # all sondes in pattern

    pdim = int(config["LEVEL 4"]["phase_space_pdim"])                                       # p and q dimension of phase space
    qdim = int(config["LEVEL 4"]["phase_space_qdim"])

    dpar_dcoord1     = np.empty((pdim, len(par[alt_var])), dtype=np.float)                  # calculate variable difference in 
    dpar_dcoord1_min = (par.max(dim=["sonde_id"]) - par.min(dim=["sonde_id"])) / (          # theta direction in p space
                        -dcoord1.max(dim=["sonde_id"]))
    dpar_dcoord1_max = (par.max(dim=["sonde_id"]) - par.min(dim=["sonde_id"])) / (
                        dcoord1.max(dim=["sonde_id"]))
    incr             = (dpar_dcoord1_max - dpar_dcoord1_min) / pdim
    
    for p in range(pdim):
        dpar_dcoord1[p] = dpar_dcoord1_min + p * incr

    dpar_dcoord2       = np.empty((qdim, len(par[alt_var])), dtype=np.float)                # calculate variable difference in
    dpar_dcoord2_min   = (par.max(dim=["sonde_id"]) - par.min(dim=["sonde_id"])) / (        # phi direction in q space
                          -dcoord2.max(dim=["sonde_id"]))
    dpar_dcoord2_max   = (par.max(dim=["sonde_id"]) - par.min(dim=["sonde_id"])) / ( 
                          dcoord2.max(dim=["sonde_id"]))
    incr               = (dpar_dcoord2_max - dpar_dcoord2_min) / qdim
    
    for q in range(qdim):
        dpar_dcoord2[q]   = dpar_dcoord2_min + q * incr

    R = np.zeros((pdim, qdim, len(par[alt_var])) , dtype=type(par))                         # Residual: R = sum_i( Residual_i^2 ) 
                                                                                            # where i is the dropsonde index
    for p in range(pdim):
        for q in range(qdim):
            nav = 0
            for i in (par["sonde_id"]):
                R[p,q]     = R[p,q] + (par.loc[dict(sonde_id=i)] - dpar_dcoord1[p] * 
                             dcoord1.loc[dict(sonde_id=i)] - dpar_dcoord2[q] *
                             dcoord2.loc[dict(sonde_id=i)])**2
                nav        = nav + 1
            if nav > 0:
                R[p,q] = (R[p,q] / nav)**0.5

    dpar_dcoord1_out = xr.DataArray(np.empty(len(par[alt_var])),                            # define output arrays
                                    coords=[par[alt_var]])
    dpar_dcoord2_out = xr.DataArray(np.empty(len(par[alt_var])),
                                    coords=[par[alt_var]])
    
    for k in np.arange(len(par[alt_var])):                                                  # locate minimum residual R 
        inds                = np.unravel_index(R[:,:,k].argmin(), R[:,:,k].shape)           # (i.e. the integrated residual^2)
        dpar_dcoord1_out[k] = dpar_dcoord1[inds[0], k]                                       # in gradient phase-space
        dpar_dcoord2_out[k] = dpar_dcoord2[inds[1], k]                                         # for each height level

    return par_av, dpar_dcoord1_out, dpar_dcoord2_out, par_

def get_div_and_vor(dataset):
    """
    Calculation of wind divergence and relative vorticity (z-component).

    For carthesian coordinates:
        Divergence: D    = div(wind) = du/dx + dv/dy
        Vorticity:  zeta = rot(wind) = du/dy - dv/dx

    For spherical coordinates:
        Divergence: D    = div(wind) = 1/(r*sin(theta)) * (du/dphi + d(sin(theta)*v)/dtheta)
        Vorticity:  zeta = rot(wind) = 1/(r*sin(theta)) * (dv/dphi - d(sin(theta)*u)/dtheta)
    
    Arguments:
        For carthesian coordinates:
            dataset (xr.Dataset): dataset with variables [duwind_dx, duwind_dx, 
                                  dvwind_dx, dvwind_dy]

        For spherical coordinates:
            dataset (xr.DataSet): dataset with variables [duwind_dtheta, duwind_dphi, 
                                  dvwind_dtheta, dvwind_dphi, dsintheta_uwind_dtheta,
                                  dsintheta_uwind_dphi, dsintheta_vwind_dtheta,
                                  dsintheta_vwind_dphi]
    """

    if coordsys == "carthesian":

        dataset["div"] = dataset["d"+uwind_var+"_dx"] + dataset["d"+vwind_var+"_dy"]        # calculate wind divergence
        dataset["vor"] = dataset["d"+vwind_var+"_dx"] + dataset["d"+uwind_var+"_dy"]        # calculate relative vorticity

        dataset.div.attrs["units"]     = "1/s"                                         # add descriptions and units 
        dataset.div.attrs["long_name"] = "wind divergence from carthesian coordinates"
        dataset.vor.attrs["units"]     = "1/s"
        dataset.vor.attrs["long_name"] = "relative vorticity from carthesian coordinates"

    if coordsys == "spherical":

        r   = float(config["CONSTANTS"]["r_Earth"])                                         # get radius of the earth from 
                                                                                            # config file

        dataset["div"] = 1/(r * np.sin(dataset.center_theta)) * (                           # calculate wind divergence
                         dataset["d"+uwind_var+"_dphi"] +         
                         dataset["dsintheta_"+vwind_var+"_dtheta"])
        
        dataset["vor"] = 1/(r * np.sin(dataset.center_theta)) * (                           # calculate vorticity
                         dataset["d"+vwind_var+"_dphi"] - 
                         dataset["dsintheta_"+uwind_var+"_dtheta"])

        dataset.div.attrs["units"]     = "1/s"                                         # add descriptions and units
        dataset.div.attrs["long_name"] = "wind divergence from spherical coordinates"
        dataset.vor.attrs["units"]     = "1/s"
        dataset.vor.attrs["long_name"] = "relative vorticity from spherical coordinates"

    return print("Finished estimating divergence and vorticity for all circles....")

def get_vertical_velocity_and_omega(dataset):
    """
    Calculation of density, vertical velocity and pressure velocity.

    Vertical velocity: w     = - int_0^z(divergence)dz
    Pressure velocity: omega = - g*density*w

    Arguments:
        dataset (xr.DataSet): dataset with variables [q_var, p_var, t_var, div] and
                              coordinated [alt_var]
    """

    mr = mpcalc.mixing_ratio_from_specific_humidity(dataset[q_var].values)                  # get mixing ratio from q
    density = mpcalc.density(dataset[p_var].values * units.Pa,                              # calculate air density
                         dataset[t_var].values * units.kelvin, mr).magnitude
    mean_density = np.nanmean(density, axis=0)

    div    = dataset.div.values  
    height = dataset[alt_var].values                                                            
    w      = np.zeros((len(height)))

    height_id, diff = 1, 0
    while (height_id < len(height)) & (diff <= int(config["LEVEL 4"]["max_div_gap"])):

        if (~np.isnan(div[height_id])) & (~np.isnan(height[height_id])):      
              
            idx  = height_id-1                                                              # make sure to not include
            while (np.isnan(w[idx])) & (diff <= int(config["LEVEL 4"]["max_div_gap"])):     # np.nan values of w
                    idx  = idx - 1                                                          # and enlarge the step size 
                    diff = diff + 1                                                         # until next value is found

            if diff <= int(config["LEVEL 4"]["max_div_gap"]):
                w[height_id] = w[idx] - (div[height_id] * (height[height_id] - height[idx]))
                diff = 0

            height_id = height_id + 1
            
        else:
            w[height_id] = np.nan
            height_id    = height_id + 1

    if diff > int(config["LEVEL 4"]["max_div_gap"]):                                        # stop calculation of w if
        print("Datagap in divergence is too large, calculation of vertical" +               # maximal datagap defined in 
              "velocity has been aborted")                                                  # config file is exceeded

    omega = - float(config["CONSTANTS"]["g"]) * mean_density * w                            # calculate pressure velocity              

    dataset["density"]      = xr.DataArray(density, coords=dataset[q_var].coords)           # saving variables to dataset
    dataset["mean_density"] = xr.DataArray(mean_density, coords=dataset[alt_var].coords)
    dataset["w_wind"]       = xr.DataArray(w, coords=dataset[alt_var].coords)
    dataset["omega"]        = xr.DataArray(omega, coords=dataset[alt_var].coords)

    dataset.density.attrs["units"]          = "kg/m^3"                                      # adding descriptions and units
    dataset.density.attrs["long_name"]      = "air density"
    dataset.mean_density.attrs["units"]     = "kg/m^3"
    dataset.mean_density.attrs["long_name"] = "mean air density averaged over all sondes in pattern"
    dataset.w_wind.attrs["units"]           = "m/s"
    dataset.w_wind.attrs["long_name"]       = "vertical wind velocity"
    dataset.omega.attrs["units"]            = "Pa/s"
    dataset.omega.attrs["long_name"]        = "pressure velocity"

    return print("Finished estimating density, w and omega ...")

def get_advection(dataset, list_of_parameters=[t_var, q_var, p_var]):
    """
    Calculation of advection for parameters specified in
    list_of_parameters.

    For carthesian coordinates:
         wind * grad(par) = u * dpar/dx + v * dpar/dy

    For spherical coordinates: 
        wind * grad(par) = u * 1/r dpar/dtheta + v * 1/(r*sin(theta)) dpar/dphi

    Arguments:
        dataset (xr.DataSet): dataset with variables [uwind_var, vwind_var] and 
                              variables listed in list_of_parameters and their
                              horizontal gradients
        list_of_parameters (list of strings)
    """

    for par in list_of_parameters:
        
        if coordsys == "carthesian":
        
            dataset[par+"_adv"] = (dataset[uwind_var+"_mean"] * dataset[f"d{par}_dx"]) + ( # calculation of advection     
                                   dataset[vwind_var+"_mean"] * dataset[f"d{par}_dy"])
            
            dataset[par+"_adv"].attrs["units"] = dataset[par].attrs["units"] + "/s"         # adding description and unit
            dataset[par+"_adv"].attrs["long_name"] = "advection of " +                    \
                                                     dataset[par].attrs["long_name"]

        elif coordsys == "spherical":

            r = float(config["CONSTANTS"]["r_Earth"])                                       # get earth radius from config file
            
            dataset[par+"_adv"] = (dataset[uwind_var+"_mean"] * 1/r *                       # calculation of advection 
                                   dataset["d"+par+"_dtheta"]) + (
                                   dataset[vwind_var+"_mean"] * 1/(
                                   r*np.sin(dataset.center_theta)) * 
                                   dataset["d"+par+"_dphi"])
            
            dataset[par+"_adv"].attrs["units"] = dataset[par].attrs["units"] + "/s"         # adding description and unit
            dataset[par+"_adv"].attrs["long_name"] = "advection of " +                    \
                                                     dataset[par].attrs["long_name"]
    
    return print("Finished estimating advection terms ...")

def add_std_err_terms(dataset):
    """
    Adding standard errors to calculated variables in dataset based on gaussian error
    propagation.

    Arguments:
        dataset (xrarray Dataset)
    """

    if coordsys == "carthesian":

        dx_mean = dataset.dx.mean(dim="sonde_id")
        dy_mean = dataset.dy.mean(dim="sonde_id")

        dx_denominator = np.sqrt(((dataset.dx - dx_mean) ** 2).sum(dim="sonde_id"))         # calculate error for dx
        dy_denominator = np.sqrt(((dataset.dy - dy_mean) ** 2).sum(dim="sonde_id"))         # calculate error for dy

        for par in tqdm(varname_L4, desc="Errors"):

            par_err = dataset[par] - (dataset[f"{par}_mean"] +                              # calculate standard error
                      (dataset[f"d{par}_dx"] * dataset.dx) +                                # for variables
                      (dataset[f"d{par}_dy"] * dataset.dy))

            par_sq_sum = (par_err**2).sum(dim="sonde_id", skipna=True)
            par_n = (~np.isnan(par_err)).sum(dim="sonde_id") 

            par_numerator = np.sqrt(par_sq_sum / (par_n - 3))

            var_name_dx = "d" + par + "_dx_stderr"
            var_name_dy = "d" + par + "_dy_stderr"

            dataset[var_name_dx] = par_numerator / dx_denominator                           # calculate standard error
            dataset[var_name_dy] = par_numerator / dy_denominator                           # for horizontal gradients

        dataset["div_stderr"] = np.sqrt((dataset[f"d{uwind_var}_dx_stderr"])**2 +           # calculate divergence 
                                        (dataset[f"d{vwind_var}_dy_stderr"])**2)            # standard error
        dataset["div_stderr"].attrs["units"]     = dataset["div"].attrs["units"]            # add unit and description
        dataset["div_stderr"].attrs["long_name"] = dataset["div"].attrs["long_name"] +    \
                                                            " standard error"

        dataset["vor_stderr"] = np.sqrt((dataset[f"d{uwind_var}_dy_stderr"])**2 +           # calculate vorticity 
                                        (dataset[f"d{vwind_var}_dx_stderr"])**2)            # standard error
        dataset["vor_stderr"].attrs["units"]     = dataset["vor"].attrs["units"]            # add unit and description
        dataset["vor_stderr"].attrs["long_name"] = dataset["vor"].attrs["long_name"] +    \
                                                            " standard error"

        dataset[f"{q_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] *            # calculate standard error
                                                  dataset[f"d{q_var}_dx_stderr"])**2 +      # for humidity advection
                                                 (dataset[f"{vwind_var}_mean"] *
                                                  dataset[f"d{q_var}_dy_stderr"])**2)
        dataset[f"{q_var}_adv_stderr"].attrs["units"]     = dataset[f"{q_var}_adv"].attrs[  # add unit and description
                                                            "units"]
        dataset[f"{q_var}_adv_stderr"].attrs["long_name"] = dataset[f"{q_var}_adv"].attrs[
                                                            "long_name"] + " standard error"

        dataset[f"{t_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] *            # calculate standard error
                                                  dataset[f"d{t_var}_dx_stderr"])**2 +      # for temperature advection
                                                 (dataset[f"{vwind_var}_mean"] * 
                                                  dataset[f"d{t_var}_dy_stderr"])**2)        
        dataset[f"{t_var}_adv_stderr"].attrs["units"]     = dataset[f"{t_var}_adv"].attrs[  # add unit and description
                                                            "units"]
        dataset[f"{t_var}_adv_stderr"].attrs["long_name"] = dataset[f"{t_var}_adv"].attrs[
                                                            "long_name"] +" standard error"

        dataset[f"{p_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] *            # calculate standard error
                                                  dataset[f"d{p_var}_dx_stderr"])**2 +      # for pressure advection
                                                 (dataset[f"{vwind_var}_mean"] * 
                                                  dataset[f"d{p_var}_dy_stderr"])**2)
        dataset[f"{p_var}_adv_stderr"].attrs["units"]     = dataset[f"{p_var}_adv"].attrs[  # add unit and description
                                                            "units"]
        dataset[f"{p_var}_adv_stderr"].attrs["long_name"] = dataset[f"{p_var}_adv"].attrs[
                                                            "long_name"] +" standard error"

    elif coordsys == "spherical":

        r = float(config["CONSTANTS"]["r_Earth"])

        dtheta_mean = dataset.dtheta.mean(dim="sonde_id")
        dphi_mean   = dataset.dphi.mean(dim="sonde_id")

        dtheta_denominator = np.sqrt(((dataset.dtheta-dtheta_mean)**2).sum(dim="sonde_id")) # calculate error for dtheta
        dphi_denominator   = np.sqrt(((dataset.dphi-dphi_mean)**2).sum(dim="sonde_id"))     # calculate error for dphi

        for par in tqdm(varname_L4, desc="Errors"):                                         

            par_err = dataset[par] - dataset[f"{par}_mean"]                                 # calculate error for all
                                                                                            # variables

            #par_err = dataset[par] - (dataset[f"{par}_mean"] +                             # caluclate error for all
                      #(dataset[f"d{par}_dtheta"] * dataset.dtheta) +                        # variables including
                      #(dataset[f"d{par}_dphi"] * dataset.dphi))                             # horizontal gradients

            par_sq_sum = (par_err**2).sum(dim="sonde_id", skipna=True)
            par_n      = (~np.isnan(par_err)).sum(dim="sonde_id")

            par_numerator = np.sqrt(par_sq_sum) / par_n
            #par_numerator = np.sqrt(par_sq_sum / (par_n - 3))

            var_name_dtheta = f"d{par}_dtheta_stderr"
            var_name_dphi   = f"d{par}_dphi_stderr"

            dataset[var_name_dtheta] = par_numerator / dtheta_denominator                   # calculate error for 
            dataset[var_name_dphi]   = par_numerator / dphi_denominator                     # horizontal gradients                
            
        dataset["div_stderr"] = (1/(r*np.sin(dataset["center_theta"]))) * np.sqrt(          # calculate divergence
                                (dataset[f"d{uwind_var}_dphi_stderr"])**2 +                 # standard error
                                (dataset[f"dsintheta_{vwind_var}_dtheta_stderr"])**2)
        dataset["div_stderr"].attrs["units"]     = dataset["div"].attrs["units"]            # add units and description       
        dataset["div_stderr"].attrs["long_name"] = dataset["div"].attrs["long_name"] +    \
                                                            " standard error"

        dataset["vor_stderr"] = (1/(r*np.sin(dataset["center_theta"]))) * np.sqrt(          # calculate vorticity
                                (dataset[f"d{vwind_var}_dphi_stderr"])**2 +                 # standard error
                                (dataset[f"dsintheta_{uwind_var}_dtheta_stderr"])**2)
        dataset["vor_stderr"].attrs["units"]     = dataset["vor"].attrs["units"]            # add units and description
        dataset["vor_stderr"].attrs["long_name"] = dataset["vor"].attrs["long_name"] +    \
                                                            " standard error"

        dataset[f"{q_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] * 1/r *      # calculate standard error 
                                                  dataset[f"d{q_var}_dphi_stderr"])**2 +    # for humidity advection
                                                 (dataset[f"{vwind_var}_mean"] * 
                                                  1/(r*np.sin(dataset["center_theta"])) *
                                                  dataset[f"d{q_var}_dtheta_stderr"])**2)
        dataset[f"{q_var}_adv_stderr"].attrs["units"]     = dataset[f"{q_var}_adv"].attrs[  # add units and description
                                                            "units"]
        dataset[f"{q_var}_adv_stderr"].attrs["long_name"] = dataset[f"{q_var}_adv"].attrs[
                                                            "long_name"] + " standard error"

        dataset[f"{t_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] * 1/r *      # calculate standard error
                                                  dataset[f"d{t_var}_dphi_stderr"])**2 +    # for temperature advection
                                                 (dataset[f"{vwind_var}_mean"] * 
                                                  1/(r*np.sin(dataset["center_theta"])) *
                                                  dataset[f"d{t_var}_dtheta_stderr"])**2)        
        dataset[f"{t_var}_adv_stderr"].attrs["units"]     = dataset[f"{t_var}_adv"].attrs[  # add units and description
                                                            "units"]
        dataset[f"{t_var}_adv_stderr"].attrs["long_name"] = dataset[f"{t_var}_adv"].attrs[
                                                            "long_name"] + " standard error"

        dataset[f"{p_var}_adv_stderr"] = np.sqrt((dataset[f"{uwind_var}_mean"] * 1/r *      # calculate standard error
                                                  dataset[f"d{p_var}_dphi_stderr"])**2 +    # for pressure advection
                                                 (dataset[f"{vwind_var}_mean"] * 
                                                  1/(r*np.sin(dataset["center_theta"])) *
                                                  dataset[f"d{p_var}_dtheta_stderr"])**2)
        dataset[f"{p_var}_adv_stderr"].attrs["units"]     = dataset[f"{p_var}_adv"].attrs[  # add units and description
                                                            "units"]
        dataset[f"{p_var}_adv_stderr"].attrs["long_name"] = dataset[f"{p_var}_adv"].attrs[
                                                            "long_name"] + " standard error"

    dataset["w_wind_stderr"] = np.sqrt((dataset["div_stderr"]**2).cumsum(skipna=True))      # calculate standard error
    dataset["w_wind_stderr"].attrs["units"]     = dataset["w_wind"].attrs["units"]          # for vertical wind
    dataset["w_wind_stderr"].attrs["long_name"] = dataset["w_wind"].attrs["long_name"] +   \
                                                  " standard error"

    dataset["omega_stderr"]  = - float(config["CONSTANTS"]["g"]) * dataset.mean_density * \
                                                  dataset["w_wind_stderr"]                  # calculate standard error
    dataset["omega_stderr"].attrs["units"]      = dataset["omega"].attrs["units"]           # for pressure velocity
    dataset["omega_stderr"].attrs["long_name"]  = dataset["omega"].attrs["long_name"] +   \
                                                  " standard error"

    return print("Finished calculating standard errors ...")

#-------------------------------------------------------------------------------------------

for idx in np.arange(len(config["FLIGHT"]["Platforms"].split(", "))):                       # loop over all specified flights 
                                                                                            # in config file

    Platform  = config["FLIGHT"]["Platforms"].split(", ")[idx]                              # adding description and unit
    Flightstr = config["FLIGHT"]["Flightstr"].split(", ")[idx]
    Datestr   = config["FLIGHT"]["Datestr"].split(", ")[idx]

    archive = config["PATH"]["data_dir"] + config["FLIGHT"]["Campaignstr"] + \
              "_" + Platform + "_Dropsondes_" + Datestr + "_" + Flightstr + "/"             # define the directory where the data 
                                                                                            # from the specified flight
                                                                                            # is stored in data_dir (defined in config)
    print(f"performing regression for {Datestr} {Platform} {Flightstr}")

    for pattern_id in config["PATTERN IDS"][Flightstr].split(", "):
        list_of_sonde_ids = config["PATTERN SONDE IDS"][Flightstr +"_" + pattern_id].split(", ")
        dataset = get_pattern(archive, list_of_sonde_ids)

        get_circle_center_coords(dataset)

        if uwind_var and vwind_var not in varname_L4:                                       # add wind variable names to list
            varname_L4 = varname_L4 + [uwind_var]                                           # of variables to process
            varname_L4 = varname_L4 + [vwind_var]

        if coordsys == "spherical":
            dataset["sintheta_"+uwind_var] = np.sin(dataset["theta"])*dataset[uwind_var]    # add sin(theta)*wind_variables
            dataset["sintheta_"+vwind_var] = np.sin(dataset["theta"])*dataset[vwind_var]    # to the dataset
            varname_L4 = varname_L4 + ["sintheta_"+uwind_var]                               # and to list of variables to process
            varname_L4 = varname_L4 + ["sintheta_"+vwind_var]                               # for nabla in spherical coordinates

        for par in tqdm(varname_L4):                                                        # loop over all variables in list

            mean_var_name = par + "_mean"                                                   # define names to save processed

            if coordsys == "carthesian":
                dvar_dx_name = "d" + par + "_dx"                                            # define names to save gradients to
                dvar_dy_name = "d" + par + "_dy"                                            # dataset for carthesian coords

            elif coordsys == "spherical":
                dvar_dtheta_name = "d" + par + "_dtheta"                                    # fit parameters
                dvar_dphi_name   = "d" + par + "_dphi"
                
            if config["REGRESSION"]["regression_method"] == "pseudoinverse":

                if coordsys == "carthesian":

                    (dataset[mean_var_name],                                                # perform least squares fit
                     dataset[dvar_dx_name],                                                 # to taylor series of variable
                     dataset[dvar_dy_name],                                                 # using the pseudoinverse
                     dataset[par + "_sounding"]) = xr.apply_ufunc(fitfunc_pinv, dataset.dx,
                                                   dataset.dy, dataset[par],
                                                   input_core_dims=[["sonde_id"], ["sonde_id"],
                                                   ["sonde_id"]], output_core_dims=[(), (),
                                                   (), ["sonde_id"]])

                elif coordsys == "spherical":
                    (dataset[mean_var_name],
                     dataset[dvar_dtheta_name],
                     dataset[dvar_dphi_name],
                     dataset[par + "_sounding"]) = xr.apply_ufunc(fitfunc_pinv, dataset.dtheta,
                                                   dataset.dphi, dataset[par],
                                                   input_core_dims=[["sonde_id"], ["sonde_id"],
                                                   ["sonde_id"]], output_core_dims=[(), (),
                                                   (), ["sonde_id"]])

            elif config["REGRESSION"]["regression_method"] == "phase_space":   

                if coordsys == "carthesian":

                    (dataset[mean_var_name],
                     dataset[dvar_dx_name], 
                     dataset[dvar_dy_name],
                     dataset[par + "_sounding"]) = fitfunc_phasespace(dataset.dx,
                                                   dataset.dy, dataset[par])
                
                if coordsys == "spherical":
                
                    (dataset[mean_var_name],
                     dataset[dvar_dtheta_name],
                     dataset[dvar_dphi_name],
                     dataset[par + "_sounding"]) = fitfunc_phasespace(dataset.dtheta, 
                                                   dataset.dphi, dataset[par])

        get_div_and_vor(dataset)
        get_vertical_velocity_and_omega(dataset)
        get_advection(dataset)
        add_std_err_terms(dataset)

        save_directory = config["PATH"]["data_dir"] + config["FLIGHT"]["Campaignstr"] + \
                         f"_{Platform}_Dropsondes_{Datestr}_{Flightstr}/Level_4/"
    
        if os.path.exists(save_directory) == False:                                         # check if directory already exists,
            os.mkdir(save_directory)                                                        # if not create it
            print("Directory for saving Level 4 data created")

        file_ending = config["LEVEL 4"]["file_ending_L4"]

        file_name = (f"{Datestr}_{Platform}_{Flightstr}_{pattern_id}{file_ending}.nc")                # save gridded dataset to netcdf file
        dataset.to_netcdf(save_directory + file_name, mode="w", 
                          format="NETCDF4", engine="netcdf4")