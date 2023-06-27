from halodrops import sonde
from halodrops.helper import paths
from halodrops.qc import profile

from importlib import reload
from gogoesgone import processing as pr
from gogoesgone import zarr_access as za

reload(pr)
reload(za)

import xarray as xr
import numpy as np
import cartopy as cp
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import LAND
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
from datetime import datetime, date
import s3fs
import pandas as pd


def convert_time_to_str(time=None, time_format="%Y%m%d %H:%M:%S"):

    """
    Convert input time into desired string format.
    """

    # Ensure time is in correct format
    timestamp = (time - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    datetime_time = datetime.utcfromtimestamp(timestamp)
    str_time = datetime_time.strftime(time_format)

    return str_time


def get_mean_launch_time(ds_flight=None, time_format="%Y%m%d %H:%M:%S"):

    """
    Compute mean launch time from all sondes in the dataset.
    """

    mean_time = convert_time_to_str(ds_flight.launch_time.mean().values, time_format)

    return mean_time


def get_satellite_data(
    satellite_time="mean_launch_time",
    time_format="%Y%m%d %H:%M:%S",
    ds_flight=None,
    satellite_name="goes16",
    channel=13,
    product="ABI-L2-CMIPF",
    extent=(-62, -48, 10, 20),
):

    """
    Access satellite data for nearest time, map to lon/lat grid, and convert to dataset.
    By default use the mean launch time from dropsonde dataset.
    """

    # Get correct time for satellite data
    if satellite_time == "mean_launch_time":
        use_time = get_mean_launch_time(ds_flight=ds_flight)
    else:
        use_time = convert_time_to_str(time=satellite_time)

    # Get filepath to satellite data at nearest time.
    flist = za.nearest_time_url(use_time, time_format, channel, product, satellite_name)
    m = za.get_mapper_from_mzz(flist)

    # Select subset of satellite domain
    img = pr.Image(m)
    subset = img.subset_region_from_latlon_extents(extent, unit="degree")

    return subset


def launch_locations_map(
    ds_flight=None,
    satellite_data=None,
    save_filepath="/path/to/save/",
    color_coding_var="flight_altitude",
    color_coding_cmap="magma",
    satellite_time=None,
    extent=(-61, -52, 10, 16),
    satellite_cmap="Greys",
    satellite_vmin=280,
    satellite_vmax=300,
):

    """
    Plot dropsonde launch locations, optionally over satellite images.
    """

    fig = plt.figure(figsize=(10, 8))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot satellite image
    if satellite_data:
        sat_im = satellite_data.CMI.isel(t=0).plot(
            ax=ax,
            x="lon",
            y="lat",
            cmap=satellite_cmap,
            add_colorbar=True,
            cbar_kwargs={
                "pad": 0.1,
                "extend": "both",
                "aspect": 15,
                "shrink": 0.7,
                "label": f"{satellite_data.CMI.name} / {satellite_data.CMI.units}",
            },
            vmin=satellite_vmin,
            vmax=satellite_vmax,
            zorder=-1,
            transform=ccrs.PlateCarree(),
        )

    # Plot flight path
    ax.plot(
        ds_flight["lon"].isel(alt=-700),
        ds_flight["lat"].isel(alt=-700),
        c="red",
        linestyle=":",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # Plot launch locations
    im_launches = ax.scatter(
        ds_flight["lon"].isel(alt=-700),
        ds_flight["lat"].isel(alt=-700),
        marker="o",
        edgecolor="grey",
        s=60,
        transform=ccrs.PlateCarree(),
        c=ds_flight[color_coding_var],
        cmap=color_coding_cmap,
    )

    # Assigning axes ticks
    xticks = np.arange(-180, 180, 4)
    yticks = np.arange(-90, 90, 4)

    # Setting up the gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.2,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10, "color": "k"}
    gl.ylabel_style = {"size": 10, "color": "k"}

    # Colorbar
    g = fig.colorbar(
        im_launches, orientation="horizontal", extend="both", aspect=30, pad=0.1
    )
    g.set_label(
        f"{ds_flight[color_coding_var].name} / {ds_flight[color_coding_var].units}",
        fontsize=12,
    )
    plt.tick_params(labelsize=10)

    # Format time stamp
    title_time = convert_time_to_str(
        time=satellite_data.t[0].values, time_format="%Y-%m-%d %H:%M"
    )
    ax.set_title("")

    plt.title(
        f"Sondes {ds_flight.sonde_id.values[0]} to {ds_flight.sonde_id.values[-1]} (Satellite Time = {title_time})",
        fontsize=12,
        pad=10,
    )

    # Save figure
    save_filename = f"{save_filepath}launch-locations-{color_coding_var}-{satellite_data.platform_ID}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")


def plot_lat_time(
    ds_flight,
    color_coding_var="flight_altitude",
    color_coding_cmap="magma",
    save_filepath="/path/to/save/",
):

    """
    Plot spatio-temporal variation (lat v/s time) of selected variable.
    """

    ax = plt.figure(figsize=(12, 4))
    plt.scatter(
        ds_flight["launch_time"].values,
        ds_flight["lat"].isel(alt=-700).values,
        s=90,
        c=ds_flight[color_coding_var],
        edgecolor="grey",
        cmap=color_coding_cmap,
    )
    plt.xlim(
        np.min(ds_flight["launch_time"].values) - np.timedelta64(4, "m"),
        np.max(ds_flight["launch_time"].values) + np.timedelta64(4, "m"),
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    g = plt.colorbar()
    g.set_label(
        f"{ds_flight[color_coding_var].name} / {ds_flight[color_coding_var].units}",
        fontsize=12,
    )

    myFmt = mdates.DateFormatter("%H:%M")
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.xlabel("Time / UTC", fontsize=12)
    plt.ylabel("Latitude / $\degree$N", fontsize=12)
    plt.title(
        f"Sondes {ds_flight.sonde_id.values[0]} to {ds_flight.sonde_id.values[-1]}",
        fontsize=12,
    )

    save_filename = f"{save_filepath}spatiotemporal-variation-{color_coding_var}.png"

    plt.savefig(
        save_filename,
        dpi=300,
        bbox_inches="tight",
    )


def plot_profiles(
    ds_flight,
    r=["ta", "theta", "rh", "wspd", "wdir"],
    r_titles=[
        "T / $\degree$C",
        "$\\theta$ / K",
        "RH / %",
        "Wind speed / ms$^{-1}$",
        "Wind direction / $\degree$",
    ],
    row=1,
    col=4,
    save_filepath="/path/to/save/",
):

    """
    Plot vertical profiles of specified variables.
    """

    f, ax = plt.subplots(row, col, sharey=True, figsize=(12, 6))

    for j in range(col):
        d = ds_flight[r[j]]
        for i in range(1, len(ds_flight["launch_time"]) - 1):
            ax[j].plot(
                d.isel(sonde_id=i),
                ds_flight["alt"] / 1000,
                c="grey",
                alpha=0.25,
                linewidth=0.5,
            )

        ax[j].plot(
            np.nanmean(d, axis=0),
            ds_flight["alt"] / 1000,
            linewidth=3,
            c="k",
        )
        ax[j].set_xlabel(r_titles[j], fontsize=12)
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        if j == 0:
            ax[j].set_ylabel("Altitude / km", fontsize=12)

    plt.suptitle(
        f"Sondes {ds_flight.sonde_id.values[0]} to {ds_flight.sonde_id.values[-1]}",
        fontsize=12,
    )

    save_filename = f"{save_filepath}vertical-profiles-measured-quantities.png"

    plt.savefig(save_filename, dpi=300, bbox_inches="tight")


def drift_plots(ds_flight=None, save_filepath="/path/to/save/"):

    print("Plotting drift in lat and lon...")

    f, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    for i in range(len(ds_flight["launch_time"])):

        max_id = np.max(np.where(~np.isnan(ds_flight["lon"].isel(sonde_id=i))))

        ax[0].plot(
            ds_flight["lat"].isel(sonde_id=i)
            - ds_flight["lat"].isel(sonde_id=i).isel(alt=max_id),
            ds_flight["alt"] / 1000,
            linewidth=1.5,
            c="grey",
            alpha=0.75,
        )

        ax[0].set_xlabel("Drift in Latitude / $\degree$", fontsize=12)
        ax[0].set_ylabel("Altitude / km", fontsize=12)
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)

        ax[1].plot(
            ds_flight["lon"].isel(sonde_id=i)
            - ds_flight["lon"].isel(sonde_id=i).isel(alt=max_id),
            ds_flight["alt"] / 1000,
            linewidth=1.5,
            c="grey",
            alpha=0.75,
        )

        ax[1].set_xlabel("Drift in Longitude / $\degree$", fontsize=12)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)

    plt.suptitle(
        f"Sondes {ds_flight.sonde_id.values[0]} to {ds_flight.sonde_id.values[-1]}",
        fontsize=12,
    )

    save_filename = f"{save_filepath}drift-in-lat-lon.png"

    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
