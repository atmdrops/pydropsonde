from halodrops.plotting import quicklooks as ql


# Access satellite data
satellite_image = ql.get_satellite_data(ds_flight=ds_sondes_first_circle_Jan24)


# Plot launch locations over satellite images
ql.launch_locations_map(
    ds_flight=ds_sondes_first_circle_Jan24,
    satellite_data=satellite_image,
    save_filepath="/path/to/save/",
)

# Plot longitude/time
ql.plot_lat_time(ds_flight=ds_sondes_first_circle_Jan24, save_filepath=".")

# Plot vertical profiles
ql.plot_profiles(ds_flight=ds_sondes_first_circle_Jan24, save_filepath=".")

# Plot dropsonde drift
ql.drift_plots(ds_flight=ds_sondes_first_circle_Jan24, save_filepath=".")
