from halodrops.plotting import quicklooks as ql


# Access satellite data
satellite_image = ql.get_satellite_data()

# Plot launch locations over satellite images
ql.launch_locations_map()

# Plot longitude/time
ql.plot_lat_time()

# Plot vertical profiles
ql.plot_profiles()

# Plot dropsonde drift
ql.drift_plots()

# Output all quicklooks into a PDF
