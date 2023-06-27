from halodrops.plotting import quicklooks as ql


# Access satellite data
satellite_image = ql.get_satellite_data(**kwargs)

# Plot launch locations over satellite images
ql.launch_locations_map(**kwargs)

# Plot longitude/time
ql.plot_lat_time(**kwargs)

# Plot vertical profiles
ql.plot_profiles(**kwargs)

# Plot dropsonde drift
ql.drift_plots(**kwargs)

# Output all quicklooks into a PDF
