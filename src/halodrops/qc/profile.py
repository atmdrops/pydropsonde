import logging
import numpy as np

# create logger
module_logger = logging.getLogger("halodrops.qc.fullness")


def fullness(dataset, variable, time_dimension="time"):
    """Return the profile-coverage for variable

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing variable along time_dimension
    variable : str
        Variable in xr.Dataset whose profile-coverage is to be estimated
    time_dimension : str, optional
        Name of independent dimension of profile, by default "time"

    Returns
    -------
    float
        Fraction of non-nan variable values along time_dimension
    """
    return np.sum(~np.isnan(dataset[variable].values)) / len(dataset[time_dimension])


def weighted_fullness(dataset, variable, sampling_frequency, time_dimension="time"):
    """Return profile-coverage for variable weighed for sampling frequency

    The assumption is that the time_dimension has coordinates spaced over 0.25 seconds,
    which is true for ASPEN-processed QC and PQC files at least for RD41.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing variable along time_dimension
    variable : str
        Variable in xr.Dataset whose weighted profile-coverage is to be estimated
    sampling_frequency : numeric
        Sampling frequency of `variable` in hertz
    time_dimension : str, optional
        Name of independent dimension of profile, by default "time"

    Returns
    -------
    float
        Fraction of non-nan variable values along time_dimension weighed for sampling frequency
    """
    # 4 is the number of timestamps every second, read assumption in description
    weighed_time_size = len(dataset[time_dimension]) / (4 / sampling_frequency)
    return np.sum(~np.isnan(dataset[variable].values)) / weighed_time_size


def weighted_fullness_for_config_vars(dataset, config_file_path, add_to_dataset=True):
    """Return weighted fullness for all variables in a provided config file

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing variable along time_dimension
    config_file : str
        Path to config file
    add_to_dataset : bool, optional
        Should values be added to the provided dataset? by default True

    Returns
    -------
    xr.Dataset or dictionary
        if True, returns weighted fullness as variables in provided dataset, else returns them as a dictionary,
    """
    # Reading the CONFIG file
    import configparser

    config = configparser.ConfigParser()
    config.read(config_file_path)

    vars = [
        (var, int(config["sampling - frequencies"][var]))
        for var in config["sampling - frequencies"].keys()
    ]

    dict = {}
    for var in vars:
        dict[f"{var[0]}_weighted_fullness"] = weighted_fullness(dataset, var[0], var[1])

    if add_to_dataset:
        return dataset.assign(dict)
    else:
        return dict
