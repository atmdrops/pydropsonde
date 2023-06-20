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
