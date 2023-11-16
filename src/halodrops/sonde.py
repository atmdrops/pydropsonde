from dataclasses import dataclass, field, KW_ONLY
from typing import Any, Optional, List
import os

import numpy as np
import xarray as xr

from halodrops.helper import rawreader as rr
import halodrops.helper as hh

_no_default = object()


@dataclass(order=True, frozen=True)
class Sonde:
    """Class identifying a sonde and containing its metadata

    A `Sonde` identifies an instrument that has been deployed. This means that pre-initialization sondes do not exist for this class.

    Every `Sonde` mandatorily has a `serial_id` which is unique. Therefore, all instances with the same `serial_id` are to be considered as having the same metadata and data.

    Optionally, the `sonde` also has metadata attributes, which can be broadly classified into:

    - campaign and flight information
    - location and time information of launch
    - performance of the instrument and sensors
    - other information such as reconditioning status, signal strength, etc.
    """

    sort_index: np.datetime64 = field(init=False, repr=False)
    serial_id: str
    _: KW_ONLY
    launch_time: Optional[Any] = None

    def __post_init__(self):
        """
        Initializes the 'qc' attribute as an empty object and sets the 'sort_index' attribute based on 'launch_time'.

        The 'sort_index' attribute is only applicable when 'launch_time' is available. If 'launch_time' is None, 'sort_index' will not be set.
        """
        object.__setattr__(self, "qc", type("", (), {})())
        if self.launch_time is not None:
            object.__setattr__(self, "sort_index", self.launch_time)

    def add_spatial_coordinates_at_launch(self, launch_coordinates: List) -> None:
        """Sets attributes of spatial coordinates at launch

        Expected units for altitude, latitude and longitude are
        meter above sea level, degree north and degree east, respectively.

        Parameters
        ----------
        launch_coordinates : List
            List must be provided in the order of [`launch_alt`,`launch_lat`,`launch_lon`]
        """
        try:
            launch_alt, launch_lat, launch_lon = launch_coordinates
            object.__setattr__(self, "launch_alt", launch_alt)
            object.__setattr__(self, "launch_lat", launch_lat)
            object.__setattr__(self, "launch_lon", launch_lon)
        except:
            print(
                "Check if the sonde detected a launch, otherwise launch coordinates cannot be set"
            )

    def add_launch_detect(self, launch_detect_bool: bool) -> None:
        """Sets bool attribute of whether launch was detected

        Parameters
        ----------
        launch_detect_bool : bool
            True if launch detected, else False
        """
        object.__setattr__(self, "launch_detect", launch_detect_bool)

    def add_afile(self, path_to_afile: str) -> None:
        """Sets attribute with path to A-file of the sonde

        Parameters
        ----------
        path_to_afile : str
            Path to the sonde's A-file
        """
        object.__setattr__(self, "afile", path_to_afile)
        return self

    def add_postaspenfile(self, path_to_postaspenfile: str = None) -> None:
        """Sets attribute with path to post-ASPEN file of the sonde

        If the A-file path is known for the sonde, i.e. if the attribute `path_to_afile` exists,
        then the function will attempt to look for a post-ASPEN file of the same date-time as in the A-file's name.
        Sometimes, the post-ASPEN file might not exist (e.g. because launch was not detected), and in
        such cases, an exception will be raised.

        If the A-file path is not known for the sonde, the function will expect the argument
        `path_to_postaspenfile` to be not empty.

        Parameters
        ----------
        path_to_postaspenfile : str, optional
            The path to the post-ASPEN file. If not provided, the function will attempt to construct the path from the `afile` attribute.

        Raises
        ------
        ValueError
            If the `afile` attribute does not exist when `path_to_postaspenfile` is not provided.
            If the post-ASPEN file does not exist at the constructed or provided path, and launch was detected in the A-file.
            If the launch was not detected in the A-file.

        Attributes Set
        --------------
        postaspenfile : str
            The path to the post-ASPEN file. This attribute is set if the file exists at the constructed or provided path.
        """

        if path_to_postaspenfile is None:
            if hasattr(self, "afile"):
                path_to_l1dir = os.path.dirname(self.afile)[:-1] + "1"
                postaspenfile = (
                    "D" + os.path.basename(self.afile).split(".")[0][1:] + "QC.nc"
                )
                path_to_postaspenfile = os.path.join(path_to_l1dir, postaspenfile)
                if os.path.exists(path_to_postaspenfile):
                    object.__setattr__(self, "postaspenfile", path_to_postaspenfile)
                else:
                    if rr.check_launch_detect_in_afile(self.afile):
                        raise ValueError(
                            f"The post-ASPEN file for {self.serial_id} with filename {postaspenfile} does not exist. Therefore, I am not setting the `postaspenfile` attribute. I checked and found that launch was detected for {self.serial_id}."
                        )
                    else:
                        raise ValueError(
                            f"Launch not detected for {self.serial_id}. Therefore, {postaspenfile} does not exist and I am not setting the `postaspenfile` attribute."
                        )
            else:
                raise ValueError("The attribute `path_to_afile` doesn't exist.")

        else:
            if os.path.exists(path_to_postaspenfile):
                object.__setattr__(self, "postaspenfile", path_to_postaspenfile)
            else:
                raise ValueError(
                    f"The post-ASPEN file for your provided {path_to_postaspenfile=} does not exist. Therefore, I am not setting the `postaspenfile` attribute."
                )
        return self

    def add_aspen_ds(self) -> None:
        """Sets attribute with an xarray Dataset read from post-ASPEN file

        The function will first check if the serial ID of the instance and that obtained from the
        global attributes of the post-ASPEN file match. If they don't, function will print out an error.

        If the `postaspenfile` attribute doesn't exist, function will print out an error
        """

        if hasattr(self, "postaspenfile"):
            ds = xr.open_dataset(self.postaspenfile)
            if ds.attrs["SondeId"] == self.serial_id:
                object.__setattr__(self, "aspen_ds", ds)
            else:
                raise ValueError(
                    f"I found the `SondeId` global attribute ({ds.attrs['SondeId']}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                )
        else:
            raise ValueError(
                f"I didn't find the `postaspenfile` attribute for Sonde {self.serial_id}, therefore I can't store the xarray dataset as an attribute"
            )
        return self

    def filter_no_launch_detect(self) -> None:
        """
        Filter out sondes that did not detect a launch

        The function will check if the `launch_detect` attribute exists and if it is False.
        If the attribute doesn't exist, the function will raise an error.
        If the attribute exists and is False, the function will print a no-launch detected message.
        If the attribute exists and is True, the function will return the object.

        This function serves as a checkpoint for filtering out sondes
        that did not detect a launch before running functions
        that will require `aspen_ds`, e.g. the QC functions.

        Parameters
        ----------
        None

        Returns
        -------
        self : Sonde object
            The Sonde object itself, if the launch was detected, else None

        Raises
        ------
        ValueError
            If the `launch_detect` attribute does not exist.
        """
        if hasattr(self, "launch_detect"):
            if self.launch_detect == False:
                print(
                    f"No launch detected for Sonde {self.serial_id}. I am not running QC checks for this Sonde."
                )
            else:
                return self
        else:
            raise ValueError(
                f"The attribute `launch_detect` does not exist for Sonde {self.serial_id}."
            )

    def profile_fullness(
        self,
        variable_dict={"u_wind": 4, "v_wind": 4, "rh": 2, "tdry": 2, "pres": 2},
        time_dimension="time",
        timestamp_frequency=4,
        fullness_threshold=0.8,
        add_fullness_fraction_attribute=False,
        skip=False,
    ):
        """
        Calculates the profile coverage for a given set of variables, considering their sampling frequency.

        This function assumes that the time_dimension coordinates are spaced over 0.25 seconds,
        implying a timestamp_frequency of 4 hertz. This is applicable for ASPEN-processed QC and PQC files,
        specifically for RD41.

        For each variable in the variable_dict, the function calculates the fullness fraction. If the fullness
        fraction is less than the fullness_threshold, it sets an attribute in `self.qc` named
        "profile_fullness_{variable}" to False. Otherwise, it sets this attribute to True.

        If add_fullness_fraction_attribute is True, the function also sets an attribute in `self` named
        "profile_fullness_fraction_{variable}" to the calculated fullness fraction.

        Parameters
        ----------
        variable_dict : dict, optional
            Dictionary containing the variables in `self.aspen_ds` and their respective sampling frequencies.
            The function will estimate the weighted profile-coverage for these variables.
            Default is {'u_wind':4,'v_wind':4,'rh':2,'tdry':2,'pres':2}.
        time_dimension : str, optional
            The independent dimension of the profile. Default is "time".
        timestamp_frequency : numeric, optional
            The sampling frequency of `time_dimension` in hertz. Default is 4.
        fullness_threshold : float or str, optional
            The threshold for the fullness fraction. If the calculated fullness fraction is less than this threshold,
            the profile is considered not full. Default is 0.8.
        add_fullness_fraction_attribute : bool or str, optional
            If True, the function will add the fullness fraction as an attribute to the object. Default is False.
            If provided as string, it should be possible to convert it to a boolean value with the helper get_bool function.
        skip : bool, optional
            If True, the function will return the object without performing any operations. Default is False.

        Returns
        -------
        self
            The object itself, possibly with new attributes added based on the function parameters.
        """
        if hh.get_bool(skip):
            return self
        else:
            if isinstance(fullness_threshold, str):
                fullness_threshold = float(fullness_threshold)

            for variable, sampling_frequency in variable_dict.items():
                dataset = self.aspen_ds[variable]
                weighed_time_size = len(dataset[time_dimension]) / (
                    timestamp_frequency / sampling_frequency
                )
                fullness_fraction = (
                    np.sum(~np.isnan(dataset.values)) / weighed_time_size
                )
                if fullness_fraction < fullness_threshold:
                    object.__setattr__(
                        self.qc,
                        f"profile_fullness_{variable}",
                        False,
                    )
                else:
                    object.__setattr__(
                        self.qc,
                        f"profile_fullness_{variable}",
                        True,
                    )
                if hh.get_bool(add_fullness_fraction_attribute):
                    object.__setattr__(
                        self,
                        f"profile_fullness_fraction_{variable}",
                        fullness_fraction,
                    )
            return self

    def near_surface_coverage(
        self,
        variables=["u_wind", "v_wind", "rh", "tdry", "pres"],
        alt_bounds=[0, 1000],
        alt_dimension_name="alt",
        count_threshold=50,
        add_near_surface_count_attribute=False,
        skip=False,
    ):
        """Return fraction of non-nan values in variables near surface

        Parameters
        ----------
        variables : list, optional
            List of variables to be considered, by default ["u_wind","v_wind","rh","tdry","pres"]
        alt_bounds : list, optional
            List of lower and upper bounds of altitude in meters, by default [0,1000]
        alt_dimension_name : str, optional
            Name of altitude dimension, by default "alt"

        Returns
        -------
        float
            Fraction of non-nan values in variables near surface

        Raises
        ------
        ValueError
            If the attribute `aspen_ds` does not exist.
        """
        if hh.get_bool(skip):
            return self
        else:
            if not hasattr(self, "aspen_ds"):
                raise ValueError(
                    "The attribute `aspen_ds` does not exist. Please run `add_aspen_ds` method first."
                )

            if isinstance(alt_bounds, str):
                alt_bounds = alt_bounds.split(",")
                alt_bounds = [float(alt_bound) for alt_bound in alt_bounds]
            if isinstance(count_threshold, str):
                count_threshold = int(count_threshold)
            if isinstance(variables, str):
                variables = variables.split(",")

            for variable in variables:
                dataset = self.aspen_ds[[variable, alt_dimension_name]]
                near_surface = dataset.where(
                    (dataset[alt_dimension_name] > alt_bounds[0])
                    & (dataset[alt_dimension_name] < alt_bounds[1]),
                    drop=True,
                )

                near_surface_count = np.sum(~np.isnan(near_surface[variable].values))
                if near_surface_count < count_threshold:
                    object.__setattr__(
                        self.qc,
                        f"near_surface_coverage_{variable}",
                        False,
                    )
                else:
                    object.__setattr__(
                        self.qc,
                        f"near_surface_coverage_{variable}",
                        True,
                    )
                if hh.get_bool(add_near_surface_count_attribute):
                    object.__setattr__(
                        self,
                        f"near_surface_count_{variable}",
                        near_surface_count,
                    )
            return self

    def filter_qc_fail(self, filter_flags=None):
        """
        Filters the sonde based on a list of QC flags. If any of the flags are False, the sonde will be filtered out from creating L2.
        If the sonde passes all the QC checks, the attributes listed in filter_flags will be removed from the sonde object.

        Parameters
        ----------
        filter_flags : str or list, optional
            Comma-separated string or list of QC-related attribute names to be checked. Each item can be a specific attribute name or a prefix to include all attributes starting with that prefix. You can also provide 'all_except_<prefix>' to filter all flags except those starting with '<prefix>'. If 'all_except_<prefix>' is provided, it should be the only value in filter_flags. If not provided, all QC attributes will be checked.

        Returns
        -------
        self : object
            The sonde object itself, with the attributes listed in filter_flags removed if it passes all the QC checks.

        Raises
        ------
        ValueError
            If a flag in filter_flags does not exist as an attribute of the sonde object, or if 'all_except_<prefix>' is provided in filter_flags along with other values. Please ensure that the flag names provided in 'filter_flags' correspond to existing QC attributes. If you're using a prefix to filter attributes, make sure the prefix is correct. Check your skipped QC functions or your provided list of filter flags.
        """
        all_qc_attributes = [attr for attr in dir(self.qc) if not attr.startswith("__")]

        if filter_flags is None:
            filter_flags = all_qc_attributes
        elif isinstance(filter_flags, str):
            filter_flags = filter_flags.split(",")
        elif isinstance(filter_flags, list):
            pass
        else:
            raise ValueError(
                "Invalid type for filter_flags. It must be one of the following:\n"
                "- None: If you want to filter against all QC attributes.\n"
                "- A string: If you want to provide a comma-separated list of individual flag values or prefixes of flag values.\n"
                "- A list: If you want to provide individual flag values or prefixes of flag values."
            )

        if (
            any(flag.startswith("all_except_") for flag in filter_flags)
            and len(filter_flags) > 1
        ):
            raise ValueError(
                "If 'all_except_<prefix>' is provided in filter_flags, it should be the only value."
            )

        new_filter_flags = []
        for flag in filter_flags:
            if flag.startswith("all_except_"):
                prefix = flag.replace("all_except_", "")
                new_filter_flags.extend(
                    [attr for attr in all_qc_attributes if not attr.startswith(prefix)]
                )
            else:
                new_filter_flags.extend(
                    [attr for attr in all_qc_attributes if attr.startswith(flag)]
                )

        filter_flags = new_filter_flags

        for flag in filter_flags:
            if not hasattr(self.qc, flag):
                raise ValueError(
                    f"The attribute '{flag}' does not exist in the QC attributes of the sonde object. "
                    "Please ensure that the flag names provided in 'filter_flags' correspond to existing QC attributes. "
                    "If you're using a prefix to filter attributes, make sure the prefix is correct. "
                    "Check your skipped QC functions or your provided list of filter flags."
                )
            if not bool(getattr(self.qc, flag)):
                print(
                    f"{flag} returned False. Therefore, filtering this sonde ({self.serial_id}) out from L2"
                )
                return None

        # If the sonde passes all the QC checks, remove all attributes listed in filter_flags
        for flag in filter_flags:
            delattr(self.qc, flag)

        return self
