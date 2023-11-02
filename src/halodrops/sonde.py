from dataclasses import dataclass, field, KW_ONLY
from typing import Any, Optional, List
import os

import numpy as np
import xarray as xr

from halodrops.helper import rawreader as rr

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
        """The `sort_index` attribute is only applicable when `launch_time` is available."""
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

    def weighted_fullness(
        self,
        variable_dict={"u_wind": 4, "v_wind": 4, "rh": 2, "tdry": 2, "pres": 2},
        time_dimension="time",
        timestamp_frequency=4,
    ):
        """Return profile-coverage for variable weighed for sampling frequency

        The assumption is that the time_dimension has coordinates spaced over 0.25 seconds,
        hence a timestamp_frequency of 4 hertz. This is true for ASPEN-processed QC and PQC files at least for RD41.

        Parameters
        ----------
        variable_dict : dict, optional
            Variable in `self.aspen_ds` with its sampling frequency whose weighted profile-coverage is to be estimated
            The default is {'u_wind':4,'v_wind':4,'rh':2,'tdry':2,'pres':2}
        sampling_frequency : numeric
            Sampling frequency of `variable` in hertz
        time_dimension : str, optional
            Name of independent dimension of profile, by default "time"
        timestamp_frequency : numeric, optional
            Sampling frequency of `time_dimension` in hertz, by default 4

        Returns
        -------
        float
            Fraction of non-nan variable values along time_dimension weighed for sampling frequency
        """
        dataset = self.aspen_ds
        variable = variable_dict.keys()
        sampling_frequency = variable_dict.values()
        weighed_time_size = len(dataset[time_dimension]) / (
            timestamp_frequency / sampling_frequency
        )
        object.__setattr__(
            self,
            "profile_coverage",
            np.sum(~np.isnan(dataset[variable].values)) / weighed_time_size,
        )

    def qc_check_profile_fullness(self, qc_threshold=0.8):
        """Return True if profile coverage is above threshold

        Parameters
        ----------
        qc_threshold : float, optional
            Threshold for profile coverage, by default 0.8

        Returns
        -------
        bool
            True if profile coverage is above threshold, else False
        """
        if hasattr(self, "profile_coverage"):
            if self.profile_coverage > qc_threshold:
                return True
            else:
                return False
        else:
            raise ValueError(
                "The attribute `profile_coverage` does not exist. Please run `weighted_fullness` method first."
            )

    def qc_check_2(self):
        pass

    def qc_check_3(self):
        pass

    def apply_qc_checks(self, qc_checks):
        qc_functions = {
            "profile_fullness": self.qc_check_profile_fullness,
            "qc_check_2": self.qc_check_2,
            "qc_check_3": self.qc_check_3,
        }

        for check in qc_checks:
            func = qc_functions.get(check)
            if func is not None:
                object.__setattr__(self, f"{check}", func())
            else:
                raise ValueError(f"The QC function '{check}' does not exist.")
