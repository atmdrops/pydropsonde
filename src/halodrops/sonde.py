from dataclasses import dataclass, field, InitVar, KW_ONLY
from typing import Any, Optional, List
import os

import numpy as np
import xarray as xr

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

    def add_postaspenfile(self, path_to_postaspenfile: str = None) -> None:
        """Sets attribute with path to post-ASPEN file of the sonde

        If the A-file path is known for the sonde, i.e. if the attribute `path_to_afile` exists,
        then the function will attempt to look for a post-ASPEN file of the same date-time as in the A-file's name.
        Sometimes, the post-ASPEN file might not exist (e.g. because launch was not detected), and in
        such cases, an exception will be printed.

        If the A-file path is not known for the sonde, the function will expect the argument
        `path_to_postaspenfile` to be not empty.

        Parameters
        ----------
        path_to_postaspenfile : str
            Path to the sonde's post-ASPEN file
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
                    print(
                        f"The post-ASPEN file for {self.serial_id} with filename {postaspenfile} does not exist. Therefore, I am not setting the `postaspenfile` attribute."
                    )
            else:
                print("The attribute `path_to_afile` doesn't exist.")

        else:
            if os.path.exists(path_to_postaspenfile):
                object.__setattr__(self, "postaspenfile", path_to_postaspenfile)
            else:
                print(
                    f"The post-ASPEN file for your provided {path_to_postaspenfile=} does not exist. Therefore, I am not setting the `postaspenfile` attribute."
                )

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


@dataclass(frozen=True)
class SondeData(Sonde):
    """Class containing data of a sonde

    Parameters
    ----------
    Sonde : class
        parent class

    Raises
    ------
    TypeError
        If data is not provided while initializing the instance, a TypeError will be raised
    """

    data: Any = _no_default

    def __post_init__(self):
        if self.data is _no_default:
            raise TypeError(
                "No data provided! __init__ missing 1 required argument: 'data'"
            )
