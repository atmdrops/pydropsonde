import ast
from dataclasses import dataclass, field, KW_ONLY

from datetime import datetime, timezone
from typing import Any, Optional, List
import os
import subprocess
import warnings
import yaml
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram

import pydropsonde.helper as hh
from pydropsonde.helper.quality import QualityControl
import pydropsonde.helper.xarray_helper as hx
import pydropsonde.helper.rawreader as rr
from importlib.metadata import version

__version__ = version("pydropsonde")

_no_default = object()


@dataclass(order=True)
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
    _serial_id: str
    cont: bool = True
    _: KW_ONLY
    _launch_time: Optional[Any] = None

    @property
    def flight_id(self):
        return self._flight_id

    @property
    def serial_id(self):
        return self._serial_id

    def set_serial_id(self, serial_id):
        self._serial_id = serial_id

    @property
    def platform_id(self):
        return self._platform_id

    @property
    def launch_time(self):
        return self._launch_time

    def set_launch_time(self, launch_time):
        self._launch_time = launch_time

    @property
    def aspen_ds(self):
        return self._aspen_ds

    def set_aspen_ds(self, value):
        self._aspen_ds = value

    @property
    def l2_ds(self):
        return self._l2_ds

    def set_l2_ds(self, ds):
        self._l2_ds = ds

    def __post_init__(self):
        """
        Initializes the 'qc' attribute as an empty object and sets the 'sort_index' attribute based on 'launch_time'.

        The 'sort_index' attribute is only applicable when 'launch_time' is available. If 'launch_time' is None, 'sort_index' will not be set.
        """
        self.qc = QualityControl()
        if self.launch_time is not None:
            self.sort_index = self.launch_time

    def add_flight_id(self, flight_id: str, flight_template: str = None) -> None:
        """Sets attribute of flight ID

        Parameters
        ----------
        flight_id : str
            The flight ID of the flight during which the sonde was launched
        """
        if flight_template is not None:
            flight_id = flight_template.format(flight_id=flight_id)

        self._flight_id = flight_id

    def add_platform_id(self, platform_id: str) -> None:
        """Sets attribute of platform ID

        Parameters
        ----------
        platform_id : str
            The platform ID of the flight during which the sonde was launched
        """
        self._platform_id = platform_id

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
            self.launch_alt = launch_alt
            self.launch_lat = launch_lat
            self.launch_lon = launch_lon
        except (ValueError, TypeError):
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
        self.launch_detect = launch_detect_bool

    def add_afile(self, path_to_afile: str) -> None:
        """Sets attribute with path to A-file of the sonde

        Parameters
        ----------
        path_to_afile : str
            Path to the sonde's A-file
        """
        self.afile = path_to_afile
        return self

    def add_level_dir(self, l0_dir: str = None, l1_dir: str = None, l2_dir: str = None):
        """
        Sets the directory paths for different data levels (Level 0, Level 1, Level 2)
        within the object. If specific directories are not provided, default paths
        are generated based on the existing 'afile' attribute or by replacing
        'Level_0' in the path with 'Level_1' or 'Level_2'.

        Parameters:
        - l0_dir (str, optional): The directory path for Level 0 data. If not provided,
        it defaults to the directory of 'afile'.
        - l1_dir (str, optional): The directory path for Level 1 data. If not provided,
        it defaults to the Level 0 directory with 'Level_0' replaced by 'Level_1'.
        Can include a placeholder '{flight_id}' for dynamic replacement.
        - l2_dir (str, optional): The directory path for Level 2 data. If not provided,
        it defaults to the Level 0 directory with 'Level_0' replaced by 'Level_2'.
        Can include a placeholder '{flight_id}' for dynamic replacement.

        Raises:
        - ValueError: If 'afile' attribute is not present in the sonde and 'l0_dir'
        is not provided.
        """
        if l0_dir is None:
            if not hasattr(self, "afile"):
                raise ValueError("No afile in sonde. Cannot continue")
            l0_dir = os.path.dirname(self.afile)
        if l1_dir is None:
            l1_dir = l0_dir.replace("Level_0", "Level_1")
        else:
            l1_dir = l1_dir.format(flight_id=self.flight_id)
        if l2_dir is None:
            l2_dir = l0_dir.replace("Level_0", "Level_2")
        else:
            l2_dir = l2_dir.format(flight_id=self.flight_id)
        self.l0_dir = l0_dir
        self.l1_dir = l1_dir
        self.l2_dir = l2_dir

    def add_broken(self, broken_sondes: dict):
        """
        Assigns a dictionary of broken sondes to this sonde.

        This method sets the 'broken_sondes' attribute of this sonde
        to the provided dictionary of broken sondes.

        Parameters:
        - broken_sondes (dict): A dictionary containing information about
        broken sondes, where keys are identifiers and values are details
        about the broken sondes.
        """
        self.broken_sondes = broken_sondes

    def run_aspen(self, path_to_postaspenfile: str = None) -> None:
        """Runs aspen and sets attribute with path to post-ASPEN file of the sonde

        If the A-file path is known for the sonde, i.e. if the attribute `path_to_afile` exists,
        then the function will attempt to look for a post-ASPEN file of the same date-time as in the A-file's name.
        Sometimes, the post-ASPEN file might not exist (e.g. because launch was not detected), and in
        such cases, ASPEN will run in a docker image and create the file.

        If the A-file path is not known for the sonde, the function will expect the argument
        `path_to_postaspenfile` to be not empty.

        Parameters
        ----------
        path_to_postaspenfile : str, optional
            The path to the post-ASPEN file. If not provided, the function will attempt to construct the path from the `afile` attribute.

        Attributes Set
        --------------
        postaspenfile : str
            The path to the post-ASPEN file. This attribute is set if the file exists at the constructed or provided path.
        """

        l0_dir = self.l0_dir  # os.path.dirname(self.afile)
        aname = os.path.basename(self.afile)
        dname = "D" + aname[1:]
        l1_dir = self.l1_dir
        l1_name = dname.split(".")[0] + "QC.nc"

        if path_to_postaspenfile is None:
            path_to_postaspenfile = os.path.join(l1_dir, l1_name)

        if not os.path.exists(path_to_postaspenfile):
            if os.path.getsize(os.path.join(l0_dir, dname)) > 0:
                os.makedirs(l1_dir, exist_ok=True)

                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--mount",
                        f"type=bind,source={l0_dir},target=/input",
                        "--mount",
                        f"type=bind,source={l1_dir},target=/output",
                        "ghcr.io/atmdrops/aspenqc:4.0.2",
                        "-i",
                        f"/input/{dname}",
                        "-n",
                        f"/output/{l1_name}",
                    ],
                    check=True,
                )
            else:
                warnings.warn(
                    f"L0 file for sonde {self.serial_id} on {self.flight_id} is empty. No processing done"
                )
        self.postaspenfile = path_to_postaspenfile
        return self

    def add_aspen_ds(self) -> None:
        """Sets attribute with an xarray Dataset read from post-ASPEN file

        The function will first check if the serial ID of the instance and that obtained from the
        global attributes of the post-ASPEN file match. If they don't, function will print out an error.

        If the `postaspenfile` attribute doesn't exist, function will print out an error
        """

        if hasattr(self, "postaspenfile"):
            try:
                ds = xr.open_dataset(self.postaspenfile, engine="netcdf4")
            except ValueError:
                warnings.warn(f"No valid l1 file for sonde {self.serial_id}")
                return None
            except OSError:
                warnings.warn(
                    f"Empty l1 file for sonde {self.serial_id} on {self.flight_id}. This might be fixed using the ASPEN software manually. "
                )
                return None
            if "SondeId" not in ds.attrs:
                if ds.attrs["SoundingDescription"].split(" ")[1] == self.serial_id:
                    self.set_aspen_ds(ds)
                else:
                    raise ValueError(
                        f"I didn't find the `SondeId` attribute, so checked the `SoundingDescription` attribute. I found the ID in the `SoundingDescription` global attribute ({ds.attrs['SoundingDescription'].split(' ')[1]}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                    )
            elif ds.attrs["SondeId"] == self.serial_id:
                self.set_aspen_ds(ds)

            elif self.launch_detect == "UGLY":
                warnings.warn(
                    f"I found the `SondeId` global attribute ({ds.attrs['SondeId']}) to not match with this instance's `serial_id` attribute ({self.serial_id}). This could be due to no afile for this sonde. Serial id is updated."
                )
                self.set_serial_id(ds.attrs["SondeId"])
                self.set_aspen_ds(ds)

            else:
                raise ValueError(
                    f"I found the `SondeId` global attribute ({ds.attrs['SondeId']}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                )
        else:
            raise ValueError(
                f"I didn't find the `postaspenfile` attribute for Sonde {self.serial_id}, therefore I can't store the xarray dataset as an attribute"
            )
        return self

    def add_aspen_history(self):
        """
        Append ASPEN processing history to the object's history attribute.

        This method retrieves the ASPEN version and processing time from the `aspen_ds` attribute
        of the sonde and appends a formatted history entry to the `history` attribute. The processing
        time is expected to be in UTC format.

        Returns:
            self: The sonde object with updated history.

        Raises:
            AssertionError: If the processing time does not end with "UTC".
        """

        history = getattr(self, "history", "")
        if hasattr(self.aspen_ds, "AspenVersion"):
            aspen_version = self.aspen_ds.AspenVersion
        else:
            aspen_version = self.aspen_ds.AvapsEditorVersion
        assert self.aspen_ds.ProcessingTime[-3:] == "UTC"

        aspen_time = datetime.strptime(
            self.aspen_ds.ProcessingTime, "%d %b %Y %H:%M %Z"
        ).replace(tzinfo=timezone.utc)

        history = (
            history
            + aspen_time.isoformat()
            + f" ASPEN processing with {aspen_version} \n"
        )
        self.history = history
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
            if not self.launch_detect:
                print(
                    f"No launch detected for Sonde {self.serial_id}. I am not running QC checks for this Sonde."
                )
            else:
                return self
        else:
            raise ValueError(
                f"The attribute `launch_detect` does not exist for Sonde {self.serial_id}."
            )

    def detect_floater(
        self,
        skip: bool = False,
    ):
        """
        Detects if a sonde is a floater.

        Parameters
        ----------
        skip : bool, optional
            If True, the function will return the object without performing any operations. Default is False.

        Returns
        -------
        self
            The object itself with the new `is_floater` entry added to the quality control object
        """
        if hh.get_bool(skip):
            return self
        else:
            if hasattr(self, "aspen_ds"):
                landing_time = self.qc.get_is_floater(aspen_ds=self.aspen_ds)
                self.landing_time = landing_time
            else:
                raise ValueError(
                    "The attribute `aspen_ds` does not exist. Please run `add_aspen_ds` method first."
                )
            return self

    def crop_aspen_ds_to_landing_time(self):
        """
        Crops the aspen_ds to the time period before landing.

        Parameters
        ----------
        None

        Returns
        -------
        self
            The object itself with the new `cropped_aspen_ds` attribute added if the sonde is a floater.
        """
        if hasattr(self.qc, "is_floater"):
            if self.qc.is_floater:
                self.cropped_aspen_ds = self.aspen_ds.sel(
                    time=slice(self.landing_time, None)
                )

        else:
            raise ValueError(
                "The attribute `is_floater` does not exist. Please run `detect_floater` method first."
            )
        return self

    def set_qc_vars(self, qc_vars=None):
        """
        set the variables for which to run the quality control.

        Parameters
        ----------
        qc_vars : list or string
            contains the Level1 variables for which to do the qc. Can be a list of strings or a string with comma-separated variable names
        """

        if qc_vars is None:
            qc_vars = ["u", "v", "rh", "ta", "p"]
        self.qc.set_qc_variables(qc_vars)
        return self

    def get_qc(self, run_qc=None):
        """
        Run qc functions.

        Parameters
        ----------
        run_qc : list or string
            contains qc functions in the QualityControl class that should be run on the qc_variables. Can be a list of strings or a string with comma-separated variable names
        """
        if run_qc is None:
            run_qc = ["profile_fullness", "near_surface_coverage", "alt_near_gpsalt"]
        elif isinstance(run_qc, str):
            run_qc = run_qc.split(",")
        ds = self.interim_l2_ds
        for fct in run_qc:
            qc_fct = getattr(self.qc, fct)
            qc_fct(ds)
        return self

    def remove_non_qc_sondes(self, used_flags=None, remove_ugly=True):
        """
        Removes sondes that do not pass quality control checks.

        This method checks the quality control (QC) status of the sonde using the specified flags.
        If the sonde passes the QC checks, it is retained; otherwise, it is filtered out.

        Parameters:
        - used_flags (optional): A list of flags to be used for the QC check. If not provided, default flags are used.
        - remove_ugly (bool, optional): A flag indicating whether drop 'UGLY' sondes.
        Defaults to True.

        Returns:
        - self: If the sonde passes the QC checks.
        - None: If the sonde fails the QC checks

        Prints a message indicating the sonde has been filtered out if it fails the QC checks.
        """
        if self.qc.check_qc(used_flags, check_ugly=remove_ugly):
            return self
        else:
            print(
                f"Quality control returned False. Therefore, filtering this sonde ({self.serial_id}) out from L2"
            )
            return None

    def create_interim_l2_ds(self):
        """
        Creates an interim L2 dataset from the aspen_ds or cropped_aspen_ds attribute.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with the interim L2 dataset added as an attribute.
        """
        if self.qc.is_floater:
            ds = self.cropped_aspen_ds
        else:
            ds = self.aspen_ds
        self.interim_l2_ds = ds

        return self

    def convert_to_si(self, variables=["rh", "p", "ta"], skip=False):
        """
        Converts variables to SI units.

        Parameters
        ----------
        variables : list or str, optional
            The variables to convert to SI units. If a string is provided, it should be a comma-separated list of variables.
            The default variables are 'rh', 'pres', and 'tdry'.

        skip : bool, optional
            If set to True, the function will skip the conversion process but will still ensure that the 'interim_l2_ds' attribute is set.
            If 'interim_l2_ds' is not already an attribute of the object, it will be set to 'aspen_ds'.
            Default is False.

        Returns
        -------
        self : object
            Returns the sonde object with the specified variables in aspen_ds converted to SI units.
            If 'skip' is set to True, it returns the sonde object with 'interim_l2_ds' set to 'aspen_ds' if it wasn't already present.
        """
        if hh.get_bool(skip):
            if hasattr(self, "interim_l2_ds"):
                return self
            else:
                self.interim_l2_ds = self.aspen_ds.copy()
                return self
        else:
            if isinstance(variables, str):
                variables = variables.split(",")

            if hasattr(self, "interim_l2_ds"):
                ds = self.interim_l2_ds
            else:
                ds = self.aspen_ds

            for variable in variables:
                func = hh.get_si_converter_function_based_on_var(variable)
                var_attrs = ds[variable].attrs
                ds = ds.assign({f"{variable}": func(ds[variable])})
                ds[variable].attrs.update(var_attrs)
            self.interim_l2_ds = ds

            return self

    def get_l2_variables(self, l2_variables: dict = hh.l2_variables):
        """
        Gets the variables from aspen_ds to create L2.

        Parameters
        ----------
        l2_variables : dict or str, optional
            A dictionary where the keys are the variables in aspen_ds to keep for L2.
            If dictionary items contain a 'rename_to' key, the variable will be renamed to the value of 'rename_to'.
            If dictionary items contain a 'attributes' key, the variable will be assigned the attributes in the value of 'attributes'.
            The default is the l2_variables dictionary from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with only the specified variables (renamed if dictionary has 'rename_to' key and attributes added if dictionary has 'attributes' key) in interim_l2_ds attribute.
            If 'interim_l2_ds' is not already an attribute of the object, it will first be set to 'aspen_ds' before reducing to the variables and renaming.
        """
        if isinstance(l2_variables, str):
            l2_variables = ast.literal_eval(l2_variables)

        l2_variables_list = list(l2_variables.keys())

        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        ds = ds[l2_variables_list]

        for variable, variable_dict in l2_variables.items():
            if "attributes" in variable_dict:
                ds[variable].attrs = variable_dict["attributes"]
            ds = ds.rename({variable: variable_dict["rename_to"]})
        self.interim_l2_ds = ds

        return self

    def get_flight_attributes(
        self, l2_flight_attributes_map: dict = hh.l2_flight_attributes_map
    ) -> None:
        """
        Gets flight attributes from the A-file and adds them to the sonde object.

        Parameters
        ----------
        l2_flight_attributes_map : dict or str, optional
            A dictionary where the keys are the flight attributes in the A-file
            and the values are the corresponding (renamed) attribute names to be used for the L2 file.
            The default is the l2_flight_attributes_map dictionary from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with the flight attributes added as attributes.
        """
        flight_attrs = {}

        with open(self.afile, "r") as f:
            lines = f.readlines()

        for attr in l2_flight_attributes_map.keys():
            for line_id, line in enumerate(lines):
                if attr in line:
                    break

            attr = l2_flight_attributes_map.get(attr, attr)
            try:
                value = lines[line_id].split("= ")[1]
                flight_attrs[attr] = float(value) if "AVAPS" not in attr else value
            except UnboundLocalError:
                print(
                    f"No flight attributes for sonde {self.serial_id} on {self.flight_id}"
                )
                break
        self.flight_attrs = flight_attrs

        return self

    def add_global_attrs(self, attributes=None):
        """
        function to add global attributes to sonde object. Whenever a dataset is written, those are added as global attributes
        Input:
        attributes : dictionary containing global attributes or None
        Returns:
        -------
        self: object
            Returns the sonde with a new attribute "global_attrs"
        """
        if attributes is None:
            attributes = {}
        self.global_attrs = attributes

        return self

    def get_sonde_attributes(self):
        """
        function to get sonde specific attributes from ASPEN software and quality control
        Parameters:
        None
        Returns:
        -------
        self: object
            Returns the sonde with a new attribute "sonde_attrs"
        """
        sonde_attrs = {
            "platform_id": self.platform_id,
            "launch_time_(UTC)": (
                str(self.aspen_ds.launch_time.values)
                if hasattr(self.aspen_ds, "launch_time")
                else np.datetime64(self.aspen_ds.base_time.values)
            ),
            "is_floater": self.qc.is_floater.__str__(),
            "sonde_serial_ID": self.serial_id,
        }
        self.sonde_attrs = sonde_attrs

        return self

    def add_sonde_id_variable(self, variable_name="sonde_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s serial_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'sonde_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing serial_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds
        attrs = {
            "descripion": "unique sonde ID",
            "long_name": "sonde identifier",
            "cf_role": "trajectory_id",
        }
        ds = ds.assign_coords({variable_name: self.serial_id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)
        self.interim_l2_ds = ds

        return self

    def add_platform_id_variable(self, variable_name="platform_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s platform_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'platform_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing platform_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        attrs = dict(
            description="unique platform ID",
            long_name="platform identifier",
        )
        ds = ds.assign_coords({variable_name: self.platform_id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)
        self.interim_l2_ds = ds
        return self

    def add_flight_id_variable(self, variable_name="flight_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s flight_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'flight_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing flight_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        attrs = dict(
            description="unique flight ID",
            long_name="flight identifier",
        )

        ds = ds.assign_coords({variable_name: self.flight_id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)
        self.interim_l2_ds = ds
        return self

    def add_l2_attributes_to_interim_l2_ds(self):
        """
        Adds flight, sonde and global attributes to interim_l2_ds.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with flight, sonde and global attributes added to interim_l2_ds.
        """
        ds = self.interim_l2_ds

        attrs_to_del = []
        for attr in ds.attrs.keys():
            attrs_to_del.append(attr)

        for attr in attrs_to_del:
            del ds.attrs[attr]

        if hasattr(self, "flight_attrs"):
            ds = ds.assign_attrs(self.flight_attrs)
        if hasattr(self, "sonde_attrs"):
            ds = ds.assign_attrs(self.sonde_attrs)
        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)

        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" quality control with pydropsonde {__version__} \n"
        )
        self.history = history
        ds = ds.assign_attrs({"history": history})
        self.interim_l2_ds = ds

        return self

    def add_qc_to_l2(
        self,
        add_var_qc=True,
        add_details=True,
    ):
        """
        Adds quality control (QC) flags to the level 2 dataset.

        This function updates the internal level 2 dataset (`interim_l2_ds`) by adding
        quality control flags. If `add_as_vars` is set to True, it adds the QC flags as
        variables within the dataset. The function assigns an overall quality flag named
        "sonde_all_qc" and updates its attributes to describe the flag values and meanings.
        It also iterates over the QC variables, adding individual QC flags and their
        details to the dataset.

        Parameters:
        - add_as_vars (bool): If True, QC flags are added as variables in the dataset.

        Returns:
        - self: The instance of the class with the updated dataset.

        """
        ds = self.interim_l2_ds
        if add_var_qc:
            for variable in self.qc.qc_vars:
                ds = self.qc.add_variable_flags_to_ds(ds, variable, details=add_details)

            ds = self.qc.add_non_var_qc_to_ds(ds)
        ds = self.qc.add_sonde_flag_to_ds(ds, "sonde_qc")
        self.interim_l2_ds = ds
        return self

    def get_l2_filename(
        self, l2_filename: str = None, l2_filename_template: str = None
    ):
        """
        Gets the L2 filename from the template provided.

        Parameters
        ----------
        l2_filename : str, optional
            The L2 filename. The default is the l2_filename_template from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 filename added as an attribute.
        """

        if l2_filename is None:
            if l2_filename_template:
                l2_filename = l2_filename_template.format(
                    platform=self.platform_id,
                    serial_id=self.serial_id,
                    flight_id=self.flight_id,
                )
            else:
                l2_filename = hh.l2_filename_template.format(
                    platform=self.platform_id,
                    serial_id=self.serial_id,
                    flight_id=self.flight_id,
                )
        self.l2_filename = l2_filename

        return self

    def write_l2(self, l2_dir: str = None):
        """
        Writes the L2 file to the specified directory.

        Parameters
        ----------
        l2_dir : str, optional
            The directory to write the L2 file to. The default is the directory of the A-file with '0' replaced by '2'.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 file written to the specified directory using the l2_filename attribute to set the name.
        """

        if l2_dir is None:
            l2_dir = self.l2_dir

        ds = self.interim_l2_ds
        if hasattr(self, "broken_sondes"):
            if self.serial_id in self.broken_sondes:
                ds.attrs.update(
                    {"comment": self.broken_sondes[self.serial_id]["error"]}
                )

        hx.write_ds(
            ds=ds,
            dir=l2_dir,
            filename=self.l2_filename,
            object_dim="sonde_id",
            alt_dim="time",
        )
        return self

    def add_l2_ds(self, l2_dir: str = None):
        """
        Adds the L2 dataset as an attribute to the sonde object.

        Parameters
        ----------
        l2_dir : str, optional
            The directory to read the L2 file from. The default is the directory of the A-file with '0' replaced by '2'.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 dataset added as an attribute.
        """
        if l2_dir is None:
            l2_dir = self.l2_dir

        try:
            self.set_l2_ds(hx.open_dataset(os.path.join(l2_dir, self.l2_filename)))

            return self
        except FileNotFoundError:
            return None

    def create_interim_l3(self):
        self.interim_l3_ds = self.l2_ds.assign_coords(
            {"sonde_id": ("sonde_id", [self.l2_ds.sonde_id.values])}
        ).sortby("time")

        return self

    def check_interim_l3(
        self,
        interim_l3_dir: str = None,
        interim_l3_filename: str = None,
        skip=False,
    ):
        if interim_l3_dir is None:
            interim_l3_dir = self.l2_dir.replace("Level_2", "Level_3_interim").replace(
                self.flight_id, ""
            )
        if interim_l3_filename is None:
            interim_l3_filename = "interim_l3_{sonde_id}_{version}.nc".format(
                sonde_id=self.serial_id, version=__version__
            )
        else:
            interim_l3_filename = interim_l3_filename.format(
                sonde_id=self.serial_id, version=__version__
            )
        self.interim_l3_dir = interim_l3_dir
        self.interim_l3_filename = interim_l3_filename
        if (not skip) and os.path.exists(
            os.path.join(interim_l3_dir, interim_l3_filename)
        ):
            self.interim_l3_ds = hx.open_dataset(
                os.path.join(interim_l3_dir, interim_l3_filename)
            )
            self.cont = False
            return self
        else:
            return self

    def add_q_and_theta_to_l2_ds(self):
        """
        Adds potential temperature and specific humidity to the L2 dataset.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with potential temperature and specific humidity added to the L2 dataset.
        """
        ds = self.interim_l3_ds

        ds = hh.calc_theta_from_T(ds)
        ds = hh.calc_q_from_rh_sonde(ds)

        self.interim_l3_ds = ds

        return self

    def recalc_rh_and_ta(self):
        ds = self.interim_l3_ds
        ds = hh.calc_T_from_theta(ds)
        ds = hh.calc_rh_from_q(ds)
        self.interim_l3_ds = ds
        return self

    def add_iwv(self):
        self.interim_l3_ds = hh.calc_iwv(self.interim_l3_ds)

        return self

    def add_thetas(self):
        ds = self.interim_l3_ds
        ds = hh.calc_theta_e(ds)
        self.interim_l3_ds = ds

        return self

    def add_wind(self):
        self.interim_l3_ds = hh.calc_wind_dir_and_speed(self.interim_l3_ds)
        return self

    def set_alt_dim(self, alt_dim="alt"):
        """
        Set the altitude dimension attribute for the sonde

        This method updates the `alt_dim` attribute of the `qc` and the sonde.
        It allows for customization of the altitude dimension used as a dimension from level 3 onwards.

        Parameters:
        alt_dim (str): The name of the altitude dimension to set. Defaults to "alt".

        Returns:
        self: Returns the sonde instance
        """
        self.qc.alt_dim = alt_dim
        self.alt_dim = alt_dim
        return self

    def replace_alt_dim(self, drop_nan=True):
        """
        Replaces the altitude dimension in the dataset if all altitude values are NaN.

        This function loads the interim level 2 dataset and checks the altitude variable for NaN values.
        If all values are NaN, it attempts to replace the altitude variable with the other (alt and gpsalt)
        Depending on the `drop_nan` parameter, it either drops sonde from the dictionary or sets the qc values.
        If the altitude values remain NaN after replacement, the dataset is dropped.

        Parameters:
        - drop_nan (bool): Determines whether to drop the dataset if altitude values are NaN.
        If True, the dataset is dropped; otherwise, it attempts to switch the altitude variable.

        Returns:
        - self: Returns the object itself if the altitude dimension is successfully replaced or remains valid.
        - None: Returns None if the dataset is dropped due to NaN altitude values.
        """
        alt_dim = self.alt_dim
        ds = self.interim_l2_ds.load()
        alt = ds[alt_dim]
        if np.all(np.isnan(alt)):
            ds = self.qc.replace_alt_var(ds, alt_dim)
            if hh.get_bool(drop_nan):
                print(
                    f"No {alt_dim} values.  Sonde {self.serial_id} from {self.flight_id} is dropped"
                )
                return None
            else:
                print(
                    f"No {alt_dim} values.  Sonde {self.serial_id} from {self.flight_id} altitude is switched"
                )
                alt = ds[alt_dim]
                if np.all(np.isnan(alt)):
                    print(
                        f"No altitude values. Sonde {self.serial_id} from {self.flight_id} is dropped"
                    )
                    return None
                self.interim_l2_ds = ds
        return self

    def swap_alt_dimension(self):
        """
        Swap the 'time' dimension with an alternative dimension (either alt or gpsalt) in the dataset.

        This method swaps the 'time' dimension of the dataset with an alternative
        dimension specified by the `alt_dim` attribute of the object. It then loads
        the dataset with the new dimension configuration and updates the object's
        internal dataset attribute `interim_l3_ds`.

        Returns:
            self: The instance of the object with the updated dataset.
        """
        alt_dim = self.alt_dim
        self.interim_l3_ds = (self.interim_l3_ds.swap_dims({"time": alt_dim})).load()
        return self

    def remove_non_mono_incr_alt(self):
        """
        This function removes the indices in the some height variable that are not monotonically increasing
        """
        alt_dim = self.alt_dim
        ds = self.interim_l3_ds.load()
        alt = ds[alt_dim]

        curr_alt = alt.isel(time=0)
        for i in range(len(alt)):
            if alt[i] > curr_alt:
                alt[i] = np.nan
            elif ~np.isnan(alt[i]):
                curr_alt = alt[i]
        ds[alt_dim] = alt

        self.interim_l3_ds = ds
        return self

    def interpolate_alt(
        self,
        interp_start=-5,
        interp_stop=14600,
        interp_step=10,
        max_gap_fill: int = 50,
        p_log=True,
        method: str = "bin",
    ):
        """
        Interpolate sonde data along comon altitude grid to prepare concatenation
        """
        alt_dim = self.alt_dim
        interpolation_grid = np.arange(interp_start, interp_stop, interp_step)
        ds = self.interim_l3_ds

        if not (ds[alt_dim].diff(dim=alt_dim) < 0).any():
            warnings.warn(
                f"your altitude for sonde {self.serial_id
                } on {self.launch_time} is not sorted."
            )
            print(ds[alt_dim].values)
        if p_log:
            ds = ds.assign(p=(ds.p.dims, np.log(ds.p.values), ds.p.attrs))
        if method == "linear_interpolate":
            interp_ds = ds.interp({alt_dim: interpolation_grid})
        elif method == "bin":
            mean_ds = {}
            count_dict = {}
            # bin variables along height, bins are right-open intervals, except the last
            for var in [
                "u",
                "v",
                "q",
                "p",
                "theta",
                "lat",
                "lon",
                "gpsalt",
                "time",
                "alt",
            ]:
                if (var in ds.variables) and (var not in ds.dims):
                    count_dict[var] = histogram(
                        ds[alt_dim].where(~np.isnan(ds[var])),
                        bins=interpolation_grid,
                        dim=[alt_dim],
                    )

                    new_ds = (
                        histogram(
                            ds[alt_dim].where(~np.isnan(ds[var])),
                            bins=interpolation_grid,
                            dim=[alt_dim],
                            weights=ds[var]
                            .astype(np.float64)
                            .where(~np.isnan(ds[var])),
                        )  # casting necessary for time
                        / count_dict[var]
                    )
                    new_ds.name = var
                    new_ds = new_ds.assign_attrs(ds[var].attrs)
                    mean_ds[var] = new_ds.copy()
            interp_ds = xr.Dataset(mean_ds)

            # interpolate missing values up to max_gap_fill meters
            interp_ds = (
                interp_ds.transpose()
                .interpolate_na(
                    dim=f"{alt_dim}_bin", max_gap=max_gap_fill, use_coordinate=True
                )
                .rename({f"{alt_dim}_bin": alt_dim})
            )
            interp_ds[alt_dim].attrs.update(ds[alt_dim].attrs)
            time_type = ds["time"].values.dtype
            interp_ds = interp_ds.assign(
                interp_time=(
                    interp_ds.time.dims,
                    interp_ds.time.astype(time_type).values,
                    interp_ds.time.attrs,
                )
            ).drop_vars("time")
            count_dict.pop("time")
            self.count_dict = count_dict

        if p_log:
            interp_ds = interp_ds.assign(
                p=(interp_ds.p.dims, np.exp(interp_ds.p.values), interp_ds.p.attrs)
            )
        self.interim_l3_ds = interp_ds
        return self

    def get_N_m_values(self):
        alt_dim = self.alt_dim
        count_dict = self.count_dict
        prep_l3 = self.interim_l3_ds

        for variable, Nvar in count_dict.items():
            N_name = f"{variable}_N_qc"
            N_attrs = dict(
                long_name=f"Number of values per bin for {variable}",
                units="1",
            )
            Nvar = Nvar.rename({alt_dim + "_bin": alt_dim})

            prep_l3 = prep_l3.assign(
                {
                    N_name: (
                        alt_dim,
                        Nvar.astype(int).values,
                        N_attrs,
                    )
                }
            )

            # cast to int
            n_mask = Nvar.where(~np.isnan(Nvar), 0)
            int_mask = prep_l3[variable].where(~np.isnan(prep_l3[variable]), 0)

            m_mask = np.invert(n_mask.astype(bool)) & int_mask.astype(bool)
            m = xr.where(Nvar > 0, x=2, y=0)
            m = xr.where(m_mask, x=1, y=m)

            m_name = f"{variable}_m_qc"
            m_attrs = {
                "long_name": f"interp method for {variable}",
                "standard_name": "status_flag",
                "flag_values": "0, 1, 2",
                "flag_meanings": "no_data interpolated_no_raw_data average_over_raw_data",
            }
            prep_l3 = prep_l3.assign(
                {
                    m_name: (
                        alt_dim,
                        m.astype(int).values,
                        m_attrs,
                    )
                }
            )
        self.interim_l3_ds = prep_l3

        return self

    def remove_N_m_duplicates(self):
        ds = self.interim_l3_ds
        nm_vars = ["lat", "u", "p", "q", "theta"]
        if "lat in ds.variables":
            np.testing.assert_array_equal(
                ds.lat_m_qc.values,
                ds.lon_m_qc.values,
                err_msg="lat_m_qc and lon_m_qc not identical",
            )
        np.testing.assert_array_equal(
            ds.u_m_qc.values,
            ds.v_m_qc.values,
            err_msg="v_m_qc and u_m_qc not identical",
        )
        np.testing.assert_array_equal(
            ds.u_N_qc.values,
            ds.v_N_qc.values,
            err_msg="v_N_qc and u_N_qc not identical",
        )

        ds = (
            ds.drop_vars(
                [f"{var}_N_qc" for var in ds.variables if var not in nm_vars],
                errors="ignore",
            )
            .drop_vars(
                [f"{var}_m_qc" for var in ds.variables if var not in nm_vars],
                errors="ignore",
            )
            .rename(
                {
                    "lat_N_qc": "gpspos_N_qc",
                    "lat_m_qc": "gpspos_m_qc",
                    "u_N_qc": "gps_N_qc",
                    "u_m_qc": "gps_m_qc",
                }
            )
        )
        self.interim_l3_ds = ds
        return self

    def add_Nm_to_vars(self):
        self.remove_N_m_duplicates()
        ds = self.interim_l3_ds

        essential_vars = ["u", "v", "q", "p", "theta", "lat", "lon"]
        mN_vars = ["gps", "gps", "q", "p", "theta", "gpspos", "gpspos"]

        for essential_var, mNvar in zip(essential_vars, mN_vars):
            ds = hx.add_ancillary_var(ds, essential_var, mNvar + "_m_qc")
            ds = hx.add_ancillary_var(ds, essential_var, mNvar + "_N_qc")
        self.interim_l3_ds = ds
        return self

    def add_ids(self):
        """
        add sonde_id, platform_id and flight_id to interim_l3_ds
        """
        ds = self.interim_l3_ds
        source_ds = self.l2_ds
        self.interim_l3_ds = ds.assign_coords(
            {
                "sonde_id": (
                    "sonde_id",
                    [source_ds.sonde_id.values],
                    source_ds.sonde_id.attrs,
                ),
                "platform_id": (
                    "sonde_id",
                    [source_ds.platform_id.values],
                    source_ds.platform_id.attrs,
                ),
                "flight_id": (
                    "sonde_id",
                    [source_ds.flight_id.values],
                    source_ds.flight_id.attrs,
                ),
            }
        )
        return self

    def add_attributes_as_var(self, essential_attrs=None):
        """
        Prepares l2 datasets to be concatenated to gridded.
        adds all attributes as variables to avoid conflicts when concatenating because attributes are different
        (and not lose information)
        """
        ds = self.interim_l3_ds
        l2_ds = self.l2_ds
        if essential_attrs is None:
            try:
                essential_attrs = hh.l3_coords
            except AttributeError:
                essential_attrs = {
                    "launch_time": {
                        "time_zone": "UTC",
                        "long_name": "dropsonde launch time",
                    }
                }

        for attr, value in l2_ds.attrs.items():
            splt = attr.split("(")
            var_name = splt[0][:-1]
            if var_name in list(essential_attrs.keys()):
                var_attrs = essential_attrs[var_name]
                ds = ds.assign({var_name: ("sonde_id", [l2_ds.attrs[attr]], var_attrs)})

        ds = ds.assign(
            dict(
                launch_time=(
                    "sonde_id",
                    ds.launch_time.astype(np.datetime64).values,
                    essential_attrs["launch_time"],
                )
            )
        )
        self.attrs = ds.attrs.keys()
        self.interim_l3_ds = ds
        return self

    def make_attr_coordinates(self):
        ds = self.interim_l3_ds
        new_coords = {
            coord: ("sonde_id", np.reshape(ds[coord].values, (1,)), ds[coord].attrs)
            for coord in hh.l3_coords
            if coord in ds.variables
        }
        self.interim_l3_ds = ds.assign_coords(new_coords)
        return self

    def add_qc_to_interim_l3(self, keep=["sonde_qc"]):
        if keep is None:
            keep = []
        elif keep == "all":
            keep = [f"{var}_qc" for var in list(self.qc.qc_by_var.keys())] + list(
                self.qc.qc_details.keys()
            )
        elif isinstance(keep, str):
            keep = keep.split(",")
        keep = keep + ["sonde_id"]
        ds_qc = self.interim_l2_ds[keep].expand_dims("sonde_id")
        self.interim_l3_ds = xr.merge([self.interim_l3_ds, ds_qc])

        return self

    def add_globals_l3(self):
        ds = self.interim_l3_ds
        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" Level 3 processing with pydropsonde {__version__} \n"
        )
        self.history = history
        ds = ds.assign_attrs({"history": history})
        self.interim_l3_ds = ds
        return self

    def save_interim_l3(self):
        ds = self.interim_l3_ds
        if hasattr(self, "broken_sondes"):
            if self.serial_id in self.broken_sondes:
                ds.attrs.update(
                    {"comment": self.broken_sondes[self.serial_id]["error"]}
                )
        hx.write_ds(
            ds=ds,
            dir=self.interim_l3_dir,
            filename=self.interim_l3_filename,
            object_dim="sonde_id",
            alt_dim=self.alt_dim,
        )

        return self

    def add_expected_coords(self):
        ds = self.interim_l3_ds
        missing_coords = set(hh.l3_coords.keys()) - set(ds.coords)
        for coord in missing_coords:
            ds = ds.assign_coords(
                {
                    coord: (
                        ("sonde_id",),
                        np.full(ds.sizes["sonde_id"], np.nan),
                        hh.l3_coords[coord],
                    )
                }
            )
        self.interim_l3_ds = ds
        return self


@dataclass(order=True)
class Gridded:
    sondes: dict
    global_attrs: dict

    @property
    def l3_ds(self):
        return self._l3_ds

    def set_l3_ds(self, ds):
        self._l3_ds = ds

    def __post_init__(self):
        if self.global_attrs is None:
            self.global_attrs = {}

    def check_aspen_version(self):
        list_of_l2_hist = [
            sonde.interim_l3_ds.attrs["history"].splitlines()[0]
            for sonde in self.sondes.values()
        ]
        aspen_versions = [asp.split("Aspen ")[1] for asp in list_of_l2_hist]
        if not aspen_versions.count(aspen_versions[0]) == len(aspen_versions):
            raise ValueError(
                "Not all sondes have been processed with the same Aspen version"
            )
        return self

    def check_pydropsonde_version(self):
        list_of_l2_hist = [
            sonde.interim_l3_ds.attrs["history"].splitlines()[1]
            for sonde in self.sondes.values()
        ]
        pydrop_versions = [pydr.split("pydropsonde ")[-1] for pydr in list_of_l2_hist]
        if not pydrop_versions.count(pydrop_versions[0]) == len(pydrop_versions):
            raise ValueError(
                "Not all sondes have been processed with the same pydropsonde version to get level 3"
            )
        return self

    def add_history_to_ds(self):
        first_sonde_history = list(self.sondes.values())[0].interim_l3_ds.attrs[
            "history"
        ]
        new_hist = ""
        for line_nb, line in enumerate(first_sonde_history.splitlines()):
            split_line = line.split(" ", 1)
            print(split_line[0])
            try:
                datetime.fromisoformat(split_line[0])
            except ValueError:
                warnings.warn(
                    f"The first part of line {line_nb} in the history is not a date. It was removed from the attribute"
                )
            new_hist += split_line[1] + "\n"
        self.history = new_hist
        return self

    def add_alt_dim(self):
        sonde = list(self.sondes.values())[0]
        self.alt_dim = sonde.alt_dim
        return self

    def check_broken(self):
        sonde = list(self.sondes.values())[0]
        if hasattr(sonde, "broken_sondes"):
            self.broken_sondes = sonde.broken_sondes
        return self

    def concat_sondes(self, sortby=None, coords=None):
        """
        function to concatenate all sondes using the combination of all measurement times and launch times and add global attributes to resulting dataset
        """
        if sortby is None:
            sortby = list(hh.l3_coords.keys())[0]
        list_of_l2_ds = [sonde.interim_l3_ds for sonde in self.sondes.values()]
        try:
            ds = xr.concat(list_of_l2_ds, dim="sonde_id", join="exact").sortby(sortby)
        except AttributeError:
            if coords is None:
                coords = hh.l3_coords
            for i, l2_ds in enumerate(list_of_l2_ds):
                missing_coords = set(coords.keys()) - set(l2_ds.coords)
                list_of_l2_ds[i] = l2_ds.assign_coords(
                    {
                        coord: (
                            ("sonde_id",),
                            np.full(ds.sizes["sonde_id"], np.nan),
                            coords[coord],
                        )
                        for coord in missing_coords
                    }
                )
            ds = xr.concat(list_of_l2_ds, dim="sonde_id", join="exact").sortby(sortby)

        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)
        self.concat_sonde_ds = ds
        return self

    def get_all_attrs(self):
        attrs = set()
        for sonde in list(self.sondes.values()):
            attrs = set(sonde.attrs) | attrs
        self.attrs = list(attrs)
        return self

    def get_l3_dir(self, l3_dir: str = None):
        if l3_dir:
            self.l3_dir = l3_dir
        elif self.sondes is not None:
            self.l3_dir = (
                list(self.sondes.values())[0]
                .l2_dir.replace("Level_2", "Level_3")
                .replace(list(self.sondes.values())[0].flight_id, "")
                .replace(list(self.sondes.values())[0].platform_id, "")
            )
        else:
            raise ValueError("No sondes and no l3 directory given, cannot continue ")
        return self

    def get_l3_filename(self, l3_filename: str = None):
        if l3_filename is None:
            l3_filename = hh.l3_filename
        else:
            l3_filename = l3_filename

        self.l3_filename = l3_filename

        return self

    def write_l3(self, l3_dir: str = None):
        """
        Writes the L3 file to the specified directory.

        Parameters
        ----------
        l3_dir : str, optional
            The directory to write the L3 file to.

        Returns
        -------
        self : object
            Returns the sonde object with the L3 file written to the specified directory using the l3_filename attribute to set the name.
        """

        if l3_dir is None:
            l3_dir = self.l3_dir
        ds = self.concat_sonde_ds
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" level3 concatenation with pydropsonde {__version__} \n"
        )
        self.history = history
        ds.attrs.update({"history": history})

        if hasattr(self, "broken_sondes"):
            ds.attrs.update({"broken_sondes": list(self.broken_sondes.keys())})

        hx.write_ds(
            ds=self.concat_sonde_ds,
            dir=l3_dir,
            filename=self.l3_filename,
            object_dim="sonde_id",
            alt_dim=self.alt_dim,
        )
        return self

    def add_l3_ds(self, l3_dir: str = None):
        if l3_dir is None:
            self.set_l3_ds(self.concat_sonde_ds.copy())
        else:
            self.set_l3_ds(hx.open_dataset(l3_dir))
        return self

    def get_simple_circle_times_from_yaml(self, yaml_file: str = None):
        with open(yaml_file) as source:
            flightinfo = yaml.load(source, Loader=yaml.SafeLoader)
        segments = []
        for c in flightinfo["segments"]:
            segments.append(
                dict(
                    segment_id=c["segment_id"],
                    platform_id=flightinfo["platform"],
                    flight_id=flightinfo["flight_id"],
                    start=c["start"],
                    end=c["end"],
                )
            )
        self.segments = segments

        return self

    def get_circle_times_from_segmentation(self, yaml_file: str = None):
        if yaml_file is None:
            warnings.warn("No segmentation file provided. No circle analysis done")
            return None
        segmentation = rr.get_flight_segmentation(yaml_file)
        platform_ids = set(self.l3_ds.platform_id.values)
        flight_ids = set(self.l3_ds.flight_id.values)
        self.segments = [
            {
                **s,
                "platform_id": platform_id,
                "flight_id": flight_id,
            }
            for platform_id in platform_ids
            for flight_id in flight_ids
            for s in segmentation.get(platform_id, {})
            .get(flight_id, {})
            .get("segments", [])
            if "circle" in s["kinds"]
        ]

        return self
