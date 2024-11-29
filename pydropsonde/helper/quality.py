import numpy as np
import warnings


class QualityControl:
    """
    Helper class to handle quality control functions and flags in a sonde object
    """

    def __init__(
        self,
    ) -> None:
        self.qc_vars = []
        self.qc_flags = {}
        self.qc_details = {}
        self.qc_by_var = {}

    def set_qc_variables(self, qc_variables):
        self.qc_vars = self.qc_vars + list(qc_variables)
        for variable in self.qc_vars:
            self.qc_by_var.update({variable: dict(qc_flags={}, qc_details={})})

    def get_is_floater(
        self,
        aspen_ds,
        gpsalt_threshold: float = 25,
        consecutive_time_steps: int = 3,
    ):
        """
        Add a qc flag, whether a sonde is a floater to a qc_flag object

        Parameters
        ----------
        gpsalt_threshold : float, optional
            The gpsalt altitude below which the sonde will check for time periods when gpsalt and pres have not changed. Default is 25.
        consecutive_time_steps : float, optional
            The number of timestapes that have to be at roughly  the same height and pressure to set landing time. default is 3

        Return
        ------
        Estimated landing time for floater or None
        """
        gpsalt_threshold = float(gpsalt_threshold)

        surface_ds = (
            aspen_ds.where(aspen_ds.gpsalt < gpsalt_threshold, drop=True)
            .sortby("time")
            .dropna(dim="time", how="any", subset=["pres", "gpsalt"])
        )
        gpsalt_diff = np.diff(surface_ds.gpsalt)
        pressure_diff = np.diff(surface_ds.pres)
        gpsalt_diff_below_threshold = (
            np.abs(gpsalt_diff) < 1
        )  # GPS altitude value at surface shouldn't change by more than 1 m
        pressure_diff_below_threshold = (
            np.abs(pressure_diff) < 1
        )  # Pressure value at surface shouldn't change by more than 1 hPa
        floater = gpsalt_diff_below_threshold & pressure_diff_below_threshold
        if np.any(floater):
            self.is_floater = True
        else:
            self.is_floater = False
            return None
        for time_index in range(len(floater) - consecutive_time_steps + 1):
            if np.all(floater[time_index : time_index + consecutive_time_steps]):
                landing_time = surface_ds.time[time_index - 1].values
                print(
                    f"{aspen_ds.attrs['SondeId']}: Floater detected! The landing time is estimated as {landing_time}."
                )
                return landing_time
        print(
            f"{aspen_ds.attrs['SondeId']}: Floater detected! However, the landing time could not be estimated. Therefore setting landing time as {surface_ds.time[0].values}"
        )
        return surface_ds.time[0].values

    def profile_fullness(
        self,
        ds,
        variable_dict={"u": 4, "v": 4, "rh": 2, "ta": 2, "p": 2},
        time_dimension="time",
        timestamp_frequency=4,
        fullness_threshold=0.8,
    ):
        """
        Calculates the profile coverage for a given set of variables, considering their sampling frequency.

        This function assumes that the time_dimension coordinates are spaced over 0.25 seconds,
        implying a timestamp_frequency of 4 hertz. This is applicable for ASPEN-processed QC and PQC files,
        specifically for RD41.

        For each variable in the variable_dict that is in self.qc_vars, the function calculates the fullness fraction. If the fullness
        fraction is less than the fullness_threshold, it sets the entry "profile_fullness_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in the variable_dict  that is in self.qc_vars, the function adds the fullness fraction to the qc_details dictionary

        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
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


        """
        var_keys = set(variable_dict.keys())
        if set(var_keys) != set(self.qc_vars):
            var_keys = set(var_keys) & set(self.qc_vars)
            warnings.warn(
                f"variables for which frequency is given do not match the qc_variables. Continue for the intersection  {var_keys}"
            )

        for variable in var_keys:
            dataset = ds[variable]
            sampling_frequency = variable_dict[variable]
            weighed_time_size = len(dataset[time_dimension]) / (
                timestamp_frequency / sampling_frequency
            )
            fullness_fraction = np.sum(~np.isnan(dataset.values)) / weighed_time_size
            if fullness_fraction < fullness_threshold:
                self.qc_flags[f"{variable}_profile_fullness"] = False
            else:
                self.qc_flags[f"{variable}_profile_fullness"] = True
            self.qc_details[f"{variable}_profile_fullness_fraction"] = fullness_fraction

    def near_surface_coverage(
        self,
        ds,
        alt_bounds=[0, 1000],
        alt_dim="gpsalt",
        count_threshold=50,
    ):
        """
        Calculates the fraction of non-null values in specified variables near the surface.


        For each variable in self.qc_variable, the function calculates the near surface count. If the near surface count is less than
        the fullness_threshold, it sets the entry "near_surface_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the near surface count to the qc_details dictionary


        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
        alt_bounds : list, optional
            The lower and upper bounds of altitude in meters to consider for the calculation. Defaults to [0,1000].
        alt_dim : str, optional
            The name of the altitude dimension. Defaults to "alt". If the sonde is a floater, this will be set to "gpsalt" regardless of user-provided value.
        count_threshold : int, optional
            The minimum count of non-null values required for a variable to be considered as having near surface coverage. Defaults to 50.


        """

        count_threshold = int(count_threshold)

        if isinstance(alt_bounds, str):
            alt_bounds = alt_bounds.split(",")
            alt_bounds = [float(alt_bound) for alt_bound in alt_bounds]
        try:
            if self.is_floater and not (alt_dim == "gpsalt"):
                warnings.warn(
                    f"{ds.attrs['SondeId']} was detected as a floater but you did not chose gpsalt as altdim in the near surface coverage qc"
                )
        except KeyError:
            warnings.warn(
                f"{ds.attrs['SondeId']} has not been checked for being a floater. Please run is_floater first."
            )

        for variable in self.qc_vars:
            dataset = ds.where(
                (ds[alt_dim] > alt_bounds[0]) & (ds[alt_dim] < alt_bounds[1]), drop=True
            )
            near_surface_count = dataset[variable].count()
            if near_surface_count < count_threshold:
                self.qc_flags[f"{variable}_near_surface"] = False

            else:
                self.qc_flags[f"{variable}_near_surface"] = True
            self.qc_details[f"{variable}_near_surface_count"] = (
                near_surface_count.values
            )

    def alt_near_gpsalt(self, ds, diff_threshold=150):
        """
        Calculates the mean difference between msl altitude and gpsaltitude in the dataset

        For each variable in self.qc_variable, the function calculates the mean difference between altitude and gpsaltitude.
        If the difference is greater than the diff_threshold, it sets the entry "mean_alt_gpsalt_diff" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the mean alt to gpsalt difference to the qc_details dictionary


        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
        diff_threshold : accepted difference between altitude and gpsaltitude. Default is 150m

        """

        dataset = ds[["alt", "gpsalt"]]
        max_diff = np.abs((dataset.alt - dataset.gpsalt).max(skipna=True))
        if max_diff > diff_threshold:
            self.qc_flags["alt_near_gpsalt"] = False
        else:
            self.qc_flags["alt_near_gpsalt"] = True
        self.qc_details["alt_near_gpsalt_max_diff"] = max_diff.values

    def check_qc(self, used_flags=None, check_ugly=True):
        """
        check if any qc check has failed.
        If any has failed, return False, if not True

        Parameters:
        -----------
        used_flags: string or list
            list of qc flags to check
        """
        if used_flags is None:
            used_flags = []
        elif used_flags == "all":
            used_flags = list(self.qc_flags.keys()).copy()
        elif isinstance(used_flags, str):
            used_flags = used_flags.split(",")
            if (len(used_flags) == 1) and used_flags[0].startswith("all_except_"):
                all_flags = self.qc_flags.copy()
                all_flags.pop(used_flags[0].replace("all_except_", ""))
                used_flags = all_flags.copy()
            elif used_flags[0].startswith("all_except_"):
                raise ValueError(
                    "If 'all_except_<prefix>' is provided in filter_flags, it should be the only value."
                )
        if not all(flag in self.qc_flags for flag in used_flags):
            raise ValueError(
                "not all flags are in the qc dict. please check you ran all qc tests"
            )

        used_flags = {key: self.qc_flags[key] for key in used_flags}
        if check_ugly and all(used_flags.values()):
            return True
        elif (not check_ugly) and any(used_flags.values()):
            return True
        else:
            return False
