import numpy as np


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
