import glob
import logging
from pathlib import Path as pp
from typing import Dict
import os.path

from halodrops.helper import rawreader as rr
from halodrops.sonde import Sonde

# create logger
module_logger = logging.getLogger("halodrops.helper.paths")


class Paths:
    """
    Deriving paths from the provided directory

    The input should align in terms of hierarchy and nomenclature
    with the {doc}`Directory Structure </handbook/directory_structure>` that `halodrops` expects.
    """

    def __init__(self, data_directory, flight_id):
        """Creates an instance of Paths object for a given flight

        Parameters
        ----------
        `data_directory` : `str`
            Main data directory

        `flight_id` : `str`
            Individual flight directory name

        Attributes
        ----------
        `flight_id`
            Path to flight data directory

        `flight_idname`
            Name of flight data directory

        `l1dir`
            Path to Level-1 data directory
        """
        self.logger = logging.getLogger("halodrops.helper.paths.Paths")
        self.flight_id = os.path.join(data_directory, flight_id)
        self.flight_idname = flight_id
        self.l0dir = os.path.join(data_directory, flight_id, "Level_0")
        self.l1dir = os.path.join(data_directory, flight_id, "Level_1")

        self.logger.info(
            f"Created Path Instance: {self.flight_id=}; {self.flight_idname=}; {self.l1dir=}"
        )

    def get_all_afiles(self):
        """Returns a list of paths to all A-files for the given directory
        and also sets it as attribute named 'afiles_list'
        """
        a_files = glob.glob(os.path.join(self.l0dir, "A*"))
        self.afiles_list = a_files
        return a_files

    def quicklooks_path(self):
        """Path to quicklooks directory

        Function checks for an existing quicklooks directory, and if not found, creates one.

        Returns
        -------
        `str`
            Path to quicklooks directory
        """
        quicklooks_path_str = os.path.join(self.flight_id, "Quicklooks")
        if pp(quicklooks_path_str).exists():
            self.logger.info(f"Path exists: {quicklooks_path_str=}")
        else:
            pp(quicklooks_path_str).mkdir(parents=True)
            self.logger.info(
                f"Path did not exist. Created directory: {quicklooks_path_str=}"
            )
        return quicklooks_path_str

    def populate_sonde_instances(self) -> Dict:
        """Returns a dictionary of `Sonde` class instances for all A-files found in `flight_id`
        and also sets the dictionary as value of `Sondes` attribute
        """
        afiles = self.get_all_afiles()

        Sondes = {}

        for a_file in afiles:
            launch_detect = rr.check_launch_detect_in_afile(a_file)
            sonde_id = rr.get_sonde_id(a_file)
            launch_time = rr.get_launch_time(a_file)

            Sondes[sonde_id] = Sonde(sonde_id, launch_time=launch_time)
            Sondes[sonde_id].add_launch_detect(launch_detect)
            Sondes[sonde_id].add_afile(a_file)
            if launch_detect:
                Sondes[sonde_id].add_postaspenfile()
                Sondes[sonde_id].add_aspen_ds()

        object.__setattr__(self, "Sondes", Sondes)

        return Sondes
