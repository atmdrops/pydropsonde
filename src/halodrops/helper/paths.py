import logging
from pathlib import Path as pp
import os.path

class Paths:
    """
    Deriving paths from the provided directory

    The input should align in terms of hierarchy and nomenclature 
    with the {doc}`Directory Structure </handbook/directory_structure>` that `halodrops` expects.
    """
    def __init__(self,directory,flightdir):
        """Creates an instance of Paths object for a given flight

        Parameters
        ----------
        `directory` : `str`
            Main data directory

        `flightdir` : `str`
            Individual flight directory name

        Attributes
        ----------
        `flightdir`
            Path to flight data directory

        `flightdirname`
            Name of flight data directory

        `l1dir`
            Path to Level-1 data directory
        """
        self.flightdir = os.path.join(directory,flightdir)
        self.flightdirname = flightdir
        self.l1dir = os.path.join(directory,flightdir,'Level_1')

        logging.info(f'Created Path Instance: Main:{directory}; Flight:{flightdir}')

    def quicklooks_path(self):
        """Path to quicklooks directory
        
        Function checks for an existing quicklooks directory, and if not found, creates one.

        Returns
        -------
        `str`
            Path to quicklooks directory
        """
        quicklooks_path_str = os.path.join(self.flightdir,'Quicklooks')
        if pp(quicklooks_path_str).exists():
            logging.info(f'Path exists: {quicklooks_path_str}')
        else:    
            pp(quicklooks_path_str).mkdir(parents=True)
            logging.info(f'Path did not exist. Created directory: {quicklooks_path_str}')
        return quicklooks_path_str