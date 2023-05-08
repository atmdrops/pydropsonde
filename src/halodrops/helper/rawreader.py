"""
Module to read from raw files, mostly to gather metadata from A files
"""
from datetime import datetime
import logging

import numpy as np

def check_launch_detect_in_afile(a_file:'str') -> bool:
    """Returns bool value of launch detect for a given A-file

    Given the path for an A-file, the function parses through the lines
    till it encounters the phrase 'Launch Obs Done?' and returns the
    boolean value for the 1 or 0 found after the '=' sign in the line with
    the aforementioned phrase.

    Parameters
    ----------
    a_file : str
        Path to A-file

    Returns
    -------
    bool
        True if launch is detected (1), else False (0)
    """
    
    with open(a_file, "r") as f:
        logging.info(f'Opened File: {a_file=}')
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "Launch Obs Done?" in line:
                line_id = i
                logging.info(f'"Launch Obs Done?" found on line {line_id=}')
                break
        
        return bool(int(lines[line_id].split('=')[1]))

def get_sonde_id(a_file:'str') -> str:
    """Returns Sonde ID for a given A-file

    Given the path for an A-file, the function parses through the lines
    till it encounters the phrase 'Sonde ID/Type' and returns the sonde ID as a string.

    The function splits the line with the aforementioned phrase at the first ':' sign.
    It takes the succeeding string and splits it again at the first ','.
    It then takes the preceding string, removes any whitespace on the left side and returns
    the string as the sonde ID.
    
    Parameters
    ----------
    a_file : str
        Path to A-file

    Returns
    -------
    str
        Sonde ID
    """
    
    with open(a_file, "r") as f:
        logging.info(f'Opened File: {a_file=}')
        lines = f.readlines()

        for i, line in enumerate(lines):
            if 'Sonde ID/Type' in line:
                logging.info(f'"Sonde ID/Type" found on line {i=}')
                break
        
        return lines[i].split(':')[1].split(',')[0].lstrip()

def get_launch_time(a_file:'str') -> np.datetime64:
    """Returns launch time for a given A-file

    Given the path for an A-file, the function parses through the lines
    till it encounters the phrase 'Launch Time (y,m,d,h,m,s)' and returns the launch time.

    The launch time is strictly defined as the time mentioned in the line with the 
    aforementioned phrase. This might lead to some discrepancies for sondes with a 
    launch detect failure, because these sondes do not have a correct launch time. For
    these sondes, since the launch detect is absent, the launch time becomes the same as
    the time when the data started being stored during the initialization phase.
    
    Parameters
    ----------
    a_file : str
        Path to A-file

    Returns
    -------
    np.datetime64
        Launch time
    """
    
    with open(a_file, "r") as f:
        logging.info(f'Opened File: {a_file=}')
        lines = f.readlines()

        for i, line in enumerate(lines):
            if 'Launch Time (y,m,d,h,m,s)' in line:
                logging.info(f'"Launch Time (y,m,d,h,m,s)" found on line {i=}')
                break
        ltime = line.split(':',1)[1].lstrip().rstrip()
        format = "%Y-%m-%d, %H:%M:%S"

        return np.datetime64(datetime.strptime(ltime, format))