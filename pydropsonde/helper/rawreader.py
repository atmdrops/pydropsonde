"""
Module to read from raw files, mostly to gather metadata from A files
"""

from datetime import datetime
import logging
from typing import List, Optional
import os
import fsspec
import yaml

import numpy as np

# create logger
module_logger = logging.getLogger("pydropsonde.helper.rawreader")


def get_flight_segmentation(yaml_file: str):
    flight_segment_file = yaml_file
    with fsspec.open(flight_segment_file) as f:
        meta = yaml.safe_load(f)
    return meta


def find_string_in_afile(
    a_file: str, search_phrase: str, end_char="\n", number_of_records=1
) -> Optional[str]:
    if a_file is None or not os.path.getsize(a_file) > 0:
        return None

    with open(a_file, "rb") as f:
        module_logger.debug(f"Opened File: {a_file=}")

        content = f.read()
        pos = content.find(bytes(search_phrase, encoding="ascii"))
        if pos == -1:
            module_logger.debug(f"{search_phrase} not found in file")
            return None
        end_pos = pos + len(search_phrase) - 1
        for _ in range(number_of_records):
            end_pos = content.find(bytes(end_char, encoding="ascii"), end_pos + 1)
            if end_pos == -1:
                module_logger.debug(
                    f"End of search phrase not found after {search_phrase}"
                )
                return []
        module_logger.debug(f"{search_phrase} found")
        try:
            res = content[pos:end_pos].decode("ascii").strip()
        except UnicodeDecodeError:
            module_logger.warning(f"Could not decode {search_phrase} in {a_file}")
            return None
        return res


def check_launch_detect_in_afile(a_file: Optional[str]) -> Optional[bool]:
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
    search_phrase = "Launch Obs Done?"
    line = find_string_in_afile(a_file, search_phrase)
    if line is None:
        return None
    return bool(int(line.split("=")[1]))


def get_serial_id(d_file: "str") -> str:
    """Returns Sonde ID for a given D-file

    Given the path for an D-file, the function reads in the first line (header) and extracts the sonde ID.

    Parameters
    ----------
    d_file : str
        Path to D-file

    Returns
    -------
    str
        Sonde ID
    """
    try:
        with open(d_file, "r") as f:
            module_logger.debug(f"Opened File: {d_file=}")
            header = f.readline()
            return header.split(" ")[2]
    except IndexError:
        dfile_base = os.path.basename(d_file)
        return dfile_base.split(".")[0][1:]


def get_sonde_rev(a_file: str | None) -> Optional[str]:
    search_phrase = "Sonde ID/Type/Rev"
    line = find_string_in_afile(
        a_file, search_phrase, end_char=",", number_of_records=3
    )
    if line is None:
        return None
    return line.split(":")[1].split(",")[2].lstrip()


def get_launch_time(a_file: str | None) -> np.datetime64:
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
    if a_file is None or not os.path.getsize(a_file) > 0:
        return np.datetime64("NaT")

    search_phrase = "Launch Time (y,m,d,h,m,s)"
    line = find_string_in_afile(a_file, search_phrase)
    if line is None:
        return np.datetime64("NaT")
    ltime = line.split(":", 1)[1].lstrip().rstrip()
    format = "%Y-%m-%d, %H:%M:%S"

    return np.datetime64(datetime.strptime(ltime, format))


def get_spatial_coordinates_at_launch(a_file: str | None) -> List[float]:
    """Returns spatial coordinates of sonde at launch

    For the provided A-file, if the sonde has detected a launch (see `check_launch_detect_in_afile` function)
    then the function returns the altitude, latitude and longitude of the sonde at the time of detected launch
    by parsing lines having the phrases "MSL Altitude (m)", "Latitude (deg)" and "Longitude (deg)".
    Unit convention is meter above sea level, degree north and degree east.

    If the sonde has not detected a launch, an empty list will be returned.

    Parameters
    ----------
    a_file : str
        Path to A-file

    Returns
    -------
    List
        [altitude at launch, latitude at launch, longitude at launch]
    """

    if check_launch_detect_in_afile(a_file):
        with open(a_file, "r") as f:
            module_logger.debug(f"Opened File: {a_file=}")
            lines = f.readlines()

            alt_id = 0
            lat_id = 0
            lon_id = 0
            while alt_id + lat_id + lon_id < 3:
                for i, line in enumerate(lines):
                    if "MSL Altitude (m)" in line:
                        line_id = i
                        module_logger.debug(
                            f'"MSL Altitude (m)" found on line {line_id=}'
                        )
                        alt_id = 1
                        alt = float(line.split("=")[1].lstrip().rstrip())
                    elif "Latitude (deg)" in line:
                        line_id = i
                        module_logger.debug(
                            f'"Latitude (deg)" found on line {line_id=}'
                        )
                        lat_id = 1
                        lat = float(line.split("=")[1].lstrip().rstrip())
                    elif "Longitude (deg)" in line:
                        line_id = i
                        module_logger.debug(
                            f'"Longitude (deg)" found on line {line_id=}'
                        )
                        lon_id = 1
                        lon = float(line.split("=")[1].lstrip().rstrip())
                    else:
                        pass
            return [alt, lat, lon]

    else:
        return [np.nan, np.nan, np.nan]
