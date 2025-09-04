from pydropsonde.helper import paths
import os
import pytest
import numpy as np
import configparser

main_data_directory = "./example_data"
platform_id = "HALO"
flightdate = "empty_afile"
flightdate2 = "HALO-20240831a"
path_structure = "{platform}/Level_0/{flight_id}"
platform_path_structure = "{platform}/Level_0"

l1_path = os.path.join(main_data_directory, platform_id, "Level_1", flightdate)

quicklooks_path = os.path.join(
    main_data_directory, platform_id, "Quicklooks", flightdate
)


@pytest.fixture
def flight():
    flight = paths.Flight(main_data_directory, flightdate, platform_id, path_structure)
    return flight


@pytest.fixture
def platform():
    platform = paths.Platform(
        main_data_directory, platform_id, path_structure=platform_path_structure
    )
    return platform


def test_get_flight_ids(platform):
    flight_ids = platform.get_flight_ids()
    assert flightdate in flight_ids
    assert flightdate2 in flight_ids


def test_l1_path(flight):
    assert flight.l1_dir == l1_path


def test_quicklooks_path(flight):
    assert flight.quicklooks_path() == quicklooks_path


def test_raw_reader():
    config = configparser.ConfigParser()

    flight = paths.Flight(
        data_directory="example_data",
        flight_id="20200210",
        platform_id="P3",
        path_structure="{platform}/Level_0/{flight_id}",
    )
    sonde = flight.populate_sonde_instances(config=config)[0]
    assert sonde.serial_id == "192620526"
    assert sonde.launch_time == np.datetime64("2020-02-10T06:24:12.000000")
    assert sonde.sonde_rev == "A2"
    assert sonde.launch_detect
