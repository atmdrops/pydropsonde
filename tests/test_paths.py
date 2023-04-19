from halodrops.helper import paths
import os

main_data_directory = '../sample'
flightdate = '20200101'
l1_path = os.path.join(main_data_directory,flightdate,'Level_1')
quicklooks_path = os.path.join(main_data_directory,flightdate,'Quicklooks')

object = paths.Paths(main_data_directory,flightdate)

def test_l1_path():
    assert (object.l1dir == l1_path)    

def test_quicklooks_path():
    assert object.quicklooks_path() == quicklooks_path