# from halodrops import sonde
from halodrops.helper import paths
from halodrops.qc import profile


def run(data_directory, flight_id):
    paths_for_flight = paths.Paths(data_directory, flight_id)
    # Create Sondes dictionary
    Sondes = paths_for_flight.populate_sonde_instances()
    [print(f"{i}: {Sondes[i].launch_detect}") for i in list(Sondes.keys())]
    ds = Sondes[list(Sondes.keys())[1]].aspen_ds
    var = "tdry"
    return print(
        f"---------------------\n For Sonde {list(Sondes.keys())[1]} the {var} variable has a weighted profile fullness of {profile.weighted_fullness(ds,var,sampling_frequency=2):.02f}. Yay!"
    )


def run2(arg1="default1", arg2="default2"):
    return print(f"{arg1=} & {arg2=}")
