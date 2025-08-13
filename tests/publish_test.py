import pydropsonde.helper.__init__ as hh
import pydropsonde.helper.rawreader as hr
import pydropsonde.helper.paths as hp
import pydropsonde.processor as pp
import pydropsonde.pipeline as pl
import pydropsonde


def test_version():
    pydropsonde.__version__


def test_imports():
    hh.calc_iwv
    hr.check_launch_detect_in_afile
    hp.path_to_flight_ids
    pp.Sonde
    pl.apply_method_to_dataset
