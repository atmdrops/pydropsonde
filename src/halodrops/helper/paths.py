import numpy as np
import pandas as pd

import logging
from pathlib import Path as pp
import os.path

class Paths:
    """
    Deriving paths from the provided directory

    Provide as input:
    (a) the main path for all data 
    (b) the directory name for the particular flight 

    The input should align in terms of hierarchy and nomenclature 
    with the {doc}`Directory Structure </handbook/directory_structure>` that `halodrops` expects.
    """