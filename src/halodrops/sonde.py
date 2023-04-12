from dataclasses import dataclass, field, InitVar, KW_ONLY
from typing import Any, Optional
import numpy as np

_no_default = object()

@dataclass(order=True,frozen=True)
class Sonde:    
    """Class identifying a sonde and containing its metadata

    A `Sonde` identifies an instrument that has been deployed. This means that pre-initialization sondes do not exist for this class.

    Every `Sonde` mandatorily has a `serial_id` which is unique. Therefore, all instances with the same `serial_id` are to be considered as having the same metadata and data.

    Optionally, the `sonde` also has metadata attributes, which can be broadly classified into:

    - campaign and flight information
    - location and time information of launch
    - performance of the instrument and sensors
    - other information such as reconditioning status, signal strength, etc.
    """
    sort_index: np.datetime64 = field(init=False,repr=False)
    serial_id: str
    _ : KW_ONLY
    launch_time: Optional[Any] = None
    
    # add more attributes of other optional metadata

    def __post_init__(self):
        """The `sort_index` attribute is only applicable when `launch_time` is available.
        """
        if self.launch_time is not None:
            object.__setattr__(self, 'sort_index', self.launch_time)

@dataclass(frozen=True)
class SondeData(Sonde):
    """Class containing data of a sonde

    Parameters
    ----------
    Sonde : class
        parent class        

    Raises
    ------
    TypeError
        If data is not provided while initializing the instance, a TypeError will be raised
    """
    data: Any = _no_default

    def __post_init__(self):
        if self.data is _no_default:
            raise TypeError("No data provided! __init__ missing 1 required argument: 'data'")