from dataclasses import dataclass, field, InitVar, KW_ONLY
from typing import Any, Optional, List
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
    
    def __post_init__(self):
        """The `sort_index` attribute is only applicable when `launch_time` is available.
        """
        if self.launch_time is not None:
            object.__setattr__(self, 'sort_index', self.launch_time)

    def add_spatial_coordinates_at_launch(self,launch_coordinates:List) -> None:
        """Sets attributes of spatial coordinates at launch

        Expected units for altitude, latitude and longitude are
        meter above sea level, degree north and degree east, respectively.

        Parameters
        ----------
        launch_coordinates : List
            List must be provided in the order of [`launch_alt`,`launch_lat`,`launch_lon`]
        """
        launch_alt, launch_lat, launch_lon = launch_coordinates
        object.__setattr__(self, 'launch_alt', launch_alt)
        object.__setattr__(self, 'launch_lat', launch_lat)
        object.__setattr__(self, 'launch_lon', launch_lon)
    
    def add_launch_detect(self,launch_detect_bool:bool) -> None:
        """Sets bool attribute of whether launch was detected
             
        Parameters
        ----------
        launch_detect_bool : bool
            True if launch detected, else False            
        """
        object.__setattr__(self, 'launch_detect', launch_detect_bool)

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