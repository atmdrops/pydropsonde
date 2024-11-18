from dataclasses import dataclass
import numpy as np
import circle_fit as cf

_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle data for a circle on a given flight
    """

    circle_ds: str
    flight_id: str
    platform_id: str
    segment_id: str

    def get_xy_coords_for_circles(self):
        if self.circle_ds.lon.size == 0 or self.circle_ds.lat.size == 0:
            print("Empty segment: 'lon' or 'lat' is empty.")
            return None  # or some default value like [], np.array([]), etc.

        x_coor = (
            self.circle_ds.lon * 111.320 * np.cos(np.radians(self.circle_ds.lat)) * 1000
        )
        y_coor = self.circle_ds.lat * 110.54 * 1000

        # converting from lat, lon to coordinates in metre from (0,0).

        c_xc = np.full(np.size(x_coor, 1), np.nan)
        c_yc = np.full(np.size(x_coor, 1), np.nan)
        c_r = np.full(np.size(x_coor, 1), np.nan)

        for j in range(np.size(x_coor, 1)):
            a = ~np.isnan(x_coor.values[:, j])
            if a.sum() > 4:
                c_xc[j], c_yc[j], c_r[j], _ = cf.least_squares_circle(
                    [
                        (x, y)
                        for x, y in zip(x_coor.values[:, j], y_coor.values[:, j])
                        if ~np.isnan(x)
                    ]
                )

        circle_y = np.nanmean(c_yc) / (110.54 * 1000)
        circle_x = np.nanmean(c_xc) / (111.320 * np.cos(np.radians(circle_y)) * 1000)

        circle_diameter = np.nanmean(c_r) * 2

        xc = [None] * len(x_coor.T)
        yc = [None] * len(y_coor.T)

        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
        delta_y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat

        delta_x_attrs = {
            "name": "x",
            "description": "Difference of sonde longitude from mean longitude",
            "units": self.circle_ds.lon.attrs["units"],
        }
        delta_y_attrs = {
            "name": "y",
            "description": "Difference of sonde latitude from mean latitude",
            "units": self.circle_ds.lat.attrs["units"],
        }
        circle_diameter_attrs = {
            "name": "circle_diameter",
            "description": "Diameter of fitted circle for all regressed sondes in circle",
            "units": "m",
        }
        circle_lon_attrs = {
            "name": "circle_lon",
            "description": "Longitude of fitted circle for all regressed sondes in circle",
            "units": self.circle_ds.lon.attrs["units"],
        }
        circle_lat_attrs = {
            "name": "circle_lat",
            "description": "Latitude of fitted circle for all regressed sondes in circle",
            "units": self.circle_ds.lat.attrs["units"],
        }
        circle_altitude_attrs = {
            "name": "circle_altitude",
            "description": "Mean altitude of the aircraft during the circle",
            "units": self.circle_ds.alt.attrs["units"],
        }
        circle_time_attrs = {
            "name": "circle_time",
            "description": "Mean launch time of all sondes in circle",
        }

        new_vars = dict(
            circle_altitude=(
                [],
                self.circle_ds["aircraft_msl_altitude"].mean().values,
                circle_altitude_attrs,
            ),
            circle_time=(
                [],
                self.circle_ds["launch_time"].mean().values,
                circle_time_attrs,
            ),
            circle_lon=([], circle_x, circle_lon_attrs),
            circle_lat=([], circle_y, circle_lat_attrs),
            circle_diameter=([], circle_diameter, circle_diameter_attrs),
            x=(["sonde_id", "alt"], delta_x.values, delta_x_attrs),
            y=(["sonde_id", "alt"], delta_y.values, delta_y_attrs),
        )

        self.circle_ds = self.circle_ds.assign(new_vars)

        return self
