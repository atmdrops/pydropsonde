from dataclasses import dataclass
import numpy as np
import xarray as xr
import circle_fit as cf
import pydropsonde.helper.physics as hp

_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle data for a circle on a given flight
    """

    circle_ds: str
    clon: float
    clat: float
    crad: float
    flight_id: str
    platform_id: str
    segment_id: str
    alt_dim: str
    sonde_dim: str

    def drop_vars(self, variables=["bin_average_time"]):
        """
        drop m and N variables from level 3 from circle dataset
        """
        ds = self.circle_ds
        ds = (
            ds.drop_vars(
                [f"{var}_m_qc" for var in ds.variables],
                errors="ignore",
            )
            .drop_vars(
                [f"{var}_N_qc" for var in ds.variables],
                errors="ignore",
            )
            .drop_vars(
                ["gps_m_qc", "gps_N_qc", "gpspos_N_qc", "gpspos_m_qc"], errors="ignore"
            )
            .drop_vars(
                [f"{var}_qc" for var in ["u", "v", "ta", "p", "rh"]],
                errors="ignore",
            )
            .drop_vars(
                variables,
                errors="ignore",
            )
        )
        self.circle_ds = ds

        return self

    def get_xy_coords_for_circles(self):
        """
        Calculate x and y from lat and lon relative to circle center.
        """

        if self.circle_ds.lon.size == 0 or self.circle_ds.lat.size == 0:
            print(f"Empty segment {self.segment_id}: 'lon' or 'lat' is empty.")
            return None  # or some default value like [], np.array([]), etc.

        x_coor = (
            self.circle_ds.lon * 111.32 * np.cos(np.radians(self.circle_ds.lat)) * 1000
        )
        y_coor = self.circle_ds.lat * 110.54 * 1000

        # converting from lat, lon to coordinates in metre from (0,0).
        if self.clat is None:
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

            self.clat = np.nanmean(c_yc) / (110.54 * 1000)
            self.clon = np.nanmean(c_xc) / (
                111.32 * np.cos(np.radians(self.clat)) * 1000
            )

            self.crad = np.nanmean(c_r)
            self.method = "fitted circle for all regressed sondes in circle (mean)"
        else:
            self.method = "circle from flight segmentation"

        yc = self.clat * 110.54 * 1000
        xc = self.clon * (111.32 * np.cos(np.radians(self.clat)) * 1000)

        delta_x = x_coor - xc
        delta_y = y_coor - yc

        delta_x_attrs = {
            "long_name": "x",
            "description": "Distance of sonde longitude to mean circle longitude",
            "units": "m",
        }
        delta_y_attrs = {
            "long_name": "y",
            "description": "Distance of sonde latitude to mean circle latitude",
            "units": "m",
        }

        self.circle_ds = self.circle_ds.assign(
            dict(
                x=([self.sonde_dim, self.alt_dim], delta_x.values, delta_x_attrs),
                y=([self.sonde_dim, self.alt_dim], delta_y.values, delta_y_attrs),
            )
        )

        return self

    def add_circle_variables_to_ds(self):
        """
        Add circle metadata to the circle dataset.
        """
        circle_radius_attrs = {
            "long_name": "circle_radius",
            "description": f"Radius of {self.method}",
            "units": "m",
        }
        circle_lon_attrs = {
            "long_name": "circle_lon",
            "description": f"Longitude of {self.method}",
            "units": self.circle_ds.lon.attrs["units"],
        }
        circle_lat_attrs = {
            "long_name": "circle_lat",
            "description": f"Latitude of {self.method}",
            "units": self.circle_ds.lat.attrs["units"],
        }
        circle_altitude_attrs = {
            "long_name": "circle_altitude",
            "description": "Mean altitude of the aircraft during the circle",
            "units": self.circle_ds[self.alt_dim].attrs["units"],
        }
        circle_time_attrs = {
            "long_name": "circle_time",
            "description": "Mean launch time of all sondes in circle",
        }
        self.circle_ds = self.circle_ds.assign(
            dict(
                circle_altitude=(
                    [],
                    self.circle_ds["aircraft_msl_altitude"].mean().values,
                    circle_altitude_attrs,
                ),
                circle_time=(
                    [],
                    self.circle_ds["sonde_time"].isel(sonde=[0, -1]).mean().values,
                    circle_time_attrs,
                ),
                circle_lon=([], self.clon, circle_lon_attrs),
                circle_lat=([], self.clat, circle_lat_attrs),
                circle_radius=([], self.crad, circle_radius_attrs),
            )
        )
        return self

    def interpolate_na(self, method="cubic", max_gap=1500):
        if method is not None:
            ds = self.circle_ds
            alt_dim = self.alt_dim
            ds["p"] = np.log(ds["p"])
            ds = ds.interpolate_na(
                dim=alt_dim,
                method=method,
                bounds_error=False,
                fill_value=np.nan,
                max_gap=int(max_gap),
            )
            ds["p"] = np.exp(ds["p"])
            self.circle_ds = ds

        return self

    def remove_values(self, n_gap=3, keep_sfc=1000):
        n_gap = int(n_gap)
        ds = self.circle_ds

        alt_mask = np.full(ds.u.shape, False)  # first sonde_id then alt_dim
        alt_mask[:, ::n_gap] = True
        if keep_sfc:
            alt_mask[:, : int(keep_sfc / 10)] = True
        for var in ["u", "v", "rh", "q", "ta", "theta", "x", "y"]:
            self.circle_ds = self.circle_ds.assign(
                {var: (ds[var].dims, ds[var].where(alt_mask).values, ds[var].attrs)}
            )
        return self

    def one_gap_one_sonde(self, alt=1500, depth=500, sonde_id=2):
        ds = self.circle_ds
        alt_mask = np.full(ds.u.shape, True)
        alt_mask[
            int(sonde_id), int(int(alt) / 10) : int((int(alt) + int(depth)) / 10)
        ] = False

        for var in ["u", "v", "rh", "q", "ta", "theta", "x", "y"]:
            self.circle_ds = self.circle_ds.assign(
                {var: (ds[var].dims, ds[var].where(alt_mask).values, ds[var].attrs)}
            )
        return self

    def remove_sonde(self, sonde_id=0):
        ds = self.circle_ds
        alt_mask = np.full(ds.u.shape, True)
        alt_mask[int(sonde_id), :] = False

        for var in ["u", "v", "rh", "q", "ta", "theta", "x", "y"]:
            self.circle_ds = self.circle_ds.assign(
                {var: (ds[var].dims, ds[var].where(alt_mask).values, ds[var].attrs)}
            )
        return self

    @staticmethod
    def fit2d(x, y, u, w):
        a = np.stack([np.ones_like(x), x, y], axis=-1)

        invalid = np.isnan(u) | np.isnan(x) | np.isnan(y)
        # remove values where fewer than 6 sondes are present. Depending on the application, this might be changed.
        under_constraint = np.sum(~invalid, axis=-1) < 6
        u_cal = np.where(invalid, 0, u)
        a[invalid] = 0
        w = np.sqrt(w)
        a = np.einsum("...m,...mr->...mr", w, a)
        u_cal = np.einsum("...m,...m->...m", w, u_cal)

        a_inv = np.linalg.pinv(a)
        intercept, dux, duy = np.einsum("...rm,...m->r...", a_inv, u_cal)

        intercept[under_constraint] = np.nan
        dux[under_constraint] = np.nan
        duy[under_constraint] = np.nan
        return intercept, dux, duy

    def fit2d_xr(self, x, y, u, w, sonde_dim="sonde"):
        return xr.apply_ufunc(
            self.__class__.fit2d,  # Call the static method without passing `self`
            x,
            y,
            u,
            w,
            input_core_dims=[
                [sonde_dim],
                [sonde_dim],
                [sonde_dim],
                [sonde_dim],
            ],  # Specify input dims
            output_core_dims=[(), (), ()],  # Output dimensions as scalars
        )

    def apply_fit2d(self, variables=None):
        if variables is None:
            variables = ["u", "v", "q", "ta", "p", "rh", "theta"]
        alt_var = self.alt_dim
        alt_attrs = self.circle_ds[alt_var].attrs

        assign_dict = {}

        for par in variables:
            long_name = self.circle_ds[par].attrs.get("long_name")
            standard_name = self.circle_ds[par].attrs.get("standard_name")
            varnames = ["mean_" + par, "d" + par + "dx", "d" + par + "dy"]
            var_units = self.circle_ds[par].attrs.get("units", None)
            long_names = [
                "circle mean of " + long_name,
                "zonal gradient of " + long_name,
                "meridional gradient of " + long_name,
            ]
            use_names = [
                standard_name + "_circle_mean",
                "derivative_of_" + standard_name + "_wrt_x",
                "derivative_of_" + standard_name + "_wrt_y",
            ]
            try:
                results = self.fit2d_xr(
                    x=self.circle_ds.x,
                    y=self.circle_ds.y,
                    u=self.circle_ds[par],
                    w=self.circle_ds[f"{par}_weights"],
                    sonde_dim=self.sonde_dim,
                )
            except KeyError:
                results = self.fit2d_xr(
                    x=self.circle_ds.x,
                    y=self.circle_ds.y,
                    u=self.circle_ds[par],
                    w=xr.ones_like(self.circle_ds[par]),
                    sonde_dim=self.sonde_dim,
                )

            for varname, result, long_name, use_name in zip(
                varnames, results, long_names, use_names
            ):
                if "mean" in varname:
                    assign_dict[varname] = (
                        [alt_var],
                        result.data,
                        {
                            "long_name": long_name,
                            "units": var_units,
                        },
                    )
                else:
                    assign_dict[varname] = (
                        [alt_var],
                        result.data,
                        {
                            "standard_name": use_name,
                            "long_name": long_name,
                            "units": f"{var_units} m-1",
                        },
                    )

            ds = self.circle_ds.assign(assign_dict)
        ds[alt_var].attrs.update(alt_attrs)

        self.circle_ds = ds
        return self

    def add_density(self):
        """
        Calculate and add the density to the circle dataset.

        This method computes each sondes density.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        assert ds.p.attrs["units"] == "Pa"
        assert ds.ta.attrs["units"] == "K"
        density = hp.density(
            ds.p,
            ds.ta,
            hp.q2mr(ds.q),
        )
        density_attrs = {
            "standard_name": "air_density",
            "long_name": "Air density",
            "units": "kg m-3",
        }
        self.circle_ds = ds.assign(
            dict(
                density=(ds.ta.dims, density.values, density_attrs),
            )
        )
        return self

    def add_divergence(self):
        """
        Calculate and add the divergence to the circle dataset.

        This method computes the area-averaged horizontal mass divergence.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        D = ds.dudx + ds.dvdy
        D_attrs = {
            "standard_name": "divergence_of_wind",
            "long_name": "Area-averaged horizontal mass divergence",
            "units": "s-1",
        }
        self.circle_ds = ds.assign(div=(ds.dudx.dims, D.values, D_attrs))
        return self

    def add_vorticity(self):
        """
        Calculate and add the vorticity to the circle dataset.

        This method computes the area-averaged horizontal vorticity.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        vor = ds.dvdx - ds.dudy
        vor_attrs = {
            "standard_name": "atmosphere_relative_vorticity",
            "long_name": "Area-averaged horizontal relative vorticity",
            "units": "s-1",
        }
        self.circle_ds = ds.assign(vor=(ds.dudx.dims, vor.values, vor_attrs))
        return self

    def add_omega(self):
        """
        Calculate vertical pressure velocity as
        \int div dp

        This calculates the vertical pressure velocity as described in
        Bony and Stevens 2019

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        alt_dim = self.alt_dim
        div = ds.div.where(~np.isnan(ds.div), drop=True).sortby(alt_dim)
        p = ds.mean_p.where(~np.isnan(ds.div), drop=True).sortby(alt_dim)
        zero_vel = xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]})
        pres_diff = xr.concat([zero_vel, p.diff(dim=alt_dim)], dim=alt_dim)
        del_omega = -div * pres_diff.values
        omega = del_omega.cumsum(dim=alt_dim) * 0.01 * 60**2
        omega_attrs = {
            "standard_name": "vertical_air_velocity_expressed_as_tendency_of_pressure",
            "long_name": "Area-averaged atmospheric pressure velocity (omega)",
            "units": "hPa hr-1",
        }
        self.circle_ds = ds.assign(
            dict(omega=(ds.div.dims, omega.broadcast_like(ds.div).values, omega_attrs))
        )
        return self

    def add_wvel(self):
        """
        Calculate vertical velocity as
        - int diff dz

        This calculates the vertical velocity from omega

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        alt_dim = self.alt_dim
        div = ds.div.where(~np.isnan(ds.div), drop=True).sortby(alt_dim)
        zero_vel = xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]})

        height = xr.concat([zero_vel, div[alt_dim]], dim=alt_dim)
        height_diff = height.diff(dim=alt_dim)

        del_w = -div * height_diff.values

        w_vel = del_w.cumsum(dim=alt_dim)
        wvel_attrs = {
            "standard_name": "upward_air_velocity",
            "long_name": "Area-averaged atmospheric vertical velocity",
            "units": "m s-1",
        }
        self.circle_ds = ds.assign(
            dict(wvel=(ds.omega.dims, w_vel.broadcast_like(ds.div).values, wvel_attrs))
        )
        return self
