import numpy as np
import xarray as xr
import numcodecs
from pathlib import Path
from zarr.errors import ContainsGroupError
import warnings


def add_ancillary_var(ds, variable, anc_name):
    """
    add ancillary variable to xarray dataset variable
    """
    var_attrs = ds[variable].attrs
    if "ancillary_variables" in var_attrs.keys():
        anc = var_attrs["ancillary_variables"] + f" {anc_name}"
    else:
        anc = "" + f"{anc_name}"
    var_attrs.update({"ancillary_variables": anc})
    ds = ds.assign(
        {
            f"{variable}": (
                ds[variable].dims,
                ds[variable].values,
                var_attrs,
            )
        }
    )
    return ds


# encode and write files
def get_chunks(ds, var, object_dim="sonde_id", alt_dim="alt"):
    chunks = {
        object_dim: min(256, ds.sonde_id.size),
        alt_dim: min(400, ds.alt.size),
    }

    return tuple((chunks[d] for d in ds[var].dims))


def get_target_dtype(ds, var):
    if isinstance(ds[var].values.flat[0], np.floating):
        return {"dtype": "float32"}
    if np.issubdtype(type(ds[var].values.flat[0]), np.datetime64):
        return {"units": "nanoseconds since 2000-01-01", "dtype": "float32"}
    else:
        return {"dtype": ds[var].values.dtype}


def get_zarr_encoding(ds, var, **kwargs):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd")
    enc = {
        "compressor": codec,
        "chunks": get_chunks(ds, var, **kwargs),
    }
    enc.update(get_target_dtype(ds, var))
    return enc


def get_nc_encoding(ds, var, **kwargs):
    if isinstance(ds[var].values.flat[0], str):
        return {}
    else:
        enc = {
            "compression": "zlib",
            "chunksizes": get_chunks(ds, var, **kwargs),
            "fletcher32": True,
        }
        enc.update(get_target_dtype(ds, var))
        return enc


enc_map = {
    "zarr": get_zarr_encoding,
    "nc": get_nc_encoding,
}


def get_encoding(ds, filetype, exclude_vars=None, **kwargs):
    enc_fct = enc_map[filetype]
    if exclude_vars is None:
        exclude_vars = []
    enc_var = {
        var: enc_fct(ds, var, **kwargs)
        for var in ds.variables
        if var not in ds.dims
        if var not in exclude_vars
    }
    return enc_var


def open_dataset(path):
    if ".nc" in path:
        return xr.open_dataset(path)
    elif ".zarr" in path:
        return xr.open_dataset(path, engine="zarr")
    else:
        raise ValueError(f"Could not open: unrecognized filetype for {path}")


def to_file(ds, path, filetype, overwrite=False, **kwargs):
    if filetype == "nc":
        ds.to_netcdf(path, **kwargs)
    elif filetype == "zarr":
        try:
            ds.to_zarr(path, **kwargs)
        except ContainsGroupError:
            if overwrite:
                ds.to_zarr(path, mode="w", **kwargs)
            else:
                warnings.warn(f"file {path} already exists. no new file written")
    else:
        raise ValueError("Could not write: unrecognized filetype")


def write_ds(ds, dir, filename, **kwargs):
    Path(dir).mkdir(parents=True, exist_ok=True)
    if ".nc" in filename:
        filetype = "nc"
    elif ".zarr" in filename:
        filetype = "zarr"
    else:
        raise ValueError("filetype unknown")
    encoding = get_encoding(ds, filetype=filetype, **kwargs)
    to_file(
        ds=ds,
        filetype=filetype,
        path=Path(dir, filename),
        encoding=encoding,
    )
