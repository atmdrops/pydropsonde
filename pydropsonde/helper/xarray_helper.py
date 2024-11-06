import warnings


def add_ancillary_var(ds, variable, anc_name):
    """
    add ancillary variable to xarray dataset variable
    """
    if variable not in ds.variables:
        warnings.warn(f"{variable} is not in dataset. No ancillary variables added")
        return ds
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
