import logging
from pathlib import Path

# create halodrops logger
logger = logging.getLogger("halodrops")
logger.setLevel(logging.DEBUG)

# File Handler
fh_info = logging.FileHandler("info.log")
fh_info.setLevel(logging.INFO)

fh_debug = logging.FileHandler("debug.log", mode="w")
fh_debug.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# Formatter
log_format = "{asctime}  {levelname:^8s} {name:^20s} {filename:^20s} Line:{lineno:03d}:\n{message}"
formatter = logging.Formatter(log_format, style="{")
fh_info.setFormatter(formatter)
fh_debug.setFormatter(formatter)
ch.setFormatter(formatter)

# Add file and streams handlers to the logger
logger.addHandler(fh_info)
logger.addHandler(fh_debug)
logger.addHandler(ch)

import inspect
import importlib
import pkgutil
import configparser


def get_all_defaults(package):
    """Retrieve the default values of all functions within a package and its subpackages.

    Parameters:
        package (module): The root package to inspect.

    Returns:
        dict: A dictionary where the keys are the fully qualified names of the functions,
              including all parent modules, and the values are dictionaries containing the
              default values of the function parameters.
    """
    function_values = {}

    def process_module(module, parent_module_names):
        """Process a module and retrieve the default values of its functions.

        Parameters:
            module (module): The module to process.
            parent_module_names (list): The names of the parent modules.

        Returns:
            None
        """
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                function_signature = inspect.signature(obj)
                defaults = {
                    param.name: param.default
                    for param in function_signature.parameters.values()
                    if param.default is not inspect.Parameter.empty
                }
                function_name = ".".join(parent_module_names + [obj.__qualname__])
                function_values[function_name] = defaults

    def process_package(package, parent_module_names):
        """Process a package and its subpackages, retrieving the default values of functions.

        Parameters:
            package (module): The package to process.
            parent_module_names (list): The names of the parent modules.

        Returns:
            None
        """
        package_path = package.__path__
        package_name = package.__name__
        module_infos = pkgutil.walk_packages(package_path, prefix=package_name + ".")

        for module_info in module_infos:
            module = importlib.import_module(module_info.name)
            module_name_parts = module_info.name.split(".")
            module_names = (
                parent_module_names + module_name_parts[1:]
            )  # Exclude root package name
            process_module(module, module_names)

    process_package(package, [])
    return function_values


def nondefault_values_from_config(config, default_dict):
    """Retrieve non-default argument values from a configuration file.

    Parameters:
        config (configparser.ConfigParser): The configuration object representing the config file.
        default_dict (dict): A dictionary containing the default values for functions.

    Returns:
        dict: A dictionary containing the non-default arguments and their corresponding values
              based on the config file.
    """
    function_defaults = {}
    for section in config.sections():
        if section in default_dict.keys():
            function_defaults[section] = {}
            for option, value in config.items(section):
                if option in default_dict[section]:
                    function_defaults[section][option] = value
            # remove empty dict if no args for section are found
            if not function_defaults[section]:
                del function_defaults[section]

    return function_defaults


def get_mandatory_args(function):
    """
    Get a list of all arguments that do not have a default value for each function in the list.

    Parameters
    ----------
    list_of_functions : list
        A list of functions to inspect.

    Returns
    -------
    list
        A list of argument names that do not have a default value.

    Examples
    --------
    >>> def func1(a, b=2):
    ...     pass
    >>> def func2(c, d=4, e=5):
    ...     pass
    >>> mandatory_args([func1, func2])
    ['a', 'c']
    """
    mandatory_args = []
    sig = inspect.signature(function)
    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            mandatory_args.append(name)
    return mandatory_args


def get_mandatory_values_from_config(config, mandatory_args):
    """
    Extracts mandatory values from the 'MANDATORY' section of a configuration file.

    Parameters
    ----------
    config : ConfigParser
        The configuration file parser.
    mandatory_args : list
        A list of argument names that are expected to be in the 'MANDATORY' section of the config file.

    Returns
    -------
    dict
        A dictionary where the keys are the argument names and the values are the corresponding values from the config file.

    Raises
    ------
    ValueError
        If the 'MANDATORY' section is not found in the config file or if a mandatory argument is not found in the 'MANDATORY' section.

    Examples
    --------
    >>> import configparser
    >>> config = configparser.ConfigParser()
    >>> config.read_string('[MANDATORY]\\narg1=value1\\narg2=value2')
    >>> mandatory_values_from_config(config, ['arg1', 'arg2'])
    {'arg1': 'value1', 'arg2': 'value2'}
    """
    if not config.has_section("MANDATORY"):
        raise ValueError(f"MANDATORY section not found in config file")
    else:
        mandatory_dict = {}
        for arg in mandatory_args:
            if config.has_option("MANDATORY", arg):
                mandatory_dict[arg] = config.get("MANDATORY", arg)
            else:
                raise ValueError(f"Mandatory argument {arg} not found in config file")
    return mandatory_dict


def main():
    import argparse
    import halodrops
    import halodrops.api.qc as qc

    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument(
        "-c",
        "--config_file_path",
        default="./halodrops.cfg",
        help="config file path for halodrops, "
        + "by default the config file is halodrops.cfg in the current directory."
        + "Otherwise path to directory and filename need to be defined",
    )

    args = parser.parse_args()
    import os

    config_file_path = args.config_file_path
    config_dirname = os.path.dirname(config_file_path)
    config_basename = os.path.basename(config_file_path)

    # check if given config file directory exists
    if not os.path.exists(config_dirname):
        raise FileNotFoundError(f"Directory {config_dirname} not found.")
    else:
        # check if config file exists inside
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(
                f"File {config_file_path} does not exist. Please check file name."
            )
        else:
            import configparser

            config = configparser.ConfigParser()
            config.read(config_file_path)

    default_dict = get_all_defaults(halodrops)
    nondefaults = nondefault_values_from_config(config, default_dict)

    list_of_functions = [qc.run, qc.run2]

    for function in list_of_functions:
        section_name = f"{function.__module__}.{function.__name__}".split("halodrops.")[
            1
        ]
        if section_name in nondefaults:
            nondefault_args = nondefaults[section_name]
        else:
            nondefault_args = {}

        mandatory = get_mandatory_args(function)
        if mandatory:
            mandatory_args = get_mandatory_values_from_config(config, mandatory)
            nondefault_args.update(mandatory_args)

        function(**nondefault_args)
