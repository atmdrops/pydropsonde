from configparser import ConfigParser
from typing import Dict
from halodrops import (
    nondefault_values_from_config,
    get_mandatory_args,
    get_mandatory_values_from_config,
)
import pytest


@pytest.fixture
def sample_config() -> ConfigParser:
    """
    Fixture for creating a sample config object with test values.

    Returns:
        ConfigParser: A sample config object.
    """
    config = ConfigParser()
    config.add_section("function1")
    config.set("function1", "arg1", "value1")
    config.set("function1", "arg2", "value2")
    config.add_section("function2")
    config.set("function2", "arg1", "value3")
    config.add_section("function_not_in_default_dict")
    config.set("function_not_in_default_dict", "arg1", "value4")
    config.add_section("function_with_empty_section_in_config")
    return config


@pytest.fixture
def default_dict() -> Dict[str, Dict[str, str]]:
    """
    Fixture for creating a sample default dictionary.

    Returns:
        dict: A sample default dictionary.
    """
    return {
        "function1": {"arg1": "default1", "arg2": "default2"},
        "function2": {"arg1": "default3"},
        "function_with_empty_section_in_config": {
            "arg1": "default4",
            "arg2": "default5",
        },
    }


def test_nondefault_values_from_config_with_sample_config(
    sample_config: ConfigParser, default_dict: Dict[str, Dict[str, str]]
) -> None:
    """
    Test nondefault_values_from_config with a sample config object and default dictionary.

    Test the behavior of the nondefault_values_from_config function when provided with a sample
    config object and default dictionary. It checks if the function correctly identifies the
    non-default values in the config object based on the default dictionary.

    Parameters:
        sample_config (ConfigParser): A sample config object with test values.
        default_dict (dict): A sample default dictionary.

    Returns:
        None
    """
    result = nondefault_values_from_config(sample_config, default_dict)

    expected_result = {
        "function1": {"arg1": "value1", "arg2": "value2"},
        "function2": {"arg1": "value3"},
    }
    assert result == expected_result


def test_nondefault_values_from_config_with_section_missing_in_default_dict(
    sample_config: ConfigParser, default_dict: Dict[str, Dict[str, str]]
) -> None:
    """
    Test nondefault_values_from_config with a missing section in the default dictionary.

    Test the behavior of the nondefault_values_from_config function when provided with a sample
    config object and a default dictionary that is missing a section. It checks if the function
    correctly handles the case where a section is present in the config object but not in the
    default dictionary.

    Parameters:
        sample_config (ConfigParser): A sample config object with test values.
        default_dict (dict): A sample default dictionary.

    Returns:
        None
    """
    result = nondefault_values_from_config(sample_config, default_dict)

    assert "function_not_in_default_dict" not in result


def test_nondefault_values_from_config_with_section_having_no_arguments(
    sample_config: ConfigParser, default_dict: Dict[str, Dict[str, str]]
) -> None:
    """
    Test nondefault_values_from_config with missing arguments in the config file.

    Test the behavior of the nondefault_values_from_config function when provided with a sample
    config object and a default dictionary that is missing arguments. It checks if the function
    correctly handles the case where a section is present in the config object, and the default
    dictionary contains the section, but some arguments are missing in the config.

    Parameters:
        sample_config (ConfigParser): A sample config object with test values.
        default_dict (dict): A sample default dictionary.

    Returns:
        None
    """
    result = nondefault_values_from_config(sample_config, default_dict)

    assert "function_with_empty_section_in_config" not in result


def test_nondefault_values_from_config_with_empty_config(
    default_dict: Dict[str, Dict[str, str]]
) -> None:
    """
    Test nondefault_values_from_config with an empty config object.

    Test the behavior of the nondefault_values_from_config function when provided with an empty
    config object and a default dictionary. It checks if the function correctly handles the case
    where the config object is empty and returns an empty dictionary.

    Parameters:
        default_dict (dict): A sample default dictionary.

    Returns:
        None
    """
    empty_config = ConfigParser()
    result = nondefault_values_from_config(empty_config, default_dict)

    assert result == {}


def test_get_mandatory_args():
    """
    Test get_mandatory_args function.
    """

    def sample_function(arg1, arg2, arg3="default"):
        pass

    result = get_mandatory_args(sample_function)
    assert result == ["arg1", "arg2"]


def test_get_mandatory_values_from_config():
    """
    Test get_mandatory_values_from_config function.
    """
    mandatory_args = ["arg1", "arg2"]
    config = ConfigParser()
    config.add_section("MANDATORY")
    config.set("MANDATORY", "arg1", "value1")
    config.set("MANDATORY", "arg2", "value2")

    result = get_mandatory_values_from_config(config, mandatory_args)
    assert result == {"arg1": "value1", "arg2": "value2"}


def test_get_mandatory_values_from_config_missing_section():
    """
    Test get_mandatory_values_from_config function with missing MANDATORY section.
    """
    mandatory_args = ["arg1", "arg2"]
    config = ConfigParser()

    with pytest.raises(ValueError, match="MANDATORY section not found in config file"):
        get_mandatory_values_from_config(config, mandatory_args)


def test_get_mandatory_values_from_config_missing_arg():
    """
    Test get_mandatory_values_from_config function with missing mandatory argument.
    """
    mandatory_args = ["arg1", "arg2", "arg3"]
    config = ConfigParser()
    config.add_section("MANDATORY")
    config.set("MANDATORY", "arg1", "value1")
    config.set("MANDATORY", "arg2", "value2")

    with pytest.raises(
        ValueError, match="Mandatory argument arg3 not found in config file"
    ):
        get_mandatory_values_from_config(config, mandatory_args)
