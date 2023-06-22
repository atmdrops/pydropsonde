import logging

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


def main():
    import argparse

    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument(
        "-c",
        "--configpath",
        default="./halodrops.cfg",
        help="config file path for halodrops, by default the config file is halodrops.cfg in the current directory",
    )

    args = parser.parse_args()
    import os

    if args.configpath[-3:] != "cfg":
        config_path = os.path.join(args.configpath, "halodrops.cfg")
    else:
        config_path = args.configpath

    if os.path.exists(config_path):
        import configparser

        config = configparser.ConfigParser()
        config.read(config_path)
    else:
        return print(
            f"{config_path}: Path does not exist. Please check config file location."
        )
