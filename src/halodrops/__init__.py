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
        "--config_file_path",
        default="./halodrops.cfg",
        help="config file path for halodrops, "+\
            "by default the config file is halodrops.cfg in the current directory."+\
            "Otherwise path to directory and filename need to be defined",
    )

    args = parser.parse_args()
    import os
    config_file_path=args.config_file_path
    #if args.config_file_path[-3:] != "cfg":
    #    config_file_path = os.path.join(args.config_file_path, "halodrops.cfg")
    #else:
    #    config_file_path = args.config_file_path
    config_dirname  = os.path.dirname(config_file_path)
    config_basename = os.path.basename(config_file_path)

    # check if given config file directory exists
    if not os.path.exists(config_dirname):
        raise FileNotFoundError("Directory ", config_dirname, "not found.")
    else:
        # check if config file exists inside
        if os.path.exists(config_file_path):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
        else:
            raise FileNotFoundError("File", config_file_path," does not exist. Please check file name .")
